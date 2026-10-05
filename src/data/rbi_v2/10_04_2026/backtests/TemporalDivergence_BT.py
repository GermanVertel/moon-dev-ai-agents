import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Load data
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper column mapping
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

# Ensure datetime index
if 'datetime' in data.columns:
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')

print("🌙✨ Moon Dev TemporalDivergence Backtest Initialized ✨🌙")
print(f"📊 Data loaded: {len(data)} bars")
print(f"📅 Date range: {data.index[0]} to {data.index[-1]}")
print(f"💫 Starting capital: $1,000,000")


class TemporalDivergence(Strategy):
    """
    TemporalDivergence Strategy 🌙
    Long near-dated futures momentum + short far-dated options premium decay
    (proxy: using price momentum divergence on a single series as a synthetic spread)
    """
    # Parameters
    roc_period = 10
    ema_short = 20
    ema_long = 50
    atr_period = 14
    corr_window = 30
    spread_lookback = 20

    # Risk params
    risk_pct = 0.02
    atr_stop_mult = 2.0
    trail_atr_mult = 3.0
    max_dd_pct = 0.03

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low

        # Momentum / ROC on the "futures" leg (price)
        self.roc_fut = self.I(talib.ROC, close, timeperiod=self.roc_period)

        # EMA trend filters
        self.ema_fast = self.I(talib.EMA, close, timeperiod=self.ema_short)
        self.ema_slow = self.I(talib.EMA, close, timeperiod=self.ema_long)

        # ATR for stops
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        # Swing low for stop
        self.swing_low = self.I(talib.MIN, low, timeperiod=20)

        # Options premium proxy: use a smoothed lagged price to simulate far-dated contract
        # Options premium tends to be smoother (smoothed price) and decays toward spot
        premium = pd.Series(close).rolling(10).mean().values
        premium = np.where(np.isnan(premium), close, premium)
        self.options_premium = self.I(lambda x: x, premium)

        # ROC on options premium (decay momentum)
        self.roc_opt = self.I(talib.ROC, self.options_premium, timeperiod=self.roc_period)

        # EMA on options premium
        self.ema_opt = self.I(talib.EMA, self.options_premium, timeperiod=self.ema_short)

        # Spread oscillator: normalized futures - premium
        spread = pd.Series(close).values - pd.Series(self.options_premium).values
        spread_series = pd.Series(spread)
        spread_norm = (spread_series - spread_series.rolling(self.spread_lookback).mean()) / (
            spread_series.rolling(self.spread_lookback).std().replace(0, np.nan)
        )
        spread_norm = spread_norm.fillna(0).values
        self.spread_osc = self.I(lambda x: x, spread_norm)

        # Spread rate of change (widening / narrowing)
        self.spread_roc = self.I(talib.ROC, self.spread_osc, timeperiod=5)

        # Rolling correlation between futures and options premium
        corr = pd.Series(close).rolling(self.corr_window).corr(pd.Series(self.options_premium))
        corr = corr.fillna(0).values
        self.correlation = self.I(lambda x: x, corr)

        # Track equity peak for drawdown guard
        self._peak_equity = self.equity
        self._entry_price = None
        self._trail_stop = None

        print("🌙 Indicators initialized: ROC, EMA, ATR, Spread Oscillator, Correlation ✨")

    def next(self):
        price = self.data.Close[-1]

        # Track peak equity for drawdown cap
        if self.equity > self._peak_equity:
            self._peak_equity = self.equity

        # Drawdown cap check
        if self._peak_equity > 0:
            dd = (self._peak_equity - self.equity) / self._peak_equity
            if dd > self.max_dd_pct and self.position:
                print(f"🛑 Moon Dev DD Cap hit ({dd*100:.2f}%) — closing position")
                self.position.close()
                self._entry_price = None
                self._trail_stop = None
                return

        # Skip if indicators not ready
        if (np.isnan(self.roc_fut[-1]) or np.isnan(self.roc_opt[-1]) or
                np.isnan(self.ema_fast[-1]) or np.isnan(self.ema_slow[-1]) or
                np.isnan(self.atr[-1]) or np.isnan(self.spread_osc[-1]) or
                np.isnan(self.correlation[-1])):
            return

        roc_fut = self.roc_fut[-1]
        roc_opt = self.roc_opt[-1]
        ema_f = self.ema_fast[-1]
        ema_s = self.ema_slow[-1]
        ema_o = self.ema_opt[-1]
        premium = self.options_premium[-1]
        atr = self.atr[-1]
        spread_osc = self.spread_osc[-1]
        spread_roc = self.spread_roc[-1]
        corr = self.correlation[-1]
        swing_low = self.swing_low[-1]

        # =====================
        # Position Management
        # =====================
        if self.position:
            # Profit target: momentum convergence (both legs same direction)
            convergence = (roc_fut > 0 and roc_opt > 0) or (roc_fut < 0 and roc_opt < 0)
            if convergence:
                print(f"🎯 Moon Dev convergence exit @ {price:.2f} (fut_roc={roc_fut:.2f}, opt_roc={roc_opt:.2f})")
                self.position.close()
                self._entry_price = None
                self._trail_stop = None
                return

            # ATR trailing stop on long leg
            if self._entry_price is not None:
                new_trail = price - self.trail_atr_mult * atr
                if self._trail_stop is None or new_trail > self._trail_stop:
                    self._trail_stop = new_trail

                if price < self._trail_stop:
                    print(f"📉 Moon Dev trailing ATR stop @ {price:.2f} (trail={self._trail_stop:.2f})")
                    self.position.close()
                    self._entry_price = None
                    self._trail_stop = None
                    return

            # Hard stop: futures breaks recent swing low
            if price < swing_low:
                print(f"🛑 Moon Dev swing-low stop @ {price:.2f} (swing_low={swing_low:.2f})")
                self.position.close()
                self._entry_price = None
                self._trail_stop = None
                return

            return

        # =====================
        # Entry Logic — Long futures / short options divergence
        # =====================
        # Divergence: futures momentum positive, options premium momentum negative
        divergence_long = roc_fut > 0 and roc_opt < 0

        # Trend confirmation: price above fast EMA, premium below its EMA
        trend_confirm = price > ema_f and premium < ema_o

        # Regime filter: spread widening upward
        spread_widening = spread_osc > 0 and spread_roc > 0

        # Correlation check: legs must be reasonably correlated
        corr_ok = corr > 0.3

        # Avoid aligned momentum
        aligned = (roc_fut > 0 and roc_opt > 0) or (roc_fut < 0 and roc_opt < 0)

        if (divergence_long and trend_confirm and spread_widening and corr_ok and not aligned):
            # Risk-based sizing with 1,000,000 base
            risk_amount = self.equity * self.risk_pct
            stop_distance = self.atr_stop_mult * atr
            if stop_distance <= 0:
                return

            raw_size = risk_amount / stop_distance
            size = int(round(raw_size))
            if size < 1:
                size = 1

            print(f"🚀🌙 Moon Dev LONG futures / SHORT options entry @ {price:.2f}")
            print(f"   fut_roc={roc_fut:.2f} opt_roc={roc_opt:.2f} spread_osc={spread_osc:.2f} "
                  f"spread_roc={spread_roc:.2f} corr={corr:.2f} size={size}")

            self.buy(size=size)
            self._entry_price = price
            self._trail_stop = price - self.trail_atr_mult * atr


# Run backtest
bt = Backtest(
    data,
    TemporalDivergence,
    cash=1_000_000,
    commission=0.0005,
    exclusive=False,
    trade_on_close=False
)

stats = bt.run()
print("\n🌙✨🌙✨ Moon Dev TemporalDivergence Full Stats ✨🌙✨🌙\n")
print(stats)
print("\n🌙✨🌙✨ Strategy Details ✨🌙✨🌙\n")
print(stats._strategy)