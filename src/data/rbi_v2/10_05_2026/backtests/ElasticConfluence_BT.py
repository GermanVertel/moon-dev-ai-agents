import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Load data
data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Map columns to proper case
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

print("🌙✨ Moon Dev ElasticConfluence Backtest Starting! 🚀")
print(f"📊 Data loaded: {len(data)} bars")
print(f"📈 Columns: {list(data.columns)}")


class ElasticConfluence(Strategy):
    ema_fast = 10
    sma_slow = 50
    rsi_period = 14
    atr_period = 14
    vol_period = 20
    spread_lookback = 100
    spread_std_mult = 2.0
    vol_mult = 1.5
    rsi_oversold = 35
    rsi_overbought = 65
    atr_stop_mult = 2.0
    atr_trail_mult = 1.5
    atr_trail_trigger = 1.0
    risk_pct = 0.02
    div_lookback = 10

    def init(self):
        print("🌙 Initializing ElasticConfluence indicators... ✨")
        close = pd.Series(self.data.Close, index=self.data.index)
        high = pd.Series(self.data.High, index=self.data.index)
        low = pd.Series(self.data.Low, index=self.data.index)
        volume = pd.Series(self.data.Volume, index=self.data.index)

        self.ema_fast = self.I(talib.EMA, close, timeperiod=self.ema_fast)
        self.sma_slow = self.I(talib.SMA, close, timeperiod=self.sma_slow)
        self.rsi = self.I(talib.RSI, close, timeperiod=self.rsi_period)
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)
        self.vol_sma = self.I(talib.SMA, volume, timeperiod=self.vol_period)

        # Spread percentage
        def spread_calc(ema, sma):
            ema = np.asarray(ema, dtype=float)
            sma = np.asarray(sma, dtype=float)
            with np.errstate(divide='ignore', invalid='ignore'):
                s = (ema - sma) / sma * 100.0
            return s

        self.spread = self.I(spread_calc, self.ema_fast, self.sma_slow)

        # Rolling mean/std of spread
        def rolling_mean(arr, n):
            return pd.Series(arr).rolling(n).mean().values

        def rolling_std(arr, n):
            return pd.Series(arr).rolling(n).std().values

        self.spread_mean = self.I(rolling_mean, self.spread, self.spread_lookback)
        self.spread_std = self.I(rolling_std, self.spread, self.spread_lookback)

        # ATR average for volatility filter
        self.atr_avg = self.I(talib.SMA, self.atr, timeperiod=50)

        # Track trade state
        self.entry_price = None
        self.stop_price = None
        self.trail_active = False
        self.highest_since_entry = None

        print("🚀 Indicators ready! Let's find those reversals! 🌙")

    def _bullish_divergence(self, i):
        """Check for bullish RSI divergence: price lower low, RSI higher low."""
        lb = self.div_lookback
        if i < lb + 1:
            return False
        low_now = self.data.Low[i]
        low_prev = self.data.Low[i - lb]
        rsi_now = self.rsi[i]
        rsi_prev = self.rsi[i - lb]
        if np.isnan(rsi_now) or np.isnan(rsi_prev):
            return False
        # price makes lower low, RSI makes higher low
        price_ll = low_now < low_prev
        rsi_hl = rsi_now > rsi_prev
        # oversold context
        oversold = rsi_now < 45 or rsi_prev < 35
        return price_ll and rsi_hl and oversold

    def next(self):
        i = len(self.data) - 1
        if i < max(self.spread_lookback, self.sma_slow, 50) + 5:
            return

        price = self.data.Close[i]
        ema = self.ema_fast[i]
        sma = self.sma_slow[i]
        rsi = self.rsi[i]
        atr = self.atr[i]
        vol = self.data.Volume[i]
        vol_avg = self.vol_sma[i]
        spread = self.spread[i]
        spread_mean = self.spread_mean[i]
        spread_std = self.spread_std[i]
        atr_avg = self.atr_avg[i]

        # Skip if abnormal volatility
        if not np.isnan(atr_avg) and atr_avg > 0 and atr > 2.0 * atr_avg:
            return

        # Manage open position
        if self.position:
            # Update trailing stop
            if self.highest_since_entry is None or price > self.highest_since_entry:
                self.highest_since_entry = price

            if self.entry_price is not None and atr > 0:
                # Activate trailing once profit > 1x ATR
                if not self.trail_active and (self.highest_since_entry - self.entry_price) >= self.atr_trail_trigger * atr:
                    self.trail_active = True
                    print(f"🌙✨ Trailing stop activated at {price:.2f} 🚀")

                if self.trail_active:
                    new_stop = self.highest_since_entry - self.atr_trail_mult * atr
                    if new_stop > self.stop_price:
                        self.stop_price = new_stop

            # Stop loss
            if self.stop_price is not None and self.data.Low[i] <= self.stop_price:
                print(f"🛑 Stop loss hit at {self.stop_price:.2f} | Exit price ~{price:.2f}")
                self.position.close()
                self._reset_trade_state()
                return

            # Primary exit: EMA crosses back above SMA (spread >= 0)
            if spread >= 0:
                print(f"🎯 Equilibrium return! EMA crossed above SMA. Exit at {price:.2f}")
                self.position.close()
                self._reset_trade_state()
                return

            # RSI overbought exit
            if rsi >= self.rsi_overbought:
                print(f"💰 RSI overbought ({rsi:.1f}) — taking profit at {price:.2f}")
                self.position.close()
                self._reset_trade_state()
                return

            # Price touches SMA from below
            if price >= sma:
                print(f"📈 Price touched SMA50 ({sma:.2f}) — exit at {price:.2f}")
                self.position.close()
                self._reset_trade_state()
                return

            return

        # Entry logic
        if np.isnan(spread) or np.isnan(spread_std) or np.isnan(spread_mean):
            return
        if np.isnan(vol_avg) or vol_avg <= 0:
            return
        if np.isnan(atr) or atr <= 0:
            return

        lower_threshold = spread_mean - self.spread_std_mult * spread_std

        # Condition A: spread at/below lower stretch threshold
        cond_a = spread <= lower_threshold and spread < 0

        # Condition B: bullish RSI divergence
        cond_b = self._bullish_divergence(i)

        # Condition C: volume spike on bullish candle
        bullish_candle = self.data.Close[i] > self.data.Open[i]
        cond_c = (vol >= self.vol_mult * vol_avg) and bullish_candle

        if cond_a and cond_b and cond_c:
            stop_price = price - self.atr_stop_mult * atr
            risk_per_unit = price - stop_price
            if risk_per_unit <= 0:
                return

            equity = self.equity
            risk_amount = equity * self.risk_pct
            size = int(round(risk_amount / risk_per_unit))
            if size < 1:
                size = 1

            print(f"🌙🚀 ELASTIC CONFLUENCE LONG SIGNAL! 🚀🌙")
            print(f"   💫 Price: {price:.2f} | Spread: {spread:.3f}% (thr {lower_threshold:.3f})")
            print(f"   📊 RSI: {rsi:.1f} | Vol: {vol:.2f} vs avg {vol_avg:.2f}")
            print(f"   🛑 Stop: {stop_price:.2f} | ATR: {atr:.2f} | Size: {size}")

            self.buy(size=size)
            self.entry_price = price
            self.stop_price = stop_price
            self.trail_active = False
            self.highest_since_entry = price

    def _reset_trade_state(self):
        self.entry_price = None
        self.stop_price = None
        self.trail_active = False
        self.highest_since_entry = None


bt = Backtest(
    data,
    ElasticConfluence,
    cash=1_000_000,
    commission=0.002,
    exclusive_orders=True,
)

stats = bt.run()
print(stats)
print(stats._strategy)