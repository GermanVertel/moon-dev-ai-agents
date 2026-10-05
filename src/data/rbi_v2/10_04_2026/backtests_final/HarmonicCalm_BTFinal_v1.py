import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from datetime import datetime, time

# 🌙 Moon Dev's HarmonicCalm Strategy ✨

def load_data(path):
    print("🌙 Loading data from the cosmos...")
    data = pd.read_csv(path)
    data.columns = data.columns.str.strip().str.lower()
    data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
    # Map to backtesting.py required columns
    data = data.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume',
    })
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    print(f"✨ Data loaded: {len(data)} bars from {data.index[0]} to {data.index[-1]}")
    return data


class HarmonicCalm(Strategy):
    # Strategy parameters
    atr_period = 14
    atr_ma_period = 20
    swing_lookback = 50  # bars to find swing high/low
    risk_pct = 0.02  # 2% risk per trade
    atr_sl_mult = 1.0
    atr_tp_mult = 1.5
    max_bars_in_trade = 10

    def init(self):
        print("🌙 Initializing HarmonicCalm indicators...")
        # ATR for volatility
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close,
                          timeperiod=self.atr_period)
        self.atr_ma = self.I(talib.SMA, self.atr, timeperiod=self.atr_ma_period)

        # Swing high/low for Fibonacci
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=self.swing_lookback)
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=self.swing_lookback)

        # Higher timeframe trend proxy: 50 EMA on same data
        self.ema50 = self.I(talib.EMA, self.data.Close, timeperiod=50)

        # Candlestick patterns
        self.hammer = self.I(talib.CDLHAMMER, self.data.Open, self.data.High, self.data.Low, self.data.Close)
        self.shooting_star = self.I(talib.CDLSHOOTINGSTAR, self.data.Open, self.data.High, self.data.Low, self.data.Close)
        self.bull_engulf = self.I(talib.CDLENGULFING, self.data.Open, self.data.High, self.data.Low, self.data.Close)
        self.bear_engulf = self.I(talib.CDLENGULFING, self.data.Open, self.data.High, self.data.Low, self.data.Close)

        # Track entry bar for time-based exit
        self.entry_bar = None
        self.entry_price = None
        self.sl_price = None
        self.tp_price = None
        self.trade_dir = None

        print("✨ Indicators ready!")

    def _is_low_vol(self):
        if np.isnan(self.atr[-1]) or np.isnan(self.atr_ma[-1]):
            return False
        return self.atr[-1] < self.atr_ma[-1]

    def _is_quiet_hour(self):
        # GMT+2 hours: 22:00 - 02:00 (wraps midnight) OR 12:00 - 14:00
        hour = self.data.index[-1].hour
        # Assume data timestamps are UTC; shift +2 for GMT+2
        gmt2_hour = (hour + 2) % 24
        return (gmt2_hour >= 22 or gmt2_hour < 2) or (12 <= gmt2_hour < 14)

    def _fib_levels(self):
        sh = self.swing_high[-1]
        sl = self.swing_low[-1]
        if np.isnan(sh) or np.isnan(sl) or sh <= sl:
            return None
        diff = sh - sl
        return {
            '0.236': sh - 0.236 * diff,
            '0.382': sh - 0.382 * diff,
            '0.500': sh - 0.500 * diff,
            '0.618': sh - 0.618 * diff,
            '0.786': sh - 0.786 * diff,
        }

    def _near_fib(self, price, fibs, tol_pct=0.002):
        for name, lvl in fibs.items():
            if abs(price - lvl) / lvl <= tol_pct:
                return name, lvl
        return None, None

    def next(self):
        # Skip if already in a position
        if self.position:
            # Time-based exit
            if self.entry_bar is not None and (len(self.data) - self.entry_bar) >= self.max_bars_in_trade:
                print(f"⏰ Moon Dev Time exit after {self.max_bars_in_trade} bars 🌙")
                self.position.close()
                self.entry_bar = None
            return

        if len(self.data) < self.swing_lookback + 5:
            return

        # Conditions
        if not self._is_quiet_hour():
            return
        if not self._is_low_vol():
            return

        fibs = self._fib_levels()
        if fibs is None:
            return

        price = self.data.Close[-1]
        low = self.data.Low[-1]
        high = self.data.High[-1]
        open_ = self.data.Open[-1]
        atr_val = self.atr[-1]
        if np.isnan(atr_val) or atr_val <= 0:
            return

        level_name, level_price = self._near_fib(price, fibs)
        if level_name is None:
            return

        # Bullish confirmation
        bull_pattern = (self.hammer[-1] != 0) or (self.bull_engulf[-1] > 0)
        bear_pattern = (self.shooting_star[-1] != 0) or (self.bear_engulf[-1] < 0)

        # Trend alignment
        uptrend = price > self.ema50[-1]
        downtrend = price < self.ema50[-1]

        # LONG: touch fib from below + bullish confirmation + uptrend
        if low <= level_price and price >= level_price and bull_pattern and uptrend:
            sl = price - self.atr_sl_mult * atr_val
            tp = price + self.atr_tp_mult * atr_val
            risk_per_unit = price - sl
            if risk_per_unit <= 0:
                return
            equity = self.equity
            risk_amount = equity * self.risk_pct
            size = risk_amount / price  # fraction of equity
            if size <= 0 or size >= 1:
                size = 0.5
            print(f"🚀🌙 LONG at Fib {level_name} ({level_price:.2f}) | price={price:.2f} SL={sl:.2f} TP={tp:.2f} size={size:.4f}")
            self.buy(size=size, sl=sl, tp=tp)
            self.entry_bar = len(self.data)
            self.entry_price = price
            self.trade_dir = 'long'

        # SHORT: touch fib from above + bearish confirmation + downtrend
        elif high >= level_price and price <= level_price and bear_pattern and downtrend:
            sl = price + self.atr_sl_mult * atr_val
            tp = price - self.atr_tp_mult * atr_val
            risk_per_unit = sl - price
            if risk_per_unit <= 0:
                return
            equity = self.equity
            risk_amount = equity * self.risk_pct
            size = risk_amount / price  # fraction of equity
            if size <= 0 or size >= 1:
                size = 0.5
            print(f"🔻🌙 SHORT at Fib {level_name} ({level_price:.2f}) | price={price:.2f} SL={sl:.2f} TP={tp:.2f} size={size:.4f}")
            self.sell(size=size, sl=sl, tp=tp)
            self.entry_bar = len(self.data)
            self.entry_price = price
            self.trade_dir = 'short'


# 🌙 Run the backtest
print("🌙✨ Starting Moon Dev's HarmonicCalm Backtest ✨🌙")
data = load_data('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')

bt = Backtest(data, HarmonicCalm, cash=1_000_000, commission=0.0002, exclusive_orders=True)
stats = bt.run()
print(stats)
print(stats._strategy)