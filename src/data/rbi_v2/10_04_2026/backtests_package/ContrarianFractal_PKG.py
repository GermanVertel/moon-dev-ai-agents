import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

print("🌙✨ Contrarian Fractal Strategy Loading... 🚀")

# Load data
data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper column mapping
data = data.rename(columns={
    'datetime': 'Date',
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

data['Date'] = pd.to_datetime(data['Date'])
data = data.set_index('Date')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

print(f"🌙 Data loaded: {len(data)} bars ✨")


class ContrarianFractal(Strategy):
    fast_ma_period = 9
    slow_ma_period = 21
    atr_period = 14
    fractal_window = 3  # bars window for MA cross + fractal confirmation
    risk_pct = 0.02
    rr_ratio = 2.0
    stop_buffer_atr = 0.2

    def init(self):
        print("🌙 Initializing Contrarian Fractal indicators... ✨")

        close = self.data.Close
        high = self.data.High
        low = self.data.Low

        # EMAs
        self.fast_ma = self.I(talib.EMA, close, timeperiod=self.fast_ma_period, name="FastEMA")
        self.slow_ma = self.I(talib.EMA, close, timeperiod=self.slow_ma_period, name="SlowEMA")

        # ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period, name="ATR")

        # Williams Fractals - using rolling max/min via talib.MAX/MIN
        # Bearish fractal: middle bar highest high of 5 bars
        # Bullish fractal: middle bar lowest low of 5 bars
        self.high5 = self.I(talib.MAX, high, timeperiod=5, name="High5")
        self.low5 = self.I(talib.MIN, low, timeperiod=5, name="Low5")

        self.fractal_high = np.full(len(close), np.nan)
        self.fractal_low = np.full(len(close), np.nan)

        print("🚀 Indicators ready! Let's hunt fakeouts! 🌙")

    def next(self):
        # Need enough bars
        if len(self.data) < 25:
            return

        i = len(self.data) - 1

        # Compute fractal confirmation at bar i-2 (middle bar of 5-bar pattern)
        # Bullish fractal at bar m: low[m] is lowest of bars [m-2..m+2]
        # Confirmed at bar m+2. So at current bar i, confirmed fractal mid is at i-2.
        mid = i - 2

        if mid < 2:
            return

        # Update fractal arrays dynamically
        # Bullish fractal at mid
        if (self.data.Low[mid] < self.data.Low[mid-1] and
            self.data.Low[mid] < self.data.Low[mid-2] and
            self.data.Low[mid] < self.data.Low[mid+1] and
            self.data.Low[mid] < self.data.Low[mid+2]):
            self.fractal_low[mid] = self.data.Low[mid]

        # Bearish fractal at mid
        if (self.data.High[mid] > self.data.High[mid-1] and
            self.data.High[mid] > self.data.High[mid-2] and
            self.data.High[mid] > self.data.High[mid+1] and
            self.data.High[mid] > self.data.High[mid+2]):
            self.fractal_high[mid] = self.data.High[mid]

        # MA crossover detection at bar i
        if len(self.fast_ma) < 2:
            return

        fast_now = self.fast_ma[-1]
        fast_prev = self.fast_ma[-2]
        slow_now = self.slow_ma[-1]
        slow_prev = self.slow_ma[-2]

        if any(np.isnan([fast_now, fast_prev, slow_now, slow_prev])):
            return

        bullish_cross = fast_prev <= slow_prev and fast_now > slow_now
        bearish_cross = fast_prev >= slow_prev and fast_now < slow_now

        # Look for a fractal within the confirmation window
        recent_bearish_fractal = False
        recent_bullish_fractal = False
        bearish_fractal_high = None
        bullish_fractal_low = None

        for k in range(0, self.fractal_window + 1):
            idx = mid - k
            if idx < 0 or idx >= len(self.data):
                continue
            if not np.isnan(self.fractal_high[idx]):
                recent_bearish_fractal = True
                bearish_fractal_high = self.fractal_high[idx]
                break
            if not np.isnan(self.fractal_low[idx]):
                recent_bullish_fractal = True
                bullish_fractal_low = self.fractal_low[idx]
                break

        atr_val = self.atr[-1]
        if np.isnan(atr_val) or atr_val <= 0:
            return

        # Contrarian SHORT: bullish MA cross + bearish fractal nearby
        if bullish_cross and recent_bearish_fractal and not self.position:
            entry_price = self.data.Close[-1]
            stop_price = bearish_fractal_high + self.stop_buffer_atr * atr_val
            risk_per_unit = stop_price - entry_price
            if risk_per_unit <= 0:
                return
            tp_price = entry_price - self.rr_ratio * risk_per_unit

            equity = self.equity
            risk_amount = equity * self.risk_pct
            size = int(round(risk_amount / risk_per_unit))
            if size < 1:
                size = 1

            print(f"🌙🔻 CONTRARIAN SHORT! Fakeout detected! Entry={entry_price:.2f} SL={stop_price:.2f} TP={tp_price:.2f} Size={size} 🚀")
            self.sell(size=size, sl=stop_price, tp=tp_price)

        # Contrarian LONG: bearish MA cross + bullish fractal nearby
        elif bearish_cross and recent_bullish_fractal and not self.position:
            entry_price = self.data.Close[-1]
            stop_price = bullish_fractal_low - self.stop_buffer_atr * atr_val
            risk_per_unit = entry_price - stop_price
            if risk_per_unit <= 0:
                return
            tp_price = entry_price + self.rr_ratio * risk_per_unit

            equity = self.equity
            risk_amount = equity * self.risk_pct
            size = int(round(risk_amount / risk_per_unit))
            if size < 1:
                size = 1

            print(f"🌙🔺 CONTRARIAN LONG! Fakeout detected! Entry={entry_price:.2f} SL={stop_price:.2f} TP={tp_price:.2f} Size={size} 🚀")
            self.buy(size=size, sl=stop_price, tp=tp_price)


print("🌙✨ Running Contrarian Fractal Backtest... 🚀")
bt = Backtest(data, ContrarianFractal, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙✨ Backtest complete! Moon Dev out! 🚀")