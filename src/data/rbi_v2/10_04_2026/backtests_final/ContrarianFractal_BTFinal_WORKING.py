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
    'datetime': 'Datetime',
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

data = data.set_index(pd.to_datetime(data['Datetime']))
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

print(f"🌙 Data loaded: {len(data)} bars ✨")


class ContrarianFractal(Strategy):
    fast_ma_period = 9
    slow_ma_period = 21
    atr_period = 14
    fractal_window = 5
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

        # Precompute fractals as arrays (vectorized) - no lookahead: confirmed at mid+2
        low_arr = np.asarray(low, dtype=float)
        high_arr = np.asarray(high, dtype=float)
        n = len(close)

        fractal_high = np.full(n, np.nan)
        fractal_low = np.full(n, np.nan)

        for m in range(2, n - 2):
            # Bullish fractal at m: low[m] is lowest of [m-2..m+2]
            if (low_arr[m] < low_arr[m-1] and low_arr[m] < low_arr[m-2] and
                low_arr[m] < low_arr[m+1] and low_arr[m] < low_arr[m+2]):
                fractal_low[m] = low_arr[m]
            # Bearish fractal at m: high[m] is highest of [m-2..m+2]
            if (high_arr[m] > high_arr[m-1] and high_arr[m] > high_arr[m-2] and
                high_arr[m] > high_arr[m+1] and high_arr[m] > high_arr[m+2]):
                fractal_high[m] = high_arr[m]

        # Shift by +2 so fractal is only visible 2 bars after it forms (no lookahead)
        fractal_high_shifted = np.full(n, np.nan)
        fractal_low_shifted = np.full(n, np.nan)
        fractal_high_shifted[2:] = fractal_high[:-2]
        fractal_low_shifted[2:] = fractal_low[:-2]

        self.fractal_high = self.I(lambda: fractal_high_shifted, name="FractalHigh")
        self.fractal_low = self.I(lambda: fractal_low_shifted, name="FractalLow")

        print("🚀 Indicators ready! Let's hunt fakeouts! 🌙")

    def next(self):
        # Need enough bars
        if len(self.data) < 25:
            return

        # Skip if we already have a position
        if self.position:
            return

        # MA crossover detection
        fast_now = self.fast_ma[-1]
        fast_prev = self.fast_ma[-2]
        slow_now = self.slow_ma[-1]
        slow_prev = self.slow_ma[-2]

        if any(np.isnan([fast_now, fast_prev, slow_now, slow_prev])):
            return

        bullish_cross = fast_prev <= slow_prev and fast_now > slow_now
        bearish_cross = fast_prev >= slow_prev and fast_now < slow_now

        if not (bullish_cross or bearish_cross):
            return

        # Look for a fractal within the confirmation window (recent bars)
        recent_bearish_fractal = False
        recent_bullish_fractal = False
        bearish_fractal_high = None
        bullish_fractal_low = None

        for k in range(0, self.fractal_window + 1):
            idx = -1 - k
            if abs(idx) > len(self.fractal_high):
                continue
            fh = self.fractal_high[idx]
            fl = self.fractal_low[idx]
            if not np.isnan(fh):
                recent_bearish_fractal = True
                bearish_fractal_high = fh
                break
            if not np.isnan(fl):
                recent_bullish_fractal = True
                bullish_fractal_low = fl
                break

        atr_val = self.atr[-1]
        if np.isnan(atr_val) or atr_val <= 0:
            return

        # Contrarian SHORT: bullish MA cross + bearish fractal nearby
        if bullish_cross and recent_bearish_fractal:
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

            # Ensure size doesn't exceed available cash
            max_size = int(equity / entry_price)
            if size > max_size:
                size = max_size
            if size < 1:
                return

            print(f"🌙🔻 CONTRARIAN SHORT! Fakeout detected! Entry={entry_price:.2f} SL={stop_price:.2f} TP={tp_price:.2f} Size={size} 🚀")
            self.sell(size=size, sl=stop_price, tp=tp_price)

        # Contrarian LONG: bearish MA cross + bullish fractal nearby
        elif bearish_cross and recent_bullish_fractal:
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

            # Ensure size doesn't exceed available cash
            max_size = int(equity / entry_price)
            if size > max_size:
                size = max_size
            if size < 1:
                return

            print(f"🌙🔺 CONTRARIAN LONG! Fakeout detected! Entry={entry_price:.2f} SL={stop_price:.2f} TP={tp_price:.2f} Size={size} 🚀")
            self.buy(size=size, sl=stop_price, tp=tp_price)


print("🌙✨ Running Contrarian Fractal Backtest... 🚀")
bt = Backtest(data, ContrarianFractal, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙✨ Backtest complete! Moon Dev out! 🚀")