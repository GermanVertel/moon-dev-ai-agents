import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# 🌙 MOON DEV FIBONACCI DIVERGENCE STRATEGY 🚀✨

data_path = '/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv'
data = pd.read_csv(data_path)

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Map to proper case
data.columns = [col.capitalize() for col in data.columns]
data['Datetime'] = pd.to_datetime(data['Datetime'])
data = data.set_index('Datetime')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
data = data.dropna()

print("🌙 Moon Dev Data Loaded! ✨")
print(f"🚀 Rows: {len(data)}")
print(data.head())


class FibonacciDivergence(Strategy):
    rsi_period = 14
    swing_window = 20
    atr_period = 14
    risk_pct = 0.01
    fib_tolerance = 0.02  # 2% tolerance around fib zone
    lookback = 5

    def init(self):
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=self.rsi_period)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close,
                          timeperiod=self.atr_period)
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=self.swing_window)
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=self.swing_window)
        print("🌙✨ Moon Dev indicators initialized! 🚀")

    def next(self):
        if len(self.data) < self.swing_window + self.lookback + 5:
            return

        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        rsi = self.rsi[-1]
        atr = self.atr[-1]

        if np.isnan(rsi) or np.isnan(atr) or atr <= 0:
            return

        sh = self.swing_high[-1]
        sl = self.swing_low[-1]

        if np.isnan(sh) or np.isnan(sl) or sh <= sl:
            return

        # Fibonacci levels from swing low -> swing high (for shorts)
        rng = sh - sl
        fib_382 = sh - rng * 0.382
        fib_500 = sh - rng * 0.500
        fib_618 = sh - rng * 0.618
        fib_786 = sh - rng * 0.786

        # 🌙 Bearish/Bullish divergence detection using array indexing (no slicing)
        lookback = self.lookback
        prev_high = -np.inf
        prev_low = np.inf
        prev_rsi_max = -np.inf
        prev_rsi_min = np.inf

        for i in range(2, lookback + 2):
            h = self.data.High[-i]
            l = self.data.Low[-i]
            r = self.rsi[-i]
            if not np.isnan(h) and h > prev_high:
                prev_high = h
            if not np.isnan(l) and l < prev_low:
                prev_low = l
            if not np.isnan(r):
                if r > prev_rsi_max:
                    prev_rsi_max = r
                if r < prev_rsi_min:
                    prev_rsi_min = r

        if prev_high == -np.inf or prev_low == np.inf or \
           prev_rsi_max == -np.inf or prev_rsi_min == np.inf:
            return

        bearish_div = (high >= prev_high) and (rsi < prev_rsi_max - 3)
        bullish_div = (low <= prev_low) and (rsi > prev_rsi_min + 3)

        # 🌙 Manage open trades
        if self.position:
            for trade in self.trades:
                if trade.is_short:
                    # TP1 at 50%, TP2 at 38.2% (short from 61.8 zone)
                    if price <= fib_500:
                        print(f"🌙✨ SHORT TP hit at {price:.2f} 🚀")
                        trade.close()
                    # Invalidation
                    elif price >= fib_786:
                        print(f"🌙❌ SHORT invalidated at {price:.2f}")
                        trade.close()
                elif trade.is_long:
                    if price >= fib_500:
                        print(f"🌙✨ LONG TP hit at {price:.2f} 🚀")
                        trade.close()
                    elif price <= fib_382:
                        print(f"🌙❌ LONG invalidated at {price:.2f}")
                        trade.close()
            return

        # 🌙 SHORT ENTRY
        in_golden_zone = (price <= fib_500 * (1 + self.fib_tolerance)) and \
                         (price >= fib_618 * (1 - self.fib_tolerance))
        bearish_candle = self.data.Close[-1] < self.data.Close[-2] and \
                         self.data.Close[-1] < self.data.Low[-2]

        if bearish_div and in_golden_zone and bearish_candle:
            stop = sh + 1.5 * atr
            risk_per_unit = stop - price
            if risk_per_unit <= 0:
                return
            risk_amount = self.equity * self.risk_pct
            size = int(round(risk_amount / risk_per_unit))
            if size < 1:
                return
            print(f"🌙🔻 MOON DEV SHORT! Price={price:.2f} RSI={rsi:.1f} "
                  f"Zone=[{fib_618:.2f}-{fib_500:.2f}] Stop={stop:.2f} Size={size} 🚀")
            self.sell(size=size, sl=stop)

        # 🌙 LONG ENTRY
        bullish_candle = self.data.Close[-1] > self.data.Close[-2] and \
                         self.data.Close[-1] > self.data.High[-2]
        if bullish_div and in_golden_zone and bullish_candle:
            stop = sl - 1.5 * atr
            risk_per_unit = price - stop
            if risk_per_unit <= 0:
                return
            risk_amount = self.equity * self.risk_pct
            size = int(round(risk_amount / risk_per_unit))
            if size < 1:
                return
            print(f"🌙🔺 MOON DEV LONG! Price={price:.2f} RSI={rsi:.1f} "
                  f"Zone=[{fib_618:.2f}-{fib_500:.2f}] Stop={stop:.2f} Size={size} 🚀")
            self.buy(size=size, sl=stop)


bt = Backtest(data, FibonacciDivergence, cash=1_000_000, commission=0.001)

print("🌙✨ Running Moon Dev Fibonacci Divergence Backtest... 🚀")
stats = bt.run()
print(stats)
print(stats._strategy)