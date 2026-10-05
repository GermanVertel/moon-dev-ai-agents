import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

print("🌙 Moon Dev's VolatilityOversold Backtest Loading... ✨🚀")

data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Rename datetime first if present
if 'datetime' in data.columns:
    data = data.rename(columns={'datetime': 'Datetime'})

data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume',
})

if 'Datetime' in data.columns:
    data['Datetime'] = pd.to_datetime(data['Datetime'])
    data = data.set_index('Datetime')

data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
data = data.dropna()

print(f"🌙 Data loaded: {len(data)} rows ✨")
print(f"🚀 Columns: {list(data.columns)}")


class VolatilityOversold(Strategy):
    rsi_period = 2
    rsi_threshold = 10
    env_period = 20
    env_mult = 2.0
    profit_target = 0.05
    trailing_stop = 0.02
    risk_per_trade = 0.01
    size = 0.95  # ✅ Fraction of equity (0 < size < 1)

    def init(self):
        # ✅ All indicators via talib, wrapped in self.I()
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=self.rsi_period)
        self.sma = self.I(talib.SMA, self.data.Close, timeperiod=self.env_period)
        self.std = self.I(talib.STDDEV, self.data.Close, timeperiod=self.env_period, nbdev=1)

        # ✅ Upper volatility band via numpy lambda (NO backtesting.lib)
        self.upper_band = self.I(
            lambda s, sd: s + self.env_mult * sd,
            self.sma, self.std
        )

        self.entry_price = None
        self.highest_price = None

        print("🌙✨ Indicators initialized: RSI(2), SMA(20), Upper Volatility Band 🚀")

    def next(self):
        price = self.data.Close[-1]

        if len(self.data) < 25:
            return

        if self.position:
            # ✅ Use self.trades[-1].entry_price instead of self.entry_price
            if self.entry_price is None and len(self.trades) > 0:
                self.entry_price = self.trades[-1].entry_price

            if self.entry_price is None:
                return

            if self.highest_price is None:
                self.highest_price = self.entry_price

            if price > self.highest_price:
                self.highest_price = price

            gain = (price - self.entry_price) / self.entry_price
            trail = (price - self.highest_price) / self.highest_price

            if gain >= self.profit_target:
                print(f"🌙💰 PROFIT TARGET HIT at {price:.2f} | Gain: {gain*100:.2f}% ✨")
                self.position.close()
                self.entry_price = None
                self.highest_price = None
                return

            if trail <= -self.trailing_stop:
                print(f"🌙🛑 TRAILING STOP HIT at {price:.2f} | Trail: {trail*100:.2f}% ✨")
                self.position.close()
                self.entry_price = None
                self.highest_price = None
                return

        else:
            rsi_now = self.rsi[-1]
            rsi_prev = self.rsi[-2]

            if np.isnan(rsi_now) or np.isnan(rsi_prev):
                return

            # ✅ Manual crossover detection (NO backtesting.lib.crossover)
            cond_a = rsi_prev >= self.rsi_threshold and rsi_now < self.rsi_threshold

            upper = self.upper_band[-1]
            if np.isnan(upper):
                return
            cond_b = price > upper

            if len(self.data) < 5:
                return

            v1 = self.data.Volume[-1]
            v2 = self.data.Volume[-2]
            v3 = self.data.Volume[-3]
            v4 = self.data.Volume[-4]

            cond_c = (v1 < v2) and (v2 < v3) and (v3 < v4)

            # 🌙 DEBUG: Track how close we are to signal
            if cond_a and cond_b:
                print(f"🌙⚠️ RSI+Envelope met but Volume declining={cond_c} | "
                      f"V: {v4:.2f}->{v3:.2f}->{v2:.2f}->{v1:.2f}")

            if cond_a and cond_b and cond_c:
                print(f"🌙🚀 LONG SIGNAL! RSI crossed below {self.rsi_threshold} "
                      f"({rsi_prev:.2f}->{rsi_now:.2f}) | "
                      f"Price {price:.2f} > Upper {upper:.2f} | Declining Vol ✨")
                self.buy(size=self.size)
                self.entry_price = price
                self.highest_price = price


bt = Backtest(data, VolatilityOversold, cash=10_000_000, commission=0.001)

print("🌙✨ Running VolatilityOversold Backtest... 🚀")
stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙✨ Backtest Complete! 🚀")