import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

print("🌙✨ Moon Dev Backtest AI initializing... Volatility-Adjusted Reversion 🚀")

# Load data
data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

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

# Parse datetime index
if 'datetime' in data.columns:
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')

print(f"🌙 Data loaded: {len(data)} rows ✨")
print(f"🚀 Columns: {list(data.columns)}")


class VolatilityAdjustedReversion(Strategy):
    sma_period = 20
    atr_period = 3
    entry_threshold = 1.5
    exit_threshold = -0.5
    atr_stop_mult = 2.0
    risk_pct = 0.02

    def init(self):
        print("🌙 Initializing indicators... ✨")
        self.sma20 = self.I(talib.SMA, self.data.Close, timeperiod=self.sma_period)
        self.atr3 = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close,
                           timeperiod=self.atr_period)
        self.mrr = self.I(
            lambda c, s, a: (c - s) / np.where(a == 0, np.nan, a),
            self.data.Close, self.sma20, self.atr3
        )
        print("🚀 Indicators ready!")

    def next(self):
        price = self.data.Close[-1]
        mrr = self.mrr[-1]
        atr = self.atr3[-1]

        if np.isnan(mrr) or np.isnan(atr) or atr <= 0:
            return

        if not self.position:
            if mrr > self.entry_threshold:
                # Volatility-adjusted sizing
                risk_amount = self.equity * self.risk_pct
                stop_distance = self.atr_stop_mult * atr
                if stop_distance <= 0:
                    return
                size = int(round(risk_amount / stop_distance))
                if size < 1:
                    size = 1
                # Cap size to available equity (safety)
                max_size = int(self.equity / price)
                size = min(size, max_size) if max_size > 0 else 1
                if size < 1:
                    return

                stop_price = price - stop_distance
                print(f"🌙✨ LONG ENTRY! MRR={mrr:.2f} > {self.entry_threshold} | "
                      f"Price={price:.2f} | ATR={atr:.2f} | Size={size} | Stop={stop_price:.2f} 🚀")
                self.buy(size=size, sl=stop_price)
        else:
            if mrr < self.exit_threshold:
                print(f"🌙 EXIT SIGNAL! MRR={mrr:.2f} < {self.exit_threshold} | Price={price:.2f} ✨")
                self.position.close()


print("🌙 Running backtest... ✨🚀")
bt = Backtest(
    data,
    VolatilityAdjustedReversion,
    cash=1_000_000,
    commission=0.002,
    exclusive_orders=True
)

stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙✨ Backtest complete! 🚀")