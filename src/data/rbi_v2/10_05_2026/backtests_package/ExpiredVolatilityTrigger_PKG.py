import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

print("🌙✨ Moon Dev Backtest AI initializing ExpiredVolatilityTrigger... 🚀")

data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper column mapping
data.columns = [c.capitalize() for c in data.columns]
if 'Datetime' in data.columns:
    data = data.set_index('Datetime')
    data.index = pd.to_datetime(data.index)

print(f"🌙 Data loaded: {len(data)} rows ✨")
print(f"🌙 Columns: {list(data.columns)} 🚀")


class ExpiredVolatilityTrigger(Strategy):
    atr_period = 14
    vol_period = 20
    vol_multiplier = 0.25
    tp_atr_mult = 2.0
    sl_atr_mult = 1.0
    risk_pct = 0.20

    def init(self):
        print("🌙✨ Initializing indicators... 🚀")
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close,
                          timeperiod=self.atr_period)

        # Daily volatility proxy: rolling std of returns (pure numpy/pandas)
        close = pd.Series(self.data.Close)
        returns = close.pct_change().values

        def rolling_std(x):
            return pd.Series(x).rolling(self.vol_period).std().values

        self.daily_vol = self.I(rolling_std, returns, name='DailyVol')

        self.entry_price = None
        self.stop_price = None
        self.tp_price = None

    def next(self):
        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        atr = self.atr[-1]
        dvol = self.daily_vol[-1]

        if np.isnan(atr) or np.isnan(dvol) or atr <= 0:
            return

        # Exit logic
        if self.position:
            if self.data.High[-1] >= self.tp_price:
                print(f"🌙✨ TP HIT at {self.data.High[-1]:.2f} (target {self.tp_price:.2f}) 🚀")
                self.position.close()
                return
            if self.data.Low[-1] <= self.stop_price:
                print(f"🌙🛑 SL HIT at {self.data.Low[-1]:.2f} (stop {self.stop_price:.2f}) 💥")
                self.position.close()
                return
            return

        # Entry logic
        # Contract age condition assumed satisfied (synthetic expired contract data)
        range_spread = (high - low) - atr
        threshold = dvol * price * self.vol_multiplier

        if range_spread >= threshold:
            size = int(round((self.equity * self.risk_pct) / price))
            if size <= 0:
                return
            self.entry_price = price
            self.stop_price = price - self.sl_atr_mult * atr
            self.tp_price = price + self.tp_atr_mult * atr
            print(f"🌙🚀 LONG ENTRY at {price:.2f} | ATR={atr:.2f} | spread={range_spread:.2f} >= thresh={threshold:.2f} | size={size} ✨")
            self.buy(size=size)


bt = Backtest(data, ExpiredVolatilityTrigger, cash=1_000_000, commission=0.002)
stats = bt.run()
print(stats)
print(stats._strategy)