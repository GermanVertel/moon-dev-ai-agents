import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# 🌙 MOON DEV BACKTEST - VOLUMETRIC SQUEEZE 🚀

class VolumetricSqueeze(Strategy):
    # Strategy parameters
    bb_period = 20
    bb_std = 2.0
    vol_period = 20
    vol_multiplier = 2.0
    rsi_period = 14
    rsi_exit = 70

    def init(self):
        print("🌙 Moon Dev initializing Volumetric Squeeze indicators... ✨")

        close = pd.Series(self.data.Close).astype(np.float64).values
        volume = pd.Series(self.data.Volume).astype(np.float64).values

        self.bb_middle = self.I(talib.SMA, close, timeperiod=self.bb_period)
        self.bb_stddev = self.I(talib.STDDEV, close, timeperiod=self.bb_period, nbdev=1)

        # 🌙 Wrap upper/lower BB in self.I with proper indicator functions
        def calc_upper(c, s):
            return c + self.bb_std * s

        def calc_lower(c, s):
            return c - self.bb_std * s

        self.bb_upper = self.I(calc_upper, close, self.bb_stddev)
        self.bb_lower = self.I(calc_lower, close, self.bb_stddev)

        self.vol_sma = self.I(talib.SMA, volume, timeperiod=self.vol_period)
        self.rsi = self.I(talib.RSI, close, timeperiod=self.rsi_period)

        self.rsi_was_above = False
        print("🚀 Moon Dev indicators ready! 🌙")

    def next(self):
        if len(self.data) < self.bb_period + 1:
            return

        price = self.data.Close[-1]
        upper = self.bb_upper[-1]
        middle = self.bb_middle[-1]
        volume = self.data.Volume[-1]
        vol_avg = self.vol_sma[-1]
        rsi = self.rsi[-1]

        if np.isnan(upper) or np.isnan(middle) or np.isnan(vol_avg) or np.isnan(rsi):
            return

        # Update RSI state
        if rsi > self.rsi_exit:
            self.rsi_was_above = True

        # 🚀 ENTRY LOGIC
        if not self.position:
            trigger_a = price > upper
            trigger_b = volume > self.vol_multiplier * vol_avg

            if trigger_a and trigger_b:
                signal_low = self.data.Low[-1]
                stop_price = max(signal_low, middle)
                if middle > signal_low:
                    stop_price = middle

                risk_per_unit = price - stop_price
                if risk_per_unit <= 0:
                    return

                # 🌙 Use fraction-based sizing (percentage of equity) to avoid unit issues
                # Risk 1% of equity per trade
                risk_pct = 0.01
                # Position size as fraction of equity such that risk = risk_pct
                size_frac = risk_pct / (risk_per_unit / price)
                # Cap at 95% of equity
                if size_frac > 0.95:
                    size_frac = 0.95
                if size_frac <= 0:
                    return

                print(f"🌙✨ MOON DEV ENTRY SIGNAL! Price: {price:.2f} > Upper BB: {upper:.2f}, Volume: {volume:.2f} > 2x Avg: {2*vol_avg:.2f}, RSI: {rsi:.2f} 🚀")
                print(f"   📏 Size fraction: {size_frac:.4f}, Stop: {stop_price:.2f}, Risk/unit: {risk_per_unit:.2f}")
                self.buy(size=size_frac, sl=stop_price)

        # 🛑 EXIT LOGIC
        else:
            if self.rsi_was_above and rsi < self.rsi_exit:
                print(f"🌙🛑 MOON DEV EXIT SIGNAL! RSI crossed below 70: {rsi:.2f} 🚀")
                self.position.close()
                self.rsi_was_above = False
            elif price < middle:
                print(f"🌙⚠️ MOON DEV HARD STOP! Price: {price:.2f} < Middle BB: {middle:.2f} 🚀")
                self.position.close()
                self.rsi_was_above = False

# 🌙 DATA LOADING AND PREPARATION
print("🌙 Moon Dev loading data from the cosmos... ✨")
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')

# Clean column names
data.columns = data.columns.str.strip().str.lower()
# Drop unnamed columns
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Ensure proper column mapping
data = data.rename(columns={
    'datetime': 'Datetime',
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

# Set datetime index
data = data.set_index(pd.to_datetime(data['Datetime']))
data = data.drop(columns=['Datetime'])

# Ensure numeric dtypes
for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    data[col] = pd.to_numeric(data[col], errors='coerce').astype(np.float64)

data = data.dropna()

print(f"🌙 Data loaded: {len(data)} rows from {data.index[0]} to {data.index[-1]} 🚀")

# 🚀 RUN BACKTEST
print("🌙 Moon Dev launching backtest... 🚀")
bt = Backtest(data, VolumetricSqueeze, cash=1_000_000, commission=0.002)

stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙✨ Moon Dev backtest complete! 🚀")