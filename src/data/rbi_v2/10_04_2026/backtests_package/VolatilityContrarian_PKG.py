import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# 🌙 Moon Dev's VolatilityContrarian Backtest 🚀

class VolatilityContrarian(Strategy):
    # Strategy parameters
    lookback = 252          # rolling window for mean/std
    long_z_entry = -3.0     # long entry threshold
    long_z_max = -4.0       # deepest long threshold
    short_z_entry = 2.0     # short entry threshold
    short_z_max = 3.0       # highest short threshold
    z_exit = 0.5            # primary exit (mean reversion)
    z_partial = 1.0         # partial exit
    long_stop = -4.5        # stop loss for longs
    short_stop = 3.5        # stop loss for shorts
    max_hold = 25           # time stop (bars)
    rsi_period = 14
    rsi_long_max = 40       # RSI confirmation for long (oversold-ish)
    rsi_short_min = 60      # RSI confirmation for short (overbought-ish)

    def init(self):
        print("🌙✨ Moon Dev VolatilityContrarian initializing...")

        # 🌙 Rolling mean & std via talib
        self.mean = self.I(talib.SMA, self.data.Close, timeperiod=self.lookback)
        self.std = self.I(talib.STDDEV, self.data.Close, timeperiod=self.lookback, nbdev=1)

        # 🌙 Z-score
        def _zscore(close_arr, mean_arr, std_arr):
            z = np.full(len(close_arr), np.nan)
            for i in range(len(close_arr)):
                if not np.isnan(mean_arr[i]) and not np.isnan(std_arr[i]) and std_arr[i] > 0:
                    z[i] = (close_arr[i] - mean_arr[i]) / std_arr[i]
            return z

        self.zscore = self.I(_zscore, self.data.Close, self.mean, self.std, name='ZScore')

        # 🌙 RSI confirmation filter
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=self.rsi_period)

        # 🌙 Track entry bar for time stop
        self.entry_bar = None
        self.entry_z = None
        self.partial_done = False

        print("🚀 Indicators ready: Mean, Std, ZScore, RSI")

    def next(self):
        price = self.data.Close[-1]
        z = self.zscore[-1]
        rsi = self.rsi[-1]

        if np.isnan(z) or np.isnan(rsi):
            return

        # ============ EXIT LOGIC ============
        if self.position:
            bars_held = len(self.data) - 1 - self.entry_bar if self.entry_bar else 0

            # 🌙 Time stop
            if bars_held >= self.max_hold:
                print(f"⏰ Moon Dev Time Stop hit ({bars_held} bars). Closing position at {price:.2f}")
                self.position.close()
                self.entry_bar = None
                self.partial_done = False
                return

            if self.position.is_long:
                # 🛑 Stop loss: regime break
                if z < self.long_stop:
                    print(f"🛑 LONG STOP hit (z={z:.2f} < {self.long_stop}). Closing at {price:.2f}")
                    self.position.close()
                    self.entry_bar = None
                    self.partial_done = False
                    return

                # 💰 Partial exit at z >= -1.0
                if not self.partial_done and z >= -self.z_partial:
                    half = int(round(self.position.size / 2))
                    if half > 0:
                        print(f"💰 Moon Dev PARTIAL EXIT long (z={z:.2f} >= -{self.z_partial}). Selling {half}")
                        self.sell(size=half)
                        self.partial_done = True

                # 🎯 Full exit at z >= -0.5 (mean reversion)
                if z >= -self.z_exit:
                    print(f"🎯 Moon Dev FULL EXIT long (z={z:.2f} >= -{self.z_exit}). Closing at {price:.2f}")
                    self.position.close()
                    self.entry_bar = None
                    self.partial_done = False
                    return

            elif self.position.is_short:
                # 🛑 Stop loss
                if z > self.short_stop:
                    print(f"🛑 SHORT STOP hit (z={z:.2f} > {self.short_stop}). Closing at {price:.2f}")
                    self.position.close()
                    self.entry_bar = None
                    self.partial_done = False
                    return

                # 💰 Partial exit at z <= +1.0
                if not self.partial_done and z <= self.z_partial:
                    half = int(round(self.position.size / 2))
                    if half > 0:
                        print(f"💰 Moon Dev PARTIAL EXIT short (z={z:.2f} <= {self.z_partial}). Buying {half}")
                        self.buy(size=half)
                        self.partial_done = True

                # 🎯 Full exit at z <= +0.5
                if z <= self.z_exit:
                    print(f"🎯 Moon Dev FULL EXIT short (z={z:.2f} <= {self.z_exit}). Closing at {price:.2f}")
                    self.position.close()
                    self.entry_bar = None
                    self.partial_done = False
                    return

        # ============ ENTRY LOGIC ============
        if not self.position:
            # 🌙 LONG: z crosses below -3.0 and stays >= -4.0, RSI confirms oversold
            if self.long_z_max <= z <= self.long_z_entry and rsi < self.rsi_long_max:
                # Inverse sizing: smaller at -4σ, larger at -3σ
                # Scale factor between 0.5 and 1.0
                extremity = abs(z + self.long_z_entry) / (self.long_z_max - self.long_z_entry)  # 0 at -3, 1 at -4
                scale = 1.0 - 0.5 * extremity  # 1.0 at -3, 0.5 at -4
                size = int(round(1_000_000 * scale))
                if size < 1:
                    size = 1
                print(f"🌙🚀 Moon Dev LONG signal! z={z:.2f}, RSI={rsi:.1f}, scale={scale:.2f}, size={size}")
                self.buy(size=size)
                self.entry_bar = len(self.data) - 1
                self.entry_z = z
                self.partial_done = False
                return

            # 🌙 SHORT: z crosses above +2.0 and stays <= +3.0, RSI confirms overbought
            if self.short_z_entry <= z <= self.short_z_max and rsi > self.rsi_short_min:
                extremity = abs(z - self.short_z_entry) / (self.short_z_max - self.short_z_entry)  # 0 at +2, 1 at +3
                scale = 1.0 - 0.5 * extremity  # 1.0 at +2, 0.5 at +3
                size = int(round(1_000_000 * scale))
                if size < 1:
                    size = 1
                print(f"🌙🚀 Moon Dev SHORT signal! z={z:.2f}, RSI={rsi:.1f}, scale={scale:.2f}, size={size}")
                self.sell(size=size)
                self.entry_bar = len(self.data) - 1
                self.entry_z = z
                self.partial_done = False
                return


# ============ DATA LOADING ============
print("🌙 Loading Moon Dev data...")
data = pd.read_csv(
    "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
)

# 🌙 Clean columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# 🌙 Proper case mapping
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume',
    'datetime': 'Datetime'
})

# 🌙 Set datetime index
if 'Datetime' in data.columns:
    data['Datetime'] = pd.to_datetime(data['Datetime'])
    data = data.set_index('Datetime')

print(f"🚀 Data loaded: {len(data)} rows")
print(data.head())

# ============ RUN BACKTEST ============
print("🌙✨ Starting Moon Dev Backtest...")
bt = Backtest(data, VolatilityContrarian, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)