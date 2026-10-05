import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Load and clean data
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
print("🌙✨ Moon Dev Data Loaded:", data.shape, "rows |", data.index[0], "->", data.index[-1])

class VolumetricDivergence(Strategy):
    atr_period = 14
    rsi_period = 14
    vol_roc_period = 14
    zscore_window = 50
    div_std_window = 50
    threshold_mult = 1.5
    atr_avg_period = 20
    risk_pct = 0.01
    stop_atr_mult = 1.5
    time_stop_bars = 10

    def init(self):
        # ATR for volatility regime & stops
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period)
        self.atr_avg = self.I(talib.SMA, self.atr, timeperiod=self.atr_avg_period)

        # Price oscillator: RSI
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=self.rsi_period)

        # Volume oscillator: ROC of volume
        self.vol_roc = self.I(talib.ROC, self.data.Volume, timeperiod=self.vol_roc_period)

        # Z-score normalization helpers
        def zscore(arr, window):
            s = pd.Series(arr)
            m = s.rolling(window).mean()
            sd = s.rolling(window).std()
            return ((s - m) / sd).values

        self.rsi_z = self.I(zscore, self.rsi, self.zscore_window)
        self.vol_z = self.I(zscore, self.vol_roc, self.zscore_window)

        # Divergence distance = vol_z - rsi_z (positive => volume strength exceeds price)
        self.div = self.I(lambda a, b: a - b, self.vol_z, self.rsi_z)

        # Rolling std of divergence for dynamic threshold
        self.div_std = self.I(lambda x: pd.Series(x).rolling(self.div_std_window).std().values, self.div)

        # Track entry bar for time stop
        self.entry_bar = None

        print("🌙 Indicators initialized: ATR, RSI, VolROC, Z-scores, Divergence, DivStd ✨")

    def next(self):
        price = self.data.Close[-1]
        atr = self.atr[-1]
        atr_avg = self.atr_avg[-1]
        div = self.div[-1]
        div_std = self.div_std[-1]
        rsi_z = self.rsi_z[-1]
        vol_z = self.vol_z[-1]

        if np.isnan(div_std) or np.isnan(atr_avg) or np.isnan(rsi_z) or np.isnan(vol_z):
            return

        vol_filter = atr > atr_avg
        upper = self.threshold_mult * div_std
        lower = -self.threshold_mult * div_std

        # Manage open position
        if self.position:
            entry_price = self.trades[-1].entry_price
            is_long = self.position.is_long
            bars_held = len(self.data) - 1 - self.entry_bar

            # Stop loss: 1.5x ATR from entry
            if is_long:
                stop_price = entry_price - self.stop_atr_mult * atr
                if price <= stop_price:
                    self.position.close()
                    print(f"🛑 LONG STOP hit @ {price:.2f} | ATR stop {stop_price:.2f} 🌙")
                    self.entry_bar = None
                    return
            else:
                stop_price = entry_price + self.stop_atr_mult * atr
                if price >= stop_price:
                    self.position.close()
                    print(f"🛑 SHORT STOP hit @ {price:.2f} | ATR stop {stop_price:.2f} 🌙")
                    self.entry_bar = None
                    return

            # Profit target: divergence reverts to zero
            if is_long and div <= 0:
                self.position.close()
                print(f"🎯 LONG TARGET: divergence reverted to {div:.3f} @ {price:.2f} 🚀")
                self.entry_bar = None
                return
            if not is_long and div >= 0:
                self.position.close()
                print(f"🎯 SHORT TARGET: divergence reverted to {div:.3f} @ {price:.2f} 🚀")
                self.entry_bar = None
                return

            # Time stop
            if bars_held >= self.time_stop_bars:
                self.position.close()
                print(f"⏰ TIME STOP after {bars_held} bars @ {price:.2f} 🌙")
                self.entry_bar = None
                return

            return

        # No position — look for entries
        if not vol_filter:
            return

        # Long entry
        if rsi_z < 0 and vol_z > 0 and div > upper:
            stop_dist = self.stop_atr_mult * atr
            if stop_dist <= 0:
                return
            risk_amount = self.equity * self.risk_pct
            size = int(round(risk_amount / stop_dist))
            if size > 0:
                self.buy(size=size)
                self.entry_bar = len(self.data) - 1
                print(f"🌙🚀 LONG ENTRY @ {price:.2f} | div={div:.3f} > {upper:.3f} | rsi_z={rsi_z:.2f} vol_z={vol_z:.2f} | size={size}")

        # Short entry
        elif rsi_z > 0 and vol_z < 0 and div < lower:
            stop_dist = self.stop_atr_mult * atr
            if stop_dist <= 0:
                return
            risk_amount = self.equity * self.risk_pct
            size = int(round(risk_amount / stop_dist))
            if size > 0:
                self.sell(size=size)
                self.entry_bar = len(self.data) - 1
                print(f"🌙🔻 SHORT ENTRY @ {price:.2f} | div={div:.3f} < {lower:.3f} | rsi_z={rsi_z:.2f} vol_z={vol_z:.2f} | size={size}")


bt = Backtest(data, VolumetricDivergence, cash=1_000_000, commission=0.0002)
stats = bt.run()
print(stats)
print(stats._strategy)