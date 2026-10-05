import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# 🌙 Moon Dev's Volatility Compression Reversal Strategy 🚀
# Mean-reversion SHORT setup triggered by Bollinger Band compression + volume anomaly

data_path = '/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv'
data = pd.read_csv(data_path)

# 🧹 Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# 📋 Proper column mapping
data = data.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
if 'datetime' in data.columns:
    data = data.set_index(pd.to_datetime(data['datetime']))
    data = data.drop(columns=['datetime'])
data.index = pd.to_datetime(data.index)

print("🌙✨ Moon Dev Data Loaded! Shape:", data.shape)
print("📊 Columns:", list(data.columns))
print("🚀 Head:\n", data.head())


class VolatilityCompressionReversal(Strategy):
    # Strategy parameters
    bb_period = 20
    bb_dev = 2.0
    vol_period = 20
    vol_tol_low = 1.95
    vol_tol_high = 2.05
    bbw_std_mult = 2.0
    bbw_lookback = 50
    sl_pct = 0.20   # 20% above entry (short)
    tp_pct = 0.10   # 10% below entry (short)
    max_hold_bars = 96  # 96 * 15m = 24 hours (intraday-ish)

    def init(self):
        print("🌙 Initializing Moon Dev indicators...")

        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        volume = pd.Series(self.data.Volume)

        # 📈 Bollinger Bands (20, 2) using talib only
        self.bb_mid = self.I(talib.SMA, close, timeperiod=self.bb_period)
        self.bb_upper = self.I(lambda c: talib.SMA(c, timeperiod=self.bb_period) +
                               2 * talib.STDDEV(c, timeperiod=self.bb_period, nbdev=1), close)
        self.bb_lower = self.I(lambda c: talib.SMA(c, timeperiod=self.bb_period) -
                               2 * talib.STDDEV(c, timeperiod=self.bb_period, nbdev=1), close)

        # 📉 Bollinger Band Width
        self.bbw = self.I(lambda u, l, m: (u - l) / m * 100.0,
                          self.bb_upper, self.bb_lower, self.bb_mid)

        # 📊 BBW rolling mean & std for compression filter
        self.bbw_mean = self.I(talib.SMA, self.bbw, timeperiod=self.bbw_lookback)
        self.bbw_std = self.I(talib.STDDEV, self.bbw, timeperiod=self.bbw_lookback, nbdev=1)

        # 📦 Volume average
        self.vol_avg = self.I(talib.SMA, volume, timeperiod=self.vol_period)

        # 🎯 ATR for optional dynamic stops
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=14)

        print("🌙✨ Indicators ready! Let's fade some moves! 🚀")

    def next(self):
        price = self.data.Close[-1]

        # Need enough history
        if len(self.data) < max(self.bbw_lookback, self.vol_period) + 5:
            return

        # If in position, check time-based exit
        if self.position:
            entry_bar = self.trades[-1].entry_bar if len(self.trades) > 0 else 0
            if len(self.data) - entry_bar >= self.max_hold_bars:
                print(f"⏰ Moon Dev Time Exit! Closing at {price:.2f}")
                self.position.close()
            return

        bbw_now = self.bbw[-1]
        bbw_mean = self.bbw_mean[-1]
        bbw_std = self.bbw_std[-1]
        vol_now = self.data.Volume[-1]
        vol_avg = self.vol_avg[-1]

        if bbw_mean is None or bbw_std is None or vol_avg is None:
            return
        if np.isnan(bbw_now) or np.isnan(bbw_mean) or np.isnan(bbw_std) or np.isnan(vol_avg):
            return
        if vol_avg <= 0:
            return

        # ✅ Condition A: BBW compression (below its own mean - 2*std)
        compression = bbw_now < (bbw_mean - self.bbw_std_mult * bbw_std)

        # ✅ Condition B: Volume ~ 2x average (tolerance band)
        vol_ratio = vol_now / vol_avg
        vol_signal = (vol_ratio >= self.vol_tol_low) and (vol_ratio <= self.vol_tol_high)

        if compression and vol_signal:
            print(f"🌙✨ SIGNAL! BBW={bbw_now:.4f} < {bbw_mean - self.bbw_std_mult*bbw_std:.4f} | "
                  f"VolRatio={vol_ratio:.3f} | Price={price:.2f} 🚀")

            # 🎯 SHORT entry (mean reversion fade)
            sl = price * (1 + self.sl_pct)   # 20% above
            tp = price * (1 - self.tp_pct)   # 10% below

            # 💰 Position sizing: 50% of full position as a fraction of equity
            size = 0.5

            self.sell(size=size, sl=sl, tp=tp)
            print(f"🔻 SHORT ENTRY | Size={size} | Entry={price:.2f} | SL={sl:.2f} | TP={tp:.2f}")


bt = Backtest(data, VolatilityCompressionReversal, cash=1_000_000, commission=0.001)

print("🌙🚀 Running Moon Dev Backtest...")
stats = bt.run()
print(stats)
print(stats._strategy)