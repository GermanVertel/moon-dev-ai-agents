import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Load data
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')

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

# Set datetime index
if 'datetime' in data.columns:
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')

# Ensure numeric dtypes for talib
for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    data[col] = pd.to_numeric(data[col], errors='coerce').astype(float)

data = data.dropna()

print("🌙 Moon Dev data loaded! Shape:", data.shape)
print("✨ Columns:", list(data.columns))


class VortexVolumePullback(Strategy):
    # Strategy parameters
    vi_period = 14
    ema_fast = 10
    ema_slow = 20
    atr_period = 14
    vol_ma_period = 20
    vi_threshold = 0.1
    vol_surge_mult = 1.5
    risk_pct = 0.02

    def init(self):
        print("🚀 Moon Dev initializing VortexVolume Pullback strategy...")
        high = self.data.High
        low = self.data.Low
        close = self.data.Close
        volume = self.data.Volume

        # Vortex Indicator
        def vortex_plus(h, l, c, n):
            h = np.asarray(h, dtype=float)
            l = np.asarray(l, dtype=float)
            c = np.asarray(c, dtype=float)
            vm_plus = np.abs(h - np.roll(l, 1))
            vm_plus[0] = np.nan
            tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
            tr[0] = np.nan
            vm_plus_s = pd.Series(vm_plus).rolling(n).sum().values
            tr_s = pd.Series(tr).rolling(n).sum().values
            with np.errstate(divide='ignore', invalid='ignore'):
                return vm_plus_s / tr_s

        def vortex_minus(h, l, c, n):
            h = np.asarray(h, dtype=float)
            l = np.asarray(l, dtype=float)
            c = np.asarray(c, dtype=float)
            vm_minus = np.abs(l - np.roll(h, 1))
            vm_minus[0] = np.nan
            tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
            tr[0] = np.nan
            vm_minus_s = pd.Series(vm_minus).rolling(n).sum().values
            tr_s = pd.Series(tr).rolling(n).sum().values
            with np.errstate(divide='ignore', invalid='ignore'):
                return vm_minus_s / tr_s

        self.vi_plus = self.I(vortex_plus, high, low, close, self.vi_period)
        self.vi_minus = self.I(vortex_minus, high, low, close, self.vi_period)

        # OBV - cast to double arrays to satisfy talib
        close_arr = np.asarray(close, dtype=np.float64)
        volume_arr = np.asarray(volume, dtype=np.float64)
        self.obv = self.I(talib.OBV, close_arr, volume_arr)

        # EMAs
        self.ema10 = self.I(talib.EMA, close_arr, timeperiod=self.ema_fast)
        self.ema20 = self.I(talib.EMA, close_arr, timeperiod=self.ema_slow)

        # ATR
        high_arr = np.asarray(high, dtype=np.float64)
        low_arr = np.asarray(low, dtype=np.float64)
        self.atr = self.I(talib.ATR, high_arr, low_arr, close_arr, timeperiod=self.atr_period)

        # Volume MA
        self.vol_ma = self.I(talib.SMA, volume_arr, timeperiod=self.vol_ma_period)

        print("🌙✨ Moon Dev indicators initialized!")

    def next(self):
        if len(self.data) < 50:
            return

        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        volume = self.data.Volume[-1]

        vi_p = self.vi_plus[-1]
        vi_m = self.vi_minus[-1]
        vi_p_1 = self.vi_plus[-2]
        vi_m_1 = self.vi_minus[-2]
        vi_p_2 = self.vi_plus[-3]
        vi_m_2 = self.vi_minus[-3]

        obv = self.obv[-1]
        obv_2 = self.obv[-3]

        ema10 = self.ema10[-1]
        ema20 = self.ema20[-1]
        atr = self.atr[-1]
        vol_ma = self.vol_ma[-1]

        # Skip if NaN
        if any(np.isnan(x) for x in [vi_p, vi_m, vi_p_1, vi_m_1, vi_p_2, vi_m_2, obv, obv_2, ema10, ema20, atr, vol_ma]):
            return

        # Trend strength filter
        spread = abs(vi_p - vi_m)

        # Bullish trend: VI+ > VI- for 3 consecutive bars
        bull_trend = (vi_p > vi_m) and (vi_p_1 > vi_m_1) and (vi_p_2 > vi_m_2)
        bear_trend = (vi_m > vi_p) and (vi_m_1 > vi_p_1) and (vi_m_2 > vi_p_2)

        # Volume surge
        vol_surge = volume > (vol_ma * self.vol_surge_mult)

        # Exit logic for long positions
        if self.position.is_long:
            # Trend exhaustion: VI+ crosses below VI- with volume surge
            if (vi_p < vi_m) and (vi_p_1 >= vi_m_1) and vol_surge:
                print(f"🌙 EXIT LONG - Vortex exhaustion + volume surge at {price:.2f}")
                self.position.close()
                return
            # Trailing stop: close below 20 EMA
            if price < ema20:
                print(f"🌙 EXIT LONG - Price below 20 EMA at {price:.2f}")
                self.position.close()
                return

        # Exit logic for short positions
        if self.position.is_short:
            if (vi_m < vi_p) and (vi_m_1 >= vi_p_1) and vol_surge:
                print(f"🌙 EXIT SHORT - Vortex exhaustion + volume surge at {price:.2f}")
                self.position.close()
                return
            if price > ema20:
                print(f"🌙 EXIT SHORT - Price above 20 EMA at {price:.2f}")
                self.position.close()
                return

        # Entry logic - Long
        if not self.position and bull_trend and spread > self.vi_threshold:
            # Pullback: 2-4 consecutive lower closes
            pullback = (self.data.Close[-1] < self.data.Close[-2]) and (self.data.Close[-2] < self.data.Close[-3])
            # OBV not making lower low (bullish divergence)
            obv_not_lower = obv >= obv_2
            # Price resumes up: closes above prior bar's high
            resume_up = price > self.data.High[-2]

            if pullback and obv_not_lower and resume_up:
                stop = price - (2 * atr)
                risk = price - stop
                if risk > 0:
                    # Position sizing as fraction of equity (0 < size < 1)
                    size_frac = 0.95
                    print(f"🚀 MOON DEV LONG ENTRY at {price:.2f} | VI+={vi_p:.3f} VI-={vi_m:.3f} | Stop={stop:.2f}")
                    self.buy(size=size_frac, sl=stop)

        # Entry logic - Short
        if not self.position and bear_trend and spread > self.vi_threshold:
            # Rally: 2-4 consecutive higher closes
            rally = (self.data.Close[-1] > self.data.Close[-2]) and (self.data.Close[-2] > self.data.Close[-3])
            # OBV not making higher high (bearish divergence)
            obv_not_higher = obv <= obv_2
            # Price resumes down: closes below prior bar's low
            resume_down = price < self.data.Low[-2]

            if rally and obv_not_higher and resume_down:
                stop = price + (2 * atr)
                risk = stop - price
                if risk > 0:
                    # Position sizing as fraction of equity (0 < size < 1)
                    size_frac = 0.95
                    print(f"🚀 MOON DEV SHORT ENTRY at {price:.2f} | VI+={vi_p:.3f} VI-={vi_m:.3f} | Stop={stop:.2f}")
                    self.sell(size=size_frac, sl=stop)


print("🌙 Moon Dev launching backtest...")
bt = Backtest(data, VortexVolumePullback, cash=1000000, commission=0.002)
stats = bt.run()
print(stats)
print(stats._strategy)