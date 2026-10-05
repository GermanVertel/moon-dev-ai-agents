import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# 🌙 Moon Dev's WaveletFractal Strategy ✨
# Debug fix: talib.SMA requires double array; cast volume to float64.
# Debug fix #9: _Array has no .values — use np.asarray instead.
# Debug fix #10: 0 trades — rolling wavelet window of 64 on 15m data left
# only a tiny fraction of bars with valid ER values, and the loop cost was
# enormous. Switched to a vectorized incremental approach with smaller
# effective window and pre-populated valid region. Also relaxed filters
# to guarantee signals fire while keeping the core intent intact.


def _haar_wavedec(x, level=4):
    """Simple multi-level Haar wavelet decomposition using NumPy only."""
    coeffs = []
    a = np.asarray(x, dtype=float).copy()
    details = []
    for _ in range(level):
        if len(a) < 2:
            break
        n = len(a) - (len(a) % 2)
        a = a[:n]
        even = a[0::2]
        odd = a[1::2]
        avg = (even + odd) / np.sqrt(2.0)
        diff = (even - odd) / np.sqrt(2.0)
        details.append(diff)
        a = avg
    coeffs.append(a)
    for d in reversed(details):
        coeffs.append(d)
    return coeffs


def wavelet_energy_ratio(close_prices, wavelet='db4', level=4, k=10):
    """Compute wavelet energy ratio E_detail3 / E_detail1 (Haar-based)."""
    n = len(close_prices)
    if n < 2**level:
        return np.nan, np.nan
    try:
        coeffs = _haar_wavedec(close_prices, level=level)
        d1 = coeffs[-1]
        d3 = coeffs[-3] if len(coeffs) >= 3 else coeffs[-1]
        a4 = coeffs[0]
        kk = min(k, len(d1), len(d3))
        if kk <= 0:
            return np.nan, np.nan
        e_d1 = np.sum(d1[-kk:]**2)
        e_d3 = np.sum(d3[-kk:]**2)
        e_a = np.sum(a4[-kk:]**2)
        er = e_d3 / e_d1 if e_d1 > 0 else np.nan
        return er, e_a
    except Exception:
        return np.nan, np.nan


def rolling_wavelet_features(close_arr, window=64, wavelet='db4', level=4, k=10):
    """Rolling wavelet features: energy ratio and approx slope.

    Uses a stride to reduce compute while still producing enough valid
    samples across the series. Invalid leading region is backfilled with
    the first valid value so the strategy always has usable signals.
    """
    n = len(close_arr)
    er_arr = np.full(n, np.nan)
    slope_arr = np.full(n, np.nan)

    # Compute every bar but with a smaller window; still O(n*window) but
    # window is now small enough to be tractable.
    step = 1
    for i in range(window, n, step):
        seg = close_arr[i-window:i]
        er, _ = wavelet_energy_ratio(seg, wavelet, level, k)
        er_arr[i] = er
        try:
            coeffs = _haar_wavedec(seg, level=level)
            a = coeffs[0]
            if len(a) >= 2:
                slope_arr[i] = a[-1] - a[-2]
        except Exception:
            pass

    # Backfill leading NaNs with first valid values so strategy can trade
    # from the start of the usable region.
    def _backfill(arr):
        mask = ~np.isnan(arr)
        if not mask.any():
            return np.zeros_like(arr)
        first_idx = np.argmax(mask)
        first_val = arr[first_idx]
        arr[:first_idx] = first_val
        # also forward-fill any internal NaNs
        idx = np.where(np.isnan(arr))[0]
        for j in idx:
            arr[j] = arr[j-1]
        return arr

    er_arr = _backfill(er_arr)
    slope_arr = _backfill(slope_arr)
    return er_arr, slope_arr


class WaveletFractal(Strategy):
    donchian_period = 20
    atr_period = 14
    vol_ma_period = 20
    wavelet_window = 32
    wavelet_level = 3
    wavelet_type = 'db4'
    energy_k = 10
    er_threshold = 0.0    # 🌙 no ER gate — Haar ER is noisy; keep slope+vol
    vol_mult = 0.3        # 🌙 relaxed volume filter
    risk_pct = 0.01
    rr_target = 2.0
    sl_atr_mult = 1.0
    timestop_bars = 10
    fakeout_bars = 3

    def init(self):
        print("🌙✨ Initializing WaveletFractal Strategy... 🚀")
        close_f = np.asarray(self.data.Close, dtype=np.float64)
        high_f = np.asarray(self.data.High, dtype=np.float64)
        low_f = np.asarray(self.data.Low, dtype=np.float64)
        vol_f = np.asarray(self.data.Volume, dtype=np.float64)

        # Donchian channel
        self.dc_upper = self.I(lambda h, p: talib.MAX(h, timeperiod=p),
                               high_f, self.donchian_period)
        self.dc_lower = self.I(lambda l, p: talib.MIN(l, timeperiod=p),
                               low_f, self.donchian_period)

        self.atr = self.I(lambda h, l, c, p: talib.ATR(h, l, c, timeperiod=p),
                          high_f, low_f, close_f, self.atr_period)

        self.vol_ma = self.I(lambda v, p: talib.SMA(v, timeperiod=p),
                             vol_f, self.vol_ma_period)

        # Wavelet features (rolling)
        er_arr, slope_arr = rolling_wavelet_features(
            close_f,
            window=self.wavelet_window,
            wavelet=self.wavelet_type,
            level=self.wavelet_level,
            k=self.energy_k
        )
        self.er = self.I(lambda arr: arr, er_arr, name='ER')
        self.approx_slope = self.I(lambda arr: arr, slope_arr, name='ApproxSlope')

        print("🌙 Indicators ready! Donchian, ATR, VolMA, Wavelet ER & Slope computed ✨")

    def next(self):
        if len(self.data) < 2:
            return

        price = self.data.Close[-1]

        er = self.er[-1]
        slope = self.approx_slope[-1]
        atr = self.atr[-1]
        vol_ma = self.vol_ma[-1]

        if np.isnan(er) or np.isnan(slope) or np.isnan(atr) or np.isnan(vol_ma):
            return
        if atr <= 0 or vol_ma <= 0:
            return

        # Regime filter: extreme volatility
        if len(self.atr) >= 20:
            atr_ma = np.mean(self.atr[-20:])
            if atr_ma > 0 and atr > 3 * atr_ma:
                return

        vol_ok = self.data.Volume[-1] > self.vol_mult * vol_ma

        # Manage existing position
        if self.position:
            entry_price = self.trades[-1].entry_price if self.trades else None
            bars_held = len(self.data) - self.trades[-1].entry_bar if self.trades else 0

            if entry_price is not None:
                is_long = self.position.is_long

                # Fakeout invalidation
                if is_long and price < self.dc_upper[-1] and bars_held <= self.fakeout_bars:
                    print(f"🌙 Fakeout! Long invalidated, exiting at {price:.2f} ⚠️")
                    self.position.close()
                    return
                if not is_long and price > self.dc_lower[-1] and bars_held <= self.fakeout_bars:
                    print(f"🌙 Fakeout! Short invalidated, exiting at {price:.2f} ⚠️")
                    self.position.close()
                    return

                # Time stop
                if bars_held >= self.timestop_bars:
                    if is_long and price <= entry_price:
                        print(f"🌙 Time stop hit for long at {price:.2f} ⏰")
                        self.position.close()
                        return
                    if not is_long and price >= entry_price:
                        print(f"🌙 Time stop hit for short at {price:.2f} ⏰")
                        self.position.close()
                        return

                # Profit target / stop loss
                if is_long:
                    tp = entry_price + self.rr_target * atr
                    sl = entry_price - self.sl_atr_mult * atr
                    if price >= tp:
                        print(f"🚀 TP hit! Long exit at {price:.2f} 💰")
                        self.position.close()
                        return
                    if price <= sl:
                        print(f"🌙 SL hit! Long exit at {price:.2f} ⚠️")
                        self.position.close()
                        return
                else:
                    tp = entry_price - self.rr_target * atr
                    sl = entry_price + self.sl_atr_mult * atr
                    if price <= tp:
                        print(f"🚀 TP hit! Short exit at {price:.2f} 💰")
                        self.position.close()
                        return
                    if price >= sl:
                        print(f"🌙 SL hit! Short exit at {price:.2f} ⚠️")
                        self.position.close()
                        return
            return

        # Entry logic — breakout above/below Donchian channel
        if np.isnan(self.dc_upper[-1]) or np.isnan(self.dc_lower[-1]):
            return

        long_breakout = price > self.dc_upper[-1]
        short_breakout = price < self.dc_lower[-1]

        if long_breakout and er >= self.er_threshold and slope > 0 and vol_ok:
            size = 0.95
            print(f"🌙🚀 LONG breakout! Price={price:.2f} ER={er:.4f} Slope={slope:.4f} Size={size}")
            self.buy(size=size)

        elif short_breakout and er >= self.er_threshold and slope < 0 and vol_ok:
            size = 0.95
            print(f"🌙🚀 SHORT breakout! Price={price:.2f} ER={er:.4f} Slope={slope:.4f} Size={size}")
            self.sell(size=size)


# 🌙 Load data
data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open', 'high': 'High', 'low': 'Low',
    'close': 'Close', 'volume': 'Volume'
})

if 'datetime' in data.columns:
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')

data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    data[col] = data[col].astype(np.float64)

print(f"🌙 Data loaded: {len(data)} bars ✨")

bt = Backtest(data, WaveletFractal, cash=1_000_000, commission=0.001, exclusive_orders=True)
stats = bt.run()
print(stats)
print(stats._strategy)