import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# ===============================
# 🌙 MOON DEV VORTEX CLUSTERING 🌙
# ===============================

def vortex(high, low, close, n):
    """Compute Vortex Indicator VI+ and VI-"""
    tr = talib.TRANGE(high, low, close)
    vm_plus = np.abs(high - np.roll(low, 1))
    vm_minus = np.abs(low - np.roll(high, 1))
    vm_plus[0] = np.nan
    vm_minus[0] = np.nan
    tr_sum = talib.SUM(tr, timeperiod=n)
    vm_plus_sum = talib.SUM(vm_plus, timeperiod=n)
    vm_minus_sum = talib.SUM(vm_minus, timeperiod=n)
    vi_plus = vm_plus_sum / tr_sum
    vi_minus = vm_minus_sum / tr_sum
    return vi_plus, vi_minus


def cmo(close, n):
    """Chande Momentum Oscillator"""
    diff = np.diff(close, prepend=close[0])
    gains = np.where(diff > 0, diff, 0.0)
    losses = np.where(diff < 0, -diff, 0.0)
    sum_gains = talib.SUM(gains, timeperiod=n)
    sum_losses = talib.SUM(losses, timeperiod=n)
    denom = sum_gains + sum_losses
    denom = np.where(denom == 0, np.nan, denom)
    return 100.0 * (sum_gains - sum_losses) / denom


class VortexClustering(Strategy):
    # Strategy parameters
    vi_period = 14
    cmo_period = 14
    cmo_threshold = 50
    bb_period = 20
    bb_k = 2.0
    atr_short = 10
    atr_long = 50
    risk_pct = 0.015
    stop_atr_mult = 1.5
    trail_atr_mult = 1.0
    time_stop_bars = 20
    pivot_window = 5

    def init(self):
        high = self.data.High
        low = self.data.Low
        close = self.data.Close

        # 🌙 Vortex Indicator
        self.vi_plus, self.vi_minus = self.I(vortex, high, low, close, self.vi_period,
                                             name="VI+/-")

        # 🌙 Chande Momentum Oscillator
        self.cmo = self.I(cmo, close, self.cmo_period, name="CMO")

        # 🌙 Adaptive Bollinger Bands
        self.bb_mid = self.I(talib.SMA, close, timeperiod=self.bb_period, name="BB_Mid")
        self.bb_std = self.I(talib.STDDEV, close, timeperiod=self.bb_period, nbdev=1, name="BB_Std")

        # 🌙 ATR short/long for volatility clustering
        self.atr_s = self.I(talib.ATR, high, low, close, timeperiod=self.atr_short, name="ATR_S")
        self.atr_l = self.I(talib.ATR, high, low, close, timeperiod=self.atr_long, name="ATR_L")

        # 🌙 Trend filter
        self.ema200 = self.I(talib.EMA, close, timeperiod=200, name="EMA200")

        # State tracking
        self.entry_bar = None
        self.entry_price = None
        self.stop_price = None
        self.trail_extreme = None

    def _vol_ratio(self):
        s = self.atr_s[-1]
        l = self.atr_l[-1]
        if l is None or np.isnan(l) or l == 0 or np.isnan(s):
            return 1.0
        return s / l

    def _adaptive_bands(self):
        ratio = self._vol_ratio()
        mult = np.clip(ratio, 0.7, 1.8)
        width = self.bb_k * self.bb_std[-1] * mult
        mid = self.bb_mid[-1]
        return mid - width, mid + width, ratio

    def _has_bullish_divergence(self, lookback=3):
        """Simple pivot-based bullish divergence check."""
        w = self.pivot_window
        close = self.data.Close
        cmo_arr = self.cmo
        n = len(close)
        if n < 2 * w + 2:
            return False
        # Find most recent confirmed pivot low within lookback bars
        for offset in range(0, lookback):
            idx = n - 1 - w - offset
            if idx - w < 0:
                continue
            # Check pivot low in price
            window_high = max(close[idx - w: idx + w + 1])
            if close[idx] == window_high:
                continue
            if close[idx] == min(close[idx - w: idx + w + 1]):
                # Look for prior pivot low
                for prev in range(idx - w - 1, w, -1):
                    if close[prev] == min(close[prev - w: prev + w + 1]):
                        if close[idx] < close[prev] and cmo_arr[idx] > cmo_arr[prev]:
                            return True
                        break
        return False

    def _has_bearish_divergence(self, lookback=3):
        w = self.pivot_window
        close = self.data.Close
        cmo_arr = self.cmo
        n = len(close)
        if n < 2 * w + 2:
            return False
        for offset in range(0, lookback):
            idx = n - 1 - w - offset
            if idx - w < 0:
                continue
            if close[idx] == max(close[idx - w: idx + w + 1]):
                for prev in range(idx - w - 1, w, -1):
                    if close[prev] == max(close[prev - w: prev + w + 1]):
                        if close[idx] > close[prev] and cmo_arr[idx] < cmo_arr[prev]:
                            return True
                        break
        return False

    def next(self):
        price = self.data.Close[-1]
        n = len(self.data)

        # Need enough history
        if n < max(self.atr_long, 200) + 5:
            return

        vi_p = self.vi_plus[-1]
        vi_m = self.vi_minus[-1]
        vi_p_prev = self.vi_plus[-2]
        vi_m_prev = self.vi_minus[-2]

        if any(np.isnan(x) for x in [vi_p, vi_m, vi_p_prev, vi_m_prev]):
            return

        bull_cross = vi_p_prev <= vi_m_prev and vi_p > vi_m
        bear_cross = vi_m_prev <= vi_p_prev and vi_m > vi_p

        cmo_val = self.cmo[-1]
        if np.isnan(cmo_val):
            return

        lower_band, upper_band, vol_ratio = self._adaptive_bands()

        # =====================
        # 🌙 MANAGE OPEN TRADE
        # =====================
        if self.position:
            bars_held = n - self.entry_bar

            if self.position.is_long:
                # Trailing stop
                if self.trail_extreme is None or price > self.trail_extreme:
                    self.trail_extreme = price
                atr_val = self.atr_s[-1]
                if not np.isnan(atr_val):
                    new_stop = self.trail_extreme - self.trail_atr_mult * atr_val
                    if new_stop > self.stop_price:
                        self.stop_price = new_stop

                # Hard stop
                if price <= self.stop_price:
                    print(f"🌙💥 LONG STOP HIT @ {price:.2f} | stop={self.stop_price:.2f}")
                    self.position.close()
                    self._reset()
                    return

                # Upper band exit
                if price > upper_band:
                    print(f"🌙✨ LONG BB EXIT @ {price:.2f} > {upper_band:.2f}")
                    self.position.close()
                    self._reset()
                    return

                # Bearish vortex reversal
                if bear_cross:
                    print(f"🌙🔄 LONG VI REVERSAL EXIT @ {price:.2f}")
                    self.position.close()
                    self._reset()
                    return

                # Time stop
                if bars_held >= self.time_stop_bars:
                    print(f"🌙⏰ LONG TIME STOP @ {price:.2f} after {bars_held} bars")
                    self.position.close()
                    self._reset()
                    return

            elif self.position.is_short:
                if self.trail_extreme is None or price < self.trail_extreme:
                    self.trail_extreme = price
                atr_val = self.atr_s[-1]
                if not np.isnan(atr_val):
                    new_stop = self.trail_extreme + self.trail_atr_mult * atr_val
                    if new_stop < self.stop_price:
                        self.stop_price = new_stop

                if price >= self.stop_price:
                    print(f"🌙💥 SHORT STOP HIT @ {price:.2f} | stop={self.stop_price:.2f}")
                    self.position.close()
                    self._reset()
                    return

                if price < lower_band:
                    print(f"🌙✨ SHORT BB EXIT @ {price:.2f} < {lower_band:.2f}")
                    self.position.close()
                    self._reset()
                    return

                if bull_cross:
                    print(f"🌙🔄 SHORT VI REVERSAL EXIT @ {price:.2f}")
                    self.position.close()
                    self._reset()
                    return

                if bars_held >= self.time_stop_bars:
                    print(f"🌙⏰ SHORT TIME STOP @ {price:.2f} after {bars_held} bars")
                    self.position.close()
                    self._reset()
                    return

            return

        # =====================
        # 🌙 ENTRY LOGIC
        # =====================
        atr_val = self.atr_s[-1]
        if np.isnan(atr_val) or atr_val <= 0:
            return

        # Long entry
        if bull_cross:
            cmo_ok = cmo_val < -self.cmo_threshold or self._has_bullish_divergence()
            not_chasing = price < upper_band
            trend_ok = price > self.ema200[-1]

            if cmo_ok and not_chasing and trend_ok:
                stop = price - self.stop_atr_mult * atr_val
                risk_per_unit = price - stop
                if risk_per_unit <= 0:
                    return
                equity = self.equity
                risk_amount = equity * self.risk_pct
                if vol_ratio > 1.5:
                    risk_amount *= 0.5
                    print(f"🌙⚠️ High vol regime (ratio={vol_ratio:.2f}) — size halved")
                size = int(round(risk_amount / risk_per_unit))
                if size < 1:
                    size = 1
                max_size = int(equity / price) if price > 0 else 1
                size = min(size, max_size)
                if size < 1:
                    return
                print(f"🌙🚀 LONG ENTRY @ {price:.2f} | VI+={vi_p:.3f} VI-={vi_m:.3f} "
                      f"CMO={cmo_val:.1f} size={size} stop={stop:.2f}")
                self.buy(size=size)
                self.entry_bar = n
                self.entry_price = price
                self.stop_price = stop
                self.trail_extreme = price

        # Short entry
        elif bear_cross:
            cmo_ok = cmo_val > self.cmo_threshold or self._has_bearish_divergence()
            not_chasing = price > lower_band
            trend_ok = price < self.ema200[-1]

            if cmo_ok and not_chasing and trend_ok:
                stop = price + self.stop_atr_mult * atr_val
                risk_per_unit = stop - price
                if risk_per_unit <= 0:
                    return
                equity = self.equity
                risk_amount = equity * self.risk_pct
                if vol_ratio > 1.5:
                    risk_amount *= 0.5
                    print(f"🌙⚠️ High vol regime (ratio={vol_ratio:.2f}) — size halved")
                size = int(round(risk_amount / risk_per_unit))
                if size < 1:
                    size = 1
                max_size = int(equity / price) if price > 0 else 1
                size = min(size, max_size)
                if size < 1:
                    return
                print(f"🌙🔻 SHORT ENTRY @ {price:.2f} | VI+={vi_p:.3f} VI-={vi_m:.3f} "
                      f"CMO={cmo_val:.1f} size={size} stop={stop:.2f}")
                self.sell(size=size)
                self.entry_bar = n
                self.entry_price = price
                self.stop_price = stop
                self.trail_extreme = price

    def _reset(self):
        self.entry_bar = None
        self.entry_price = None
        self.stop_price = None
        self.trail_extreme = None


# ===============================
# 🌙 DATA LOADING
# ===============================
data = pd.read_csv(
    "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
)

data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[c for c in data.columns if 'unnamed' in c.lower()])
data = data.rename(columns={
    'open': 'Open', 'high': 'High', 'low': 'Low',
    'close': 'Close', 'volume': 'Volume'
})

if 'datetime' in data.columns:
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')

data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

print(f"🌙 Loaded {len(data)} bars of data ✨")
print(f"🌙 Date range: {data.index[0]} → {data.index[-1]} 🚀")

# ===============================
# 🌙 RUN BACKTEST
# ===============================
bt = Backtest(
    data,
    VortexClustering,
    cash=1_000_000,
    commission=0.001,
    exclusive=True,
    trade_on_close=False,
)

stats = bt.run()
print(stats)
print(stats._strategy)