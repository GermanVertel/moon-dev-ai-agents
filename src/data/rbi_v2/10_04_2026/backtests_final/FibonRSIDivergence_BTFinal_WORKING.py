import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# ─────────────────────────────────────────────
# 🌙 Moon Dev's FibonRSIDivergence Backtest 🚀
# ✨ Package Check: NO backtesting.lib usage — talib + numpy only ✨
# ─────────────────────────────────────────────

class FibonRSIDivergence(Strategy):
    # ── Tunable parameters ──
    ema_period = 200
    rsi_period = 14
    atr_period = 14
    swing_lookback = 20
    divergence_lookback = 60
    min_divergence_bars = 5
    fib_tolerance = 0.005
    risk_pct = 0.02
    rr_min = 2.0
    atr_trail_mult = 2.0
    time_exit_bars = 20
    vol_lookback = 20

    def init(self):
        print("🌙✨ Moon Dev initializing FibonRSIDivergence strategy...")
        print("🔍 Package check: NO backtesting.lib — using talib + numpy ✅")
        self.ema = self.I(talib.EMA, self.data.Close, timeperiod=self.ema_period)
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=self.rsi_period)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close,
                          timeperiod=self.atr_period)
        self.atr_avg = self.I(talib.SMA, self.atr, timeperiod=self.vol_lookback)
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=self.swing_lookback)
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=self.swing_lookback)
        self.entry_bar = None
        self.entry_price = None
        self.stop_price = None
        self.target_price = None
        self.trail_active = False
        print("🚀 Indicators loaded: EMA200, RSI14, ATR14, Swing Hi/Lo")

    # ── helpers ──
    def _find_swing_low(self, idx, lookback):
        start = max(0, idx - lookback)
        if idx <= start:
            return None
        seg = np.asarray(self.data.Low[start:idx])
        if len(seg) < 3:
            return None
        return int(np.argmin(seg)) + start

    def _find_swing_high(self, idx, lookback):
        start = max(0, idx - lookback)
        if idx <= start:
            return None
        seg = np.asarray(self.data.High[start:idx])
        if len(seg) < 3:
            return None
        return int(np.argmax(seg)) + start

    def _near_fib(self, price, fib_levels):
        for lvl in fib_levels:
            if lvl > 0 and abs(price - lvl) / lvl <= self.fib_tolerance:
                return True
        return False

    def _bullish_divergence(self, idx):
        cur_low_idx = self._find_swing_low(idx, self.swing_lookback)
        if cur_low_idx is None or cur_low_idx < self.min_divergence_bars:
            return None
        prior_low_idx = self._find_swing_low(cur_low_idx - 1, self.divergence_lookback)
        if prior_low_idx is None:
            return None
        if (cur_low_idx - prior_low_idx) < self.min_divergence_bars:
            return None
        price_now = self.data.Low[cur_low_idx]
        price_prev = self.data.Low[prior_low_idx]
        rsi_now = self.rsi[cur_low_idx]
        rsi_prev = self.rsi[prior_low_idx]
        if np.isnan(rsi_now) or np.isnan(rsi_prev):
            return None
        if price_now < price_prev and rsi_now > rsi_prev and rsi_now < 45:
            return {"low_idx": cur_low_idx, "rsi": rsi_now}
        return None

    def _bearish_divergence(self, idx):
        cur_high_idx = self._find_swing_high(idx, self.swing_lookback)
        if cur_high_idx is None or cur_high_idx < self.min_divergence_bars:
            return None
        prior_high_idx = self._find_swing_high(cur_high_idx - 1, self.divergence_lookback)
        if prior_high_idx is None:
            return None
        if (cur_high_idx - prior_high_idx) < self.min_divergence_bars:
            return None
        price_now = self.data.High[cur_high_idx]
        price_prev = self.data.High[prior_high_idx]
        rsi_now = self.rsi[cur_high_idx]
        rsi_prev = self.rsi[prior_high_idx]
        if np.isnan(rsi_now) or np.isnan(rsi_prev):
            return None
        if price_now > price_prev and rsi_now < rsi_prev and rsi_now > 55:
            return {"high_idx": cur_high_idx, "rsi": rsi_now}
        return None

    def _calc_fib_levels(self, a, b):
        diff = a - b
        return {
            "0.382": a - diff * 0.382,
            "0.5":   a - diff * 0.5,
            "0.618": a - diff * 0.618,
            "0.786": a - diff * 0.786,
        }

    def _position_size(self, entry, stop):
        risk_amount = self.equity * self.risk_pct
        risk_per_unit = abs(entry - stop)
        if risk_per_unit <= 0:
            return 0
        size = risk_amount / risk_per_unit
        # 🌙 Must be a whole positive integer for unit-based sizing
        size = int(round(size))
        return size

    def next(self):
        i = len(self.data) - 1
        if i < self.ema_period + 5:
            return

        price = self.data.Close[-1]
        ema = self.ema[-1]
        rsi = self.rsi[-1]
        atr = self.atr[-1]
        atr_avg = self.atr_avg[-1]

        if np.isnan(ema) or np.isnan(rsi) or np.isnan(atr) or np.isnan(atr_avg):
            return

        # ── Manage open position ──
        if self.position:
            bars_held = i - self.entry_bar
            if bars_held >= self.time_exit_bars:
                print(f"⏰ Moon Dev time-exit after {bars_held} bars 🌙")
                self.position.close()
                return
            if self.position.is_long:
                if not self.trail_active and self.data.High[-1] >= self.entry_price + (self.entry_price - self.stop_price):
                    self.trail_active = True
                    self.stop_price = self.entry_price
                    print("🛡️ Moon Dev trail activated → stop @ breakeven")
                if self.trail_active:
                    new_stop = self.data.Close[-1] - self.atr_trail_mult * atr
                    if new_stop > self.stop_price:
                        self.stop_price = new_stop
                if self.data.Low[-1] <= self.stop_price:
                    print(f"🛑 Moon Dev LONG stop hit @ {self.stop_price:.2f}")
                    self.position.close()
                    return
                if self.data.High[-1] >= self.target_price:
                    print(f"🎯 Moon Dev LONG target hit @ {self.target_price:.2f} 🚀")
                    self.position.close()
                    return
            else:
                if not self.trail_active and self.data.Low[-1] <= self.entry_price - (self.stop_price - self.entry_price):
                    self.trail_active = True
                    self.stop_price = self.entry_price
                    print("🛡️ Moon Dev trail activated → stop @ breakeven")
                if self.trail_active:
                    new_stop = self.data.Close[-1] + self.atr_trail_mult * atr
                    if new_stop < self.stop_price:
                        self.stop_price = new_stop
                if self.data.High[-1] >= self.stop_price:
                    print(f"🛑 Moon Dev SHORT stop hit @ {self.stop_price:.2f}")
                    self.position.close()
                    return
                if self.data.Low[-1] <= self.target_price:
                    print(f"🎯 Moon Dev SHORT target hit @ {self.target_price:.2f} 🚀")
                    self.position.close()
                    return
            return

        # ── Volatility filter ──
        if atr < atr_avg:
            return

        # ── UPTREND LONG ──
        if price > ema:
            div = self._bullish_divergence(i)
            if div is None:
                return
            low_idx = div["low_idx"]
            sh_idx = self._find_swing_high(low_idx, self.divergence_lookback)
            if sh_idx is None:
                return
            sh_price = self.data.High[sh_idx]
            sl_price = self.data.Low[low_idx]
            if sh_price <= sl_price:
                return
            fibs = self._calc_fib_levels(sh_price, sl_price)
            if not self._near_fib(price, list(fibs.values())):
                return
            trigger = (self.data.Close[-1] > self.data.High[-2]) or (rsi > 50)
            if not trigger:
                return

            stop = sl_price * (1 - 0.003)
            target = sh_price
            rr = (target - price) / (price - stop) if (price - stop) > 0 else 0
            if rr < self.rr_min:
                print(f"⚠️ Moon Dev LONG skipped — R:R {rr:.2f} < {self.rr_min}")
                return
            size = self._position_size(price, stop)
            if size <= 0:
                return
            # 🌙 Cap size to available cash-worth of units
            max_units = int(self.equity / price)
            if size > max_units:
                size = max_units
            if size <= 0:
                return
            self.buy(size=size)
            self.entry_bar = i
            self.entry_price = price
            self.stop_price = stop
            self.target_price = target
            self.trail_active = False
            print(f"🌙🚀 Moon Dev LONG @ {price:.2f} | stop {stop:.2f} | target {target:.2f} | size {size}")

        # ── DOWNTREND SHORT ──
        elif price < ema:
            div = self._bearish_divergence(i)
            if div is None:
                return
            high_idx = div["high_idx"]
            sl_idx = self._find_swing_low(high_idx, self.divergence_lookback)
            if sl_idx is None:
                return
            sh_price = self.data.High[high_idx]
            sl_price = self.data.Low[sl_idx]
            if sh_price <= sl_price:
                return
            fibs = self._calc_fib_levels(sh_price, sl_price)
            if not self._near_fib(price, list(fibs.values())):
                return
            trigger = (self.data.Close[-1] < self.data.Low[-2]) or (rsi < 50)
            if not trigger:
                return

            stop = sh_price * (1 + 0.003)
            target = sl_price
            rr = (price - target) / (stop - price) if (stop - price) > 0 else 0
            if rr < self.rr_min:
                print(f"⚠️ Moon Dev SHORT skipped — R:R {rr:.2f} < {self.rr_min}")
                return
            size = self._position_size(price, stop)
            if size <= 0:
                return
            max_units = int(self.equity / price)
            if size > max_units:
                size = max_units
            if size <= 0:
                return
            self.sell(size=size)
            self.entry_bar = i
            self.entry_price = price
            self.stop_price = stop
            self.target_price = target
            self.trail_active = False
            print(f"🌙🔻 Moon Dev SHORT @ {price:.2f} | stop {stop:.2f} | target {target:.2f} | size {size}")


# ─────────────────────────────────────────────
# 🌙 Data loading
# ─────────────────────────────────────────────
data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
print(f"🌙 Loading data from {data_path} ✨")
data = pd.read_csv(data_path)

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Ensure proper mapping
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume',
})

# Parse datetime & set index
if 'datetime' in data.columns:
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')

data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
print(f"🚀 Data ready: {len(data)} bars")

# ─────────────────────────────────────────────
# 🌙 Run backtest
# ─────────────────────────────────────────────
bt = Backtest(data, FibonRSIDivergence, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)