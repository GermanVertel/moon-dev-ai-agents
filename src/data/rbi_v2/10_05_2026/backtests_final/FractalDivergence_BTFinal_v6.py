import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# ============================================================
# 🌙 MOON DEV - FRACTAL DIVERGENCE STRATEGY 🌙
# ============================================================

def higuchi_fd(series, k_max=10):
    """Higuchi Fractal Dimension of a price series."""
    series = np.asarray(series, dtype=float)
    N = len(series)
    if N < k_max * 2:
        return np.nan
    Lk = []
    kk = []
    for k in range(1, k_max + 1):
        Lm = []
        for m in range(k):
            idxs = np.arange(m, N, k)
            if len(idxs) < 2:
                continue
            Lmk = np.sum(np.abs(np.diff(series[idxs])))
            Lmk = Lmk * (N - 1) / (k * len(idxs) * k)
            Lm.append(Lmk)
        if len(Lm) == 0:
            continue
        Lk.append(np.mean(Lm))
        kk.append(1.0 / k)
    Lk = np.array(Lk)
    kk = np.array(kk)
    if len(Lk) < 2 or np.any(Lk <= 0):
        return np.nan
    coeffs = np.polyfit(np.log(kk), np.log(Lk), 1)
    return coeffs[0]


def rolling_hfd(arr, window=60, k_max=10):
    out = np.full(len(arr), np.nan)
    for i in range(window, len(arr)):
        seg = np.log(np.clip(arr[i - window:i], 1e-9, None))
        out[i] = higuchi_fd(seg, k_max=k_max)
    return out


class FractalDivergence(Strategy):
    # Strategy parameters
    rsi_period = 14
    atr_period = 14
    ema_fast = 50
    ema_slow = 200
    fd_window = 60
    fd_lookback = 5
    fd_threshold = 1.5
    swing_lookback = 20
    div_lookback = 5
    tp_atr = 2.5
    sl_atr = 1.5
    trail_trigger_atr = 1.5
    trail_atr = 1.0
    time_exit_bars = 30
    risk_pct = 0.01

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        print("🌙 Moon Dev initializing FractalDivergence indicators... ✨")

        # RSI
        self.rsi = self.I(talib.RSI, close, timeperiod=self.rsi_period, name="RSI")

        # ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period, name="ATR")

        # EMAs
        self.ema_f = self.I(talib.EMA, close, timeperiod=self.ema_fast, name="EMA50")
        self.ema_s = self.I(talib.EMA, close, timeperiod=self.ema_slow, name="EMA200")

        # Volume SMAs
        vol_arr = np.asarray(volume, dtype=float)
        self.vol_sma3 = self.I(talib.SMA, vol_arr, timeperiod=3, name="VolSMA3")
        self.vol_sma10 = self.I(talib.SMA, vol_arr, timeperiod=10, name="VolSMA10")
        self.vol_sma20 = self.I(talib.SMA, vol_arr, timeperiod=20, name="VolSMA20")

        # Rolling Fractal Dimension
        close_arr = np.asarray(close, dtype=float)
        fd = rolling_hfd(close_arr, window=self.fd_window, k_max=10)
        self.fd = self.I(lambda: fd, name="HFD")

        # Swing lows/highs
        low_arr = np.asarray(low, dtype=float)
        high_arr = np.asarray(high, dtype=float)
        self.swing_low = self.I(talib.MIN, low_arr, timeperiod=self.swing_lookback, name="SwingLow")
        self.swing_high = self.I(talib.MAX, high_arr, timeperiod=self.swing_lookback, name="SwingHigh")

        # State
        self.entry_bar = None
        self.stop_price = None
        self.tp_price = None
        self.trail_extreme = None
        self.trade_dir = 0
        self.last_long_div_bar = -1
        self.last_short_div_bar = -1

        print("🚀 Moon Dev indicators ready! Let the fractal hunt begin 🌙")

    def _detect_bull_div(self, i):
        """Price lower low, RSI higher low within swing lookback."""
        lb = self.swing_lookback
        if i < lb + 2:
            return False
        window = self.swing_lookback
        start = max(0, i - window)
        lows = np.asarray(self.data.Low[start:i], dtype=float)
        rsi_vals = np.asarray(self.rsi[start:i], dtype=float)
        if len(lows) < 5:
            return False
        min_low_idx = np.argmin(lows)
        prev_low = lows[min_low_idx]
        cur_low = self.data.Low[i]
        if cur_low >= prev_low:
            return False
        prev_rsi = rsi_vals[min_low_idx]
        cur_rsi = self.rsi[i]
        if np.isnan(prev_rsi) or np.isnan(cur_rsi):
            return False
        if cur_rsi <= prev_rsi:
            return False
        atr_val = self.atr[i]
        if np.isnan(atr_val):
            return False
        if (prev_low - cur_low) < atr_val:
            return False
        return True

    def _detect_bear_div(self, i):
        """Price higher high, RSI lower high within swing lookback."""
        lb = self.swing_lookback
        if i < lb + 2:
            return False
        window = self.swing_lookback
        start = max(0, i - window)
        highs = np.asarray(self.data.High[start:i], dtype=float)
        rsi_vals = np.asarray(self.rsi[start:i], dtype=float)
        if len(highs) < 5:
            return False
        max_high_idx = np.argmax(highs)
        prev_high = highs[max_high_idx]
        cur_high = self.data.High[i]
        if cur_high <= prev_high:
            return False
        prev_rsi = rsi_vals[max_high_idx]
        cur_rsi = self.rsi[i]
        if np.isnan(prev_rsi) or np.isnan(cur_rsi):
            return False
        if cur_rsi >= prev_rsi:
            return False
        atr_val = self.atr[i]
        if np.isnan(atr_val):
            return False
        if (cur_high - prev_high) < atr_val:
            return False
        return True

    def _fd_rising(self, i):
        if i < self.fd_lookback:
            return False
        fd_now = self.fd[i]
        fd_prev = self.fd[i - self.fd_lookback]
        if np.isnan(fd_now) or np.isnan(fd_prev):
            return False
        return fd_now > fd_prev and fd_now > self.fd_threshold

    def _volume_decaying(self, i):
        v3 = self.vol_sma3[i]
        v10 = self.vol_sma10[i]
        v20 = self.vol_sma20[i]
        if np.isnan(v3) or np.isnan(v10) or np.isnan(v20):
            return False
        return v3 < v10 and v10 < v20

    def next(self):
        i = len(self.data) - 1
        if i < self.ema_slow + 5:
            return

        price = self.data.Close[-1]
        atr_val = self.atr[-1]
        if np.isnan(atr_val) or atr_val <= 0:
            return

        # Update divergence flags
        if self._detect_bull_div(i):
            self.last_long_div_bar = i
        if self._detect_bear_div(i):
            self.last_short_div_bar = i

        # Manage open trade
        if self.position:
            self._manage_trade(i, price, atr_val)
            return

        # Entry logic
        trend_long = (self.ema_f[-1] > self.ema_s[-1]) and (price > self.ema_s[-1])
        trend_short = (self.ema_f[-1] < self.ema_s[-1]) and (price < self.ema_s[-1])

        fd_ok = self._fd_rising(i)
        vol_ok = self._volume_decaying(i)

        long_div = (self.last_long_div_bar >= 0 and
                    (i - self.last_long_div_bar) <= self.div_lookback)
        short_div = (self.last_short_div_bar >= 0 and
                     (i - self.last_short_div_bar) <= self.div_lookback)

        if trend_long and long_div and fd_ok and vol_ok:
            self._enter_long(i, price, atr_val)
        elif trend_short and short_div and fd_ok and vol_ok:
            self._enter_short(i, price, atr_val)

    def _enter_long(self, i, price, atr_val):
        stop = price - self.sl_atr * atr_val
        tp = price + self.tp_atr * atr_val
        risk = price - stop
        if risk <= 0:
            return
        equity = self.equity
        risk_amount = equity * self.risk_pct
        size = risk_amount / risk
        size = int(round(size))
        if size <= 0:
            size = 1
        max_size = int(equity / price)
        if max_size > 0 and size > max_size:
            size = max_size
        if size <= 0:
            return
        print(f"🌙✨ LONG SIGNAL! Price={price:.2f} Stop={stop:.2f} TP={tp:.2f} Size={size} 🚀")
        self.buy(size=size)
        self.entry_bar = i
        self.stop_price = stop
        self.tp_price = tp
        self.trail_extreme = price
        self.trade_dir = 1

    def _enter_short(self, i, price, atr_val):
        stop = price + self.sl_atr * atr_val
        tp = price - self.tp_atr * atr_val
        risk = stop - price
        if risk <= 0:
            return
        equity = self.equity
        risk_amount = equity * self.risk_pct
        size = risk_amount / risk
        size = int(round(size))
        if size <= 0:
            size = 1
        max_size = int(equity / price)
        if max_size > 0 and size > max_size:
            size = max_size
        if size <= 0:
            return
        print(f"🌙✨ SHORT SIGNAL! Price={price:.2f} Stop={stop:.2f} TP={tp:.2f} Size={size} 🚀")
        self.sell(size=size)
        self.entry_bar = i
        self.stop_price = stop
        self.tp_price = tp
        self.trail_extreme = price
        self.trade_dir = -1

    def _manage_trade(self, i, price, atr_val):
        if self.trade_dir == 0:
            # Sync state if position exists but trade_dir was reset
            self.trade_dir = 1 if self.position.size > 0 else -1
            if self.entry_bar is None:
                self.entry_bar = i
            if self.stop_price is None:
                entry_px = price
                if len(self.trades) > 0:
                    entry_px = self.trades[-1].entry_price
                if self.trade_dir == 1:
                    self.stop_price = entry_px - self.sl_atr * atr_val
                else:
                    self.stop_price = entry_px + self.sl_atr * atr_val
            if self.tp_price is None:
                entry_px = price
                if len(self.trades) > 0:
                    entry_px = self.trades[-1].entry_price
                if self.trade_dir == 1:
                    self.tp_price = entry_px + self.tp_atr * atr_val
                else:
                    self.tp_price = entry_px - self.tp_atr * atr_val
            if self.trail_extreme is None:
                if len(self.trades) > 0:
                    self.trail_extreme = self.trades[-1].entry_price
                else:
                    self.trail_extreme = price

        bars_held = i - self.entry_bar

        # Time exit
        if bars_held >= self.time_exit_bars:
            print(f"⏰ Moon Dev time exit after {bars_held} bars 🌙")
            self.position.close()
            self.trade_dir = 0
            return

        if self.trade_dir == 1:
            if price > self.trail_extreme:
                self.trail_extreme = price
            entry_px = self.trades[-1].entry_price if len(self.trades) > 0 else price
            moved = self.trail_extreme - entry_px
            if moved >= self.trail_trigger_atr * atr_val:
                new_stop = self.trail_extreme - self.trail_atr * atr_val
                if new_stop > self.stop_price:
                    self.stop_price = new_stop
            if price <= self.stop_price:
                print(f"🛑 Long stop hit @ {price:.2f} 🌙")
                self.position.close()
                self.trade_dir = 0
                return
            if price >= self.tp_price:
                print(f"🎯 Long TP hit @ {price:.2f} 🚀")
                self.position.close()
                self.trade_dir = 0
                return
        else:
            if price < self.trail_extreme:
                self.trail_extreme = price
            entry_px = self.trades[-1].entry_price if len(self.trades) > 0 else price
            moved = entry_px - self.trail_extreme
            if moved >= self.trail_trigger_atr * atr_val:
                new_stop = self.trail_extreme + self.trail_atr * atr_val
                if new_stop < self.stop_price:
                    self.stop_price = new_stop
            if price >= self.stop_price:
                print(f"🛑 Short stop hit @ {price:.2f} 🌙")
                self.position.close()
                self.trade_dir = 0
                return
            if price <= self.tp_price:
                print(f"🎯 Short TP hit @ {price:.2f} 🚀")
                self.position.close()
                self.trade_dir = 0
                return


# ============================================================
# 🌙 DATA LOADING & BACKTEST EXECUTION 🌙
# ============================================================
if __name__ == "__main__":
    data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
    print("🌙 Moon Dev loading data... ✨")
    data = pd.read_csv(data_path)

    # Clean columns
    data.columns = data.columns.str.strip().str.lower()
    data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

    # Map to backtesting.py required columns
    data = data.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume',
    })

    if 'datetime' in data.columns:
        data['datetime'] = pd.to_datetime(data['datetime'])
        data = data.set_index('datetime')

    data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

    # Force float64 dtype for all numeric columns
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        data[col] = data[col].astype(float)

    print(f"🚀 Data loaded: {len(data)} bars 🌙")

    bt = Backtest(
        data,
        FractalDivergence,
        cash=1_000_000,
        commission=0.001,
        exclusive_orders=True,
    )

    print("🌙✨ Running Moon Dev FractalDivergence backtest... 🚀")
    stats = bt.run()
    print(stats)
    print(stats._strategy)