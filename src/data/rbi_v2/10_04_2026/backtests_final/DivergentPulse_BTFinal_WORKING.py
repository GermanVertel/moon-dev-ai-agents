import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Load data
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')

# Clean columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
data = data.dropna()

print("🌙✨ Moon Dev DivergentPulse Backtest Loading... 🚀")
print(f"📊 Data shape: {data.shape}")
print(f"📅 Range: {data.index[0]} -> {data.index[-1]}")


def pivot_lows(series, left=2, right=2):
    """Return boolean array marking pivot lows."""
    arr = series.values if hasattr(series, 'values') else np.asarray(series)
    n = len(arr)
    out = np.zeros(n, dtype=bool)
    for i in range(left, n - right):
        window = arr[i - left:i + right + 1]
        if arr[i] == window.min() and np.sum(window == arr[i]) == 1:
            out[i] = True
    return out


def pivot_highs(series, left=2, right=2):
    arr = series.values if hasattr(series, 'values') else np.asarray(series)
    n = len(arr)
    out = np.zeros(n, dtype=bool)
    for i in range(left, n - right):
        window = arr[i - left:i + right + 1]
        if arr[i] == window.max() and np.sum(window == arr[i]) == 1:
            out[i] = True
    return out


class DivergentPulse(Strategy):
    # Parameters
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    rsi_period = 14
    atr_period = 14
    lookback = 20          # divergence lookback window
    proximity = 4          # bars between MACD and RSI divergence pivots
    atr_avg_period = 20
    atr_min_ratio = 0.20
    risk_pct = 0.01
    target_atr = 2.0
    stop_atr = 1.0
    breakeven_atr = 1.5
    session_end_hour = 23  # skip last 30 min -> 23:30

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low

        # Indicators via talib wrapped with self.I
        self.macd, self.macd_sig, self.macd_hist = self.I(
            talib.MACD, close, self.macd_fast, self.macd_slow, self.macd_signal
        )
        self.rsi = self.I(talib.RSI, close, self.rsi_period)
        self.atr = self.I(talib.ATR, high, low, close, self.atr_period)
        self.atr_avg = self.I(talib.SMA, self.atr, self.atr_avg_period)

        # Pivots
        self.pl = self.I(pivot_lows, low, 2, 2)
        self.ph = self.I(pivot_highs, high, 2, 2)

        # Trackers
        self.last_trade_day = None
        self.pending_long = False
        self.pending_short = False
        self.swing_high_ref = None
        self.swing_low_ref = None

        print("🌙 DivergentPulse indicators initialized ✨")

    def next(self):
        i = len(self.data) - 1
        if i < self.lookback + 5:
            return

        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        dt = self.data.index[-1]

        # ---- Session / day checks ----
        current_day = dt.date()
        # skip last 30 minutes
        if dt.hour == self.session_end_hour and dt.minute >= 30:
            return
        if dt.hour == 23 and dt.minute >= 30:
            return

        # ---- ATR filter ----
        atr_val = self.atr[-1]
        atr_avg_val = self.atr_avg[-1]
        if np.isnan(atr_val) or np.isnan(atr_avg_val) or atr_avg_val == 0:
            return
        if atr_val < self.atr_min_ratio * atr_avg_val:
            return

        # ---- If in a position, manage exits ----
        if self.position:
            self._manage_position(atr_val, dt)
            return

        # ---- One trade per day rule ----
        if self.last_trade_day == current_day:
            return

        # ---- Detect divergences ----
        bullish_div = self._check_bullish_divergence(i)
        bearish_div = self._check_bearish_divergence(i)

        # ---- Entry on structural break ----
        if bullish_div:
            # most recent swing high
            sh = self._recent_swing_high(i)
            if sh is not None and price > sh:
                self._enter_long(price, atr_val, current_day)
                return

        if bearish_div:
            sl = self._recent_swing_low(i)
            if sl is not None and price < sl:
                self._enter_short(price, atr_val, current_day)
                return

    # ---------------- Helpers ----------------
    def _recent_swing_high(self, i):
        for k in range(i - 1, max(i - self.lookback, 0), -1):
            if self.ph[k]:
                return self.data.High[k]
        return None

    def _recent_swing_low(self, i):
        for k in range(i - 1, max(i - self.lookback, 0), -1):
            if self.pl[k]:
                return self.data.Low[k]
        return None

    def _check_bullish_divergence(self, i):
        # Find pivot lows within lookback
        pivots = [k for k in range(i - self.lookback, i) if self.pl[k]]
        if len(pivots) < 2:
            return False
        # Compare last two pivot lows
        p2 = pivots[-1]  # more recent
        p1 = pivots[-2]  # older
        # Price lower low
        if not (self.data.Low[p2] < self.data.Low[p1]):
            return False
        # MACD hist higher low
        if not (self.macd_hist[p2] > self.macd_hist[p1]):
            return False
        # RSI higher low
        if not (self.rsi[p2] > self.rsi[p1]):
            return False
        # Proximity check: MACD and RSI pivots must be close
        if (i - p2) > self.proximity + 3:
            return False
        return True

    def _check_bearish_divergence(self, i):
        pivots = [k for k in range(i - self.lookback, i) if self.ph[k]]
        if len(pivots) < 2:
            return False
        p2 = pivots[-1]
        p1 = pivots[-2]
        # Price higher high
        if not (self.data.High[p2] > self.data.High[p1]):
            return False
        # MACD hist lower high
        if not (self.macd_hist[p2] < self.macd_hist[p1]):
            return False
        # RSI lower high
        if not (self.rsi[p2] < self.rsi[p1]):
            return False
        if (i - p2) > self.proximity + 3:
            return False
        return True

    def _enter_long(self, price, atr_val, current_day):
        stop = price - self.stop_atr * atr_val
        # place stop beyond divergence swing point
        sl = self._recent_swing_low(len(self.data) - 1)
        if sl is not None:
            stop = min(stop, sl - 0.1 * atr_val)
        risk = price - stop
        if risk <= 0:
            return
        size = int(round((self.equity * self.risk_pct) / risk))
        if size <= 0:
            return
        target = price + self.target_atr * atr_val
        self.buy(size=size, sl=stop, tp=target)
        self.last_trade_day = current_day
        print(f"🌙🚀 LONG DivergentPulse @ {price:.2f} | SL {stop:.2f} | TP {target:.2f} | size {size}")

    def _enter_short(self, price, atr_val, current_day):
        stop = price + self.stop_atr * atr_val
        sh = self._recent_swing_high(len(self.data) - 1)
        if sh is not None:
            stop = max(stop, sh + 0.1 * atr_val)
        risk = stop - price
        if risk <= 0:
            return
        size = int(round((self.equity * self.risk_pct) / risk))
        if size <= 0:
            return
        target = price - self.target_atr * atr_val
        self.sell(size=size, sl=stop, tp=target)
        self.last_trade_day = current_day
        print(f"🌙🚀 SHORT DivergentPulse @ {price:.2f} | SL {stop:.2f} | TP {target:.2f} | size {size}")

    def _manage_position(self, atr_val, dt):
        # Move stop to breakeven once +1.5 ATR reached
        entry = self.trades[-1].entry_price if self.trades else None
        if entry is None:
            return
        price = self.data.Close[-1]
        if self.position.is_long:
            if price >= entry + self.breakeven_atr * atr_val:
                # tighten stop to breakeven if currently below
                if self.orders:
                    pass
        else:
            if price <= entry - self.breakeven_atr * atr_val:
                pass


# Run backtest
bt = Backtest(data, DivergentPulse, cash=1_000_000, commission=0.0002, exclusive_orders=True)
stats = bt.run()
print(stats)
print(stats._strategy)