import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# ============================================================
# 🌙 Moon Dev's DivergentBandSync Backtest 🚀
# ============================================================

data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper column mapping
data = data.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})

# Set datetime index
if 'datetime' in data.columns:
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')
elif 'date' in data.columns:
    data['date'] = pd.to_datetime(data['date'])
    data = data.set_index('date')

# Ensure only OHLCV columns remain
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

print("🌙 Moon Dev data loaded! Shape:", data.shape)
print("✨ Columns:", list(data.columns))
print(data.head())


class DivergentBandSync(Strategy):
    # Strategy parameters
    lookback = 14
    bb_period = 20
    bb_std = 2.0
    atr_period = 14
    atr_tp_mult = 2.0
    atr_sl_mult = 1.2
    time_stop = 25
    risk_pct = 0.01
    swing_order = 3  # bars left/right for swing detection

    def init(self):
        print("🌙 Initializing DivergentBandSync indicators...")
        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)

        # MACD
        self.macd, self.macd_signal, self.macd_hist = self.I(
            talib.MACD, close, fastperiod=12, slowperiod=26, signalperiod=9
        )

        # Bollinger Bands
        self.bb_upper, self.bb_mid, self.bb_lower = self.I(
            talib.BBANDS, close, timeperiod=self.bb_period,
            nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0
        )

        # ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        # Track entry bar for time stop
        self._entry_bar = 0
        self._last_pos_size = 0

        print("✨ Indicators ready! MACD, BB, ATR loaded 🌙")

    def next(self):
        # Need enough bars
        if len(self.data) < max(self.bb_period, self.atr_period, self.lookback) + 5:
            return

        # Manage existing positions with ATR-based TP/SL and time stop
        if self.position:
            self._manage_position()
            return

        # Volatility filter
        atr_now = self.atr[-1]
        if np.isnan(atr_now) or atr_now <= 0:
            return

        # Rolling ATR average for volatility filter
        atr_len = len(self.atr)
        window_size = min(50, atr_len)
        atr_window = np.array([self.atr[-i] for i in range(1, window_size + 1)])
        atr_avg = np.nanmean(atr_window)
        if np.isnan(atr_avg) or atr_avg <= 0:
            return
        if atr_now < 0.3 * atr_avg or atr_now > 3.0 * atr_avg:
            return  # choppy or spike filter

        # Detect signals
        long_signal = self._check_long()
        short_signal = self._check_short()

        if long_signal:
            self._enter_long()
        elif short_signal:
            self._enter_short()

    def _find_swings(self, series, order):
        """Find swing highs/lows using a simple pivot detector."""
        highs = []
        lows = []
        arr = np.array(series)
        n = len(arr)
        for i in range(order, n - order):
            window = arr[i - order:i + order + 1]
            if arr[i] == window.max() and np.sum(window == arr[i]) == 1:
                highs.append(i)
            if arr[i] == window.min() and np.sum(window == arr[i]) == 1:
                lows.append(i)
        return highs, lows

    def _check_long(self):
        """Bullish divergence + lower band + MACD cross up."""
        lb = self.lookback
        if len(self.data) < lb + self.swing_order * 2 + 2:
            return False

        # MACD cross up confirmation
        if not (self.macd[-2] <= self.macd_signal[-2] and self.macd[-1] > self.macd_signal[-1]):
            return False

        # Bollinger lower band proximity (close <= lower band * 1.005)
        price = self.data.Close[-1]
        lower = self.bb_lower[-1]
        if np.isnan(lower):
            return False
        if price > lower * 1.005:
            return False

        # Divergence: last two swing lows in window
        window_low = self.data.Low[-lb:]
        highs_idx, lows_idx = self._find_swings(window_low, self.swing_order)
        if len(lows_idx) < 2:
            return False

        # Most recent two swing lows (indices relative to window)
        p1_idx, p2_idx = lows_idx[-2], lows_idx[-1]
        price_low1 = window_low[p1_idx]
        price_low2 = window_low[p2_idx]

        # MACD at those swing points (offset from end)
        macd_arr = np.array(self.macd)
        offset = len(self.data) - lb
        m1 = macd_arr[offset + p1_idx]
        m2 = macd_arr[offset + p2_idx]

        if np.isnan(m1) or np.isnan(m2):
            return False

        # Price lower low, MACD higher low
        if price_low2 < price_low1 and m2 > m1:
            print(f"🌙✨ BULLISH DIVERGENCE detected! Price LL {price_low1:.2f}->{price_low2:.2f}, MACD HL {m1:.4f}->{m2:.4f} 🚀")
            return True
        return False

    def _check_short(self):
        """Bearish divergence + upper band + MACD cross down."""
        lb = self.lookback
        if len(self.data) < lb + self.swing_order * 2 + 2:
            return False

        # MACD cross down confirmation
        if not (self.macd[-2] >= self.macd_signal[-2] and self.macd[-1] < self.macd_signal[-1]):
            return False

        # Bollinger upper band proximity
        price = self.data.Close[-1]
        upper = self.bb_upper[-1]
        if np.isnan(upper):
            return False
        if price < upper * 0.995:
            return False

        # Divergence: last two swing highs in window
        window_high = self.data.High[-lb:]
        highs_idx, lows_idx = self._find_swings(window_high, self.swing_order)
        if len(highs_idx) < 2:
            return False

        p1_idx, p2_idx = highs_idx[-2], highs_idx[-1]
        price_high1 = window_high[p1_idx]
        price_high2 = window_high[p2_idx]

        macd_arr = np.array(self.macd)
        offset = len(self.data) - lb
        m1 = macd_arr[offset + p1_idx]
        m2 = macd_arr[offset + p2_idx]

        if np.isnan(m1) or np.isnan(m2):
            return False

        # Price higher high, MACD lower high
        if price_high2 > price_high1 and m2 < m1:
            print(f"🌙✨ BEARISH DIVERGENCE detected! Price HH {price_high1:.2f}->{price_high2:.2f}, MACD LH {m1:.4f}->{m2:.4f} 🚀")
            return True
        return False

    def _enter_long(self):
        atr_val = self.atr[-1]
        price = self.data.Close[-1]
        sl = price - self.atr_sl_mult * atr_val
        tp = price + self.atr_tp_mult * atr_val

        risk_per_unit = price - sl
        if risk_per_unit <= 0:
            return

        # Position sizing as fraction of equity (0-1)
        risk_fraction = (self.risk_pct * price) / risk_per_unit
        size = min(max(risk_fraction, 0.01), 0.95)

        print(f"🚀🌙 LONG ENTRY @ {price:.2f} | SL={sl:.2f} | TP={tp:.2f} | Size={size:.4f} | ATR={atr_val:.2f}")
        self.buy(size=size, sl=sl, tp=tp)
        self._entry_bar = len(self.data)
        self._last_pos_size = size

    def _enter_short(self):
        atr_val = self.atr[-1]
        price = self.data.Close[-1]
        sl = price + self.atr_sl_mult * atr_val
        tp = price - self.atr_tp_mult * atr_val

        risk_per_unit = sl - price
        if risk_per_unit <= 0:
            return

        risk_fraction = (self.risk_pct * price) / risk_per_unit
        size = min(max(risk_fraction, 0.01), 0.95)

        print(f"🚀🌙 SHORT ENTRY @ {price:.2f} | SL={sl:.2f} | TP={tp:.2f} | Size={size:.4f} | ATR={atr_val:.2f}")
        self.sell(size=size, sl=sl, tp=tp)
        self._entry_bar = len(self.data)
        self._last_pos_size = size

    def _manage_position(self):
        """Time stop management."""
        # Detect new entry
        if getattr(self, '_last_pos_size', 0) != self.position.size:
            self._entry_bar = len(self.data)
            self._last_pos_size = self.position.size

        bars_held = len(self.data) - self._entry_bar
        if bars_held >= self.time_stop:
            print(f"⏰ Time stop hit after {bars_held} bars. Closing position. 🌙")
            self.position.close()
            self._last_pos_size = 0


print("🌙✨ Starting DivergentBandSync backtest... 🚀")

bt = Backtest(
    data,
    DivergentBandSync,
    cash=1_000_000,
    commission=0.001,
    trade_on_close=True,
)

stats = bt.run()
print(stats)
print(stats._strategy)