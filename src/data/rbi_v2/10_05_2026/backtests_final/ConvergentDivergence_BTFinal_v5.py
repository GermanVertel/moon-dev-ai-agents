import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# ============================
# 🌙 MOON DEV CONVERGENT DIVERGENCE STRATEGY 🚀
# ============================

def load_data(path):
    data = pd.read_csv(path)
    data.columns = data.columns.str.strip().str.lower()
    data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
    # Map to backtesting.py required columns
    data = data.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume',
        'datetime': 'Date'
    })
    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'])
        data = data.set_index('Date')
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    data = data.dropna()
    # 🌙 Ensure all numeric columns are float64 (talib requires double)
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        data[col] = data[col].astype(np.float64)
    return data


class ConvergentDivergence(Strategy):
    # Strategy parameters
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    bb_period = 20
    bb_std = 2.0
    atr_period = 14
    vol_sma_period = 20
    swing_window = 5
    bbw_lookback = 100
    bbw_quartile = 0.25
    risk_pct = 0.01
    rr_min = 1.5
    time_exit_bars = 20
    atr_buffer = 0.25

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        # 🌙 Convert to float64 arrays for talib
        close_arr = np.asarray(close, dtype=np.float64)
        high_arr = np.asarray(high, dtype=np.float64)
        low_arr = np.asarray(low, dtype=np.float64)
        volume_arr = np.asarray(volume, dtype=np.float64)

        # --- MACD ---
        def _macd(c):
            return talib.MACD(c, fastperiod=self.macd_fast,
                              slowperiod=self.macd_slow,
                              signalperiod=self.macd_signal)
        self.macd, self.macd_signal_line, self.macd_hist = self.I(_macd, close)

        # --- Bollinger Bands ---
        def _bbands(c):
            return talib.BBANDS(c, timeperiod=self.bb_period,
                                nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0)
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(_bbands, close)

        # --- ATR ---
        def _atr(h, l, c):
            return talib.ATR(h, l, c, timeperiod=self.atr_period)
        self.atr = self.I(_atr, high, low, close)

        # --- Volume SMA ---
        def _sma(v):
            return talib.SMA(v, timeperiod=self.vol_sma_period)
        self.vol_sma = self.I(_sma, volume)

        # --- Bollinger Band Width ---
        def bbw(upper, middle, lower):
            upper = np.asarray(upper, dtype=np.float64)
            middle = np.asarray(middle, dtype=np.float64)
            lower = np.asarray(lower, dtype=np.float64)
            return (upper - lower) / np.where(middle == 0, np.nan, middle)
        self.bbw = self.I(bbw, self.bb_upper, self.bb_middle, self.bb_lower)

        # --- Swing Highs / Lows (fractal) ---
        def _max(h):
            return talib.MAX(h, timeperiod=self.swing_window)
        def _min(l):
            return talib.MIN(l, timeperiod=self.swing_window)
        self.swing_high = self.I(_max, high)
        self.swing_low = self.I(_min, low)

        # State tracking
        self.trade_entry_bar = None
        self.trade_direction = None
        self.entry_price = None
        self.stop_price = None
        self.target_price = None

        print("🌙✨ ConvergentDivergence indicators initialized! 🚀")

    def _check_bullish_divergence(self, i):
        """Check bullish MACD divergence on last two swing lows."""
        window = self.swing_window
        lows = np.asarray(self.data.Low, dtype=np.float64)
        macd = np.asarray(self.macd, dtype=np.float64)
        if i < window * 4:
            return False
        pivots = []
        for j in range(max(window, i - 60), i - window + 1):
            if j - window < 0 or j + window + 1 > len(lows):
                continue
            seg = lows[j - window:j + window + 1]
            if len(seg) == 2 * window + 1 and lows[j] == min(seg):
                pivots.append(j)
        if len(pivots) < 2:
            return False
        p1, p2 = pivots[-2], pivots[-1]
        if np.isnan(macd[p1]) or np.isnan(macd[p2]):
            return False
        price_div = lows[p2] < lows[p1]
        macd_div = macd[p2] > macd[p1]
        return price_div and macd_div

    def _check_bearish_divergence(self, i):
        """Check bearish MACD divergence on last two swing highs."""
        window = self.swing_window
        highs = np.asarray(self.data.High, dtype=np.float64)
        macd = np.asarray(self.macd, dtype=np.float64)
        if i < window * 4:
            return False
        pivots = []
        for j in range(max(window, i - 60), i - window + 1):
            if j - window < 0 or j + window + 1 > len(highs):
                continue
            seg = highs[j - window:j + window + 1]
            if len(seg) == 2 * window + 1 and highs[j] == max(seg):
                pivots.append(j)
        if len(pivots) < 2:
            return False
        p1, p2 = pivots[-2], pivots[-1]
        if np.isnan(macd[p1]) or np.isnan(macd[p2]):
            return False
        price_div = highs[p2] > highs[p1]
        macd_div = macd[p2] < macd[p1]
        return price_div and macd_div

    def _volume_fading(self, i, mode='low'):
        """Volume on latest swing leg < previous swing leg."""
        window = self.swing_window
        vol = np.asarray(self.data.Volume, dtype=np.float64)
        if mode == 'low':
            series = np.asarray(self.data.Low, dtype=np.float64)
            cmp_fn = lambda x: x == min(x)
        else:
            series = np.asarray(self.data.High, dtype=np.float64)
            cmp_fn = lambda x: x == max(x)

        pivots = []
        for j in range(max(window, i - 60), i - window + 1):
            if j - window < 0 or j + window + 1 > len(series):
                continue
            seg = series[j - window:j + window + 1]
            if len(seg) == 2 * window + 1 and cmp_fn(seg):
                pivots.append(j)
        if len(pivots) < 2:
            return False
        p1, p2 = pivots[-2], pivots[-1]
        return vol[p2] < vol[p1]

    def _bbw_compressed(self, i):
        """Check if BBW is in bottom quartile of lookback."""
        if i < self.bbw_lookback:
            return False
        bbw_arr = np.asarray(self.bbw, dtype=np.float64)
        window = bbw_arr[i - self.bbw_lookback:i]
        window = window[~np.isnan(window)]
        if len(window) < 20:
            return False
        threshold = np.quantile(window, self.bbw_quartile)
        if np.isnan(bbw_arr[i]):
            return False
        return bbw_arr[i] <= threshold

    def _is_bullish_reversal_candle(self, i):
        """Simple bullish reversal: close > open and close back inside lower band."""
        o = self.data.Open[i]
        c = self.data.Close[i]
        return c > o and c > self.bb_lower[i]

    def _is_bearish_reversal_candle(self, i):
        o = self.data.Open[i]
        c = self.data.Close[i]
        return c < o and c < self.bb_upper[i]

    def next(self):
        i = len(self.data) - 1
        if i < max(self.bbw_lookback, self.swing_window * 4, 50):
            return

        price = self.data.Close[-1]

        # ============ MANAGE OPEN TRADE ============
        if self.position:
            bars_held = i - self.trade_entry_bar if self.trade_entry_bar else 0

            # Time-based exit
            if bars_held >= self.time_exit_bars:
                print(f"⏰🌙 Time exit after {bars_held} bars — mean reversion failed. Closing.")
                self.position.close()
                self.trade_entry_bar = None
                return

            # Take profit at middle band
            if self.trade_direction == 'long' and price >= self.bb_middle[-1]:
                print(f"🎯🚀 LONG TP hit at middle band! Price: {price:.2f} | Target: {self.bb_middle[-1]:.2f}")
                self.position.close()
                self.trade_entry_bar = None
                return
            if self.trade_direction == 'short' and price <= self.bb_middle[-1]:
                print(f"🎯🚀 SHORT TP hit at middle band! Price: {price:.2f} | Target: {self.bb_middle[-1]:.2f}")
                self.position.close()
                self.trade_entry_bar = None
                return

            # Stop loss check (handled by bracket orders, but double-check)
            if self.trade_direction == 'long' and price <= self.stop_price:
                print(f"🛑🌙 LONG stop hit at {price:.2f}")
                self.position.close()
                self.trade_entry_bar = None
                return
            if self.trade_direction == 'short' and price >= self.stop_price:
                print(f"🛑🌙 SHORT stop hit at {price:.2f}")
                self.position.close()
                self.trade_entry_bar = None
                return
            return

        # ============ ENTRY LOGIC ============
        atr_val = self.atr[-1]
        if np.isnan(atr_val) or atr_val <= 0:
            return

        # ---- LONG SETUP ----
        near_lower = price <= self.bb_lower[-1] * 1.005
        if (near_lower
                and self._bbw_compressed(i)
                and self._check_bullish_divergence(i)
                and self._volume_fading(i, mode='low')
                and self._is_bullish_reversal_candle(i)):

            # Stop below recent swing low + ATR buffer
            recent_low = self.swing_low[-1]
            stop = recent_low - self.atr_buffer * atr_val
            risk = price - stop
            reward = self.bb_middle[-1] - price

            if risk > 0 and reward / risk >= self.rr_min:
                # Position sizing: risk_pct of equity, expressed as fraction of equity
                equity = self.equity
                risk_amount = equity * self.risk_pct
                size_units = risk_amount / risk
                # Convert to fraction of equity (0 < size < 1)
                size_frac = (size_units * price) / equity
                if size_frac <= 0:
                    size_frac = 0.01
                if size_frac >= 1:
                    size_frac = 0.99

                self.buy(size=size_frac)
                self.trade_entry_bar = i
                self.trade_direction = 'long'
                self.entry_price = price
                self.stop_price = stop
                self.target_price = self.bb_middle[-1]
                print(f"🌙🚀 LONG ENTRY! Price: {price:.2f} | Stop: {stop:.2f} | Target: {self.bb_middle[-1]:.2f} | R:R: {reward/risk:.2f} | Size: {size_frac:.4f}")
            else:
                print(f"🌙⚠️ Long setup skipped — R:R {reward/risk if risk>0 else 0:.2f} below {self.rr_min}")

        # ---- SHORT SETUP ----
        near_upper = price >= self.bb_upper[-1] * 0.995
        if (near_upper
                and self._bbw_compressed(i)
                and self._check_bearish_divergence(i)
                and self._volume_fading(i, mode='high')
                and self._is_bearish_reversal_candle(i)):

            recent_high = self.swing_high[-1]
            stop = recent_high + self.atr_buffer * atr_val
            risk = stop - price
            reward = price - self.bb_middle[-1]

            if risk > 0 and reward / risk >= self.rr_min:
                equity = self.equity
                risk_amount = equity * self.risk_pct
                size_units = risk_amount / risk
                size_frac = (size_units * price) / equity
                if size_frac <= 0:
                    size_frac = 0.01
                if size_frac >= 1:
                    size_frac = 0.99

                self.sell(size=size_frac)
                self.trade_entry_bar = i
                self.trade_direction = 'short'
                self.entry_price = price
                self.stop_price = stop
                self.target_price = self.bb_middle[-1]
                print(f"🌙🔻 SHORT ENTRY! Price: {price:.2f} | Stop: {stop:.2f} | Target: {self.bb_middle[-1]:.2f} | R:R: {reward/risk:.2f} | Size: {size_frac:.4f}")
            else:
                print(f"🌙⚠️ Short setup skipped — R:R {reward/risk if risk>0 else 0:.2f} below {self.rr_min}")


# ============================
# 🚀 RUN BACKTEST
# ============================
data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = load_data(data_path)

print(f"🌙 Loaded {len(data)} bars of BTC-USD 15m data ✨")

bt = Backtest(
    data,
    ConvergentDivergence,
    cash=1_000_000,
    commission=0.002,
    exclusive_orders=True
)

stats = bt.run()
print(stats)
print(stats._strategy)