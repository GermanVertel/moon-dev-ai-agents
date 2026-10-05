import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

print("🌙 Moon Dev's KineticDivergence Backtest Initializing... ✨")
print("🚀 Loading cosmic data from the moon base... 🌙")

# Load data
data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper column mapping
data = data.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

print(f"🌙 Data loaded: {len(data)} bars from {data.index[0]} to {data.index[-1]} ✨")


class KineticDivergence(Strategy):
    # Strategy parameters
    rsi_period = 14
    atr_period = 14
    ema_period = 200
    pivot_window = 3
    range_lookback = 10
    atr_buffer_mult = 0.5
    sl_atr_mult = 1.5
    tp_rr = 2.0
    time_stop_bars = 20
    risk_pct = 0.0075  # 0.75% risk per trade
    volume_ma_period = 20
    volume_mult = 1.2

    def init(self):
        print("🌙 Initializing Moon Dev indicators... ✨")
        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        volume = pd.Series(self.data.Volume)

        # RSI
        self.rsi = self.I(talib.RSI, close, timeperiod=self.rsi_period)
        # ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)
        # 200 EMA
        self.ema200 = self.I(talib.EMA, close, timeperiod=self.ema_period)
        # Volume MA
        self.vol_ma = self.I(talib.SMA, volume, timeperiod=self.volume_ma_period)

        # Rolling range high/low
        self.range_high = self.I(talib.MAX, high, timeperiod=self.range_lookback)
        self.range_low = self.I(talib.MIN, low, timeperiod=self.range_lookback)

        # Fractal pivots (3-bar)
        pw = self.pivot_window
        self.pivot_high = self.I(talib.MAX, high, timeperiod=2 * pw + 1)
        self.pivot_low = self.I(talib.MIN, low, timeperiod=2 * pw + 1)

        # State tracking
        self.entry_bar = None
        self.entry_price = None
        self.stop_price = None
        self.tp_price = None
        self.trade_dir = None

        print("🌙 Indicators ready! Let's ride the kinetic waves! 🚀")

    def _find_pivots(self, series, kind='high'):
        """Return list of (index, value) for confirmed fractal pivots."""
        pivots = []
        pw = self.pivot_window
        vals = np.array(series)
        if len(vals) < 2 * pw + 1:
            return pivots
        for i in range(pw, len(vals) - pw):
            window = vals[i - pw:i + pw + 1]
            if kind == 'high':
                if vals[i] == window.max() and np.sum(window == vals[i]) == 1:
                    pivots.append((i, vals[i]))
            else:
                if vals[i] == window.min() and np.sum(window == vals[i]) == 1:
                    pivots.append((i, vals[i]))
        return pivots

    def _bullish_divergence(self, i):
        """Check bullish divergence: price lower low, RSI higher low."""
        if i < 20:
            return False
        lookback = 40
        start = max(0, i - lookback)
        low_arr = np.array(self.data.Low)
        rsi_arr = np.array(self.rsi)
        price_series = pd.Series(low_arr[start:i + 1])
        rsi_series = pd.Series(rsi_arr[start:i + 1])
        price_pivots = self._find_pivots(price_series, 'low')
        rsi_pivots = self._find_pivots(rsi_series, 'low')
        if len(price_pivots) < 2 or len(rsi_pivots) < 2:
            return False
        # Compare last two price pivots
        p1, p2 = price_pivots[-2], price_pivots[-1]
        r1, r2 = rsi_pivots[-2], rsi_pivots[-1]
        # Price makes lower low, RSI makes higher low
        if p2[1] < p1[1] and r2[1] > r1[1]:
            return True
        return False

    def _bearish_divergence(self, i):
        """Check bearish divergence: price higher high, RSI lower high."""
        if i < 20:
            return False
        lookback = 40
        start = max(0, i - lookback)
        high_arr = np.array(self.data.High)
        rsi_arr = np.array(self.rsi)
        price_series = pd.Series(high_arr[start:i + 1])
        rsi_series = pd.Series(rsi_arr[start:i + 1])
        price_pivots = self._find_pivots(price_series, 'high')
        rsi_pivots = self._find_pivots(rsi_series, 'high')
        if len(price_pivots) < 2 or len(rsi_pivots) < 2:
            return False
        p1, p2 = price_pivots[-2], price_pivots[-1]
        r1, r2 = rsi_pivots[-2], rsi_pivots[-1]
        if p2[1] > p1[1] and r2[1] < r1[1]:
            return True
        return False

    def _calc_size(self, entry, stop):
        """Return fractional size (0 < size < 1) representing % of equity."""
        risk_per_unit = abs(entry - stop)
        if risk_per_unit <= 0:
            return 0
        risk_amount = self.equity * self.risk_pct
        units = risk_amount / risk_per_unit
        frac = (units * entry) / self.equity
        if frac <= 0:
            return 0
        # Ensure minimum viable size
        frac = max(frac, 0.01)
        return min(frac, 0.95)

    def next(self):
        i = len(self.data) - 1
        if i < self.ema_period + 5:
            return

        price = self.data.Close[-1]
        atr = self.atr[-1]
        ema = self.ema200[-1]

        # Guard against NaN indicators
        if np.isnan(atr) or np.isnan(ema) or atr <= 0:
            return

        rng_hi = self.range_high[-1] + self.atr_buffer_mult * atr
        rng_lo = self.range_low[-1] - self.atr_buffer_mult * atr

        vol_ma_val = self.vol_ma[-1]
        vol_ok = self.data.Volume[-1] > self.volume_mult * vol_ma_val if (vol_ma_val and vol_ma_val > 0) else False

        # === Manage open position ===
        if self.position:
            bars_held = i - self.entry_bar
            # Time stop
            if bars_held >= self.time_stop_bars:
                print(f"⏰ Moon Dev Time Stop hit after {bars_held} bars! Closing. 🌙")
                self.position.close()
                self._reset_trade()
                return

            # Invalidation: close back inside range within 3 bars
            if bars_held <= 3:
                if self.trade_dir == 'long' and price < self.range_high[-1]:
                    print(f"❌ Long invalidation: price back inside range at {price:.2f} 🌙")
                    self.position.close()
                    self._reset_trade()
                    return
                if self.trade_dir == 'short' and price > self.range_low[-1]:
                    print(f"❌ Short invalidation: price back inside range at {price:.2f} 🌙")
                    self.position.close()
                    self._reset_trade()
                    return

            # Trail after 1R
            if self.trade_dir == 'long':
                r_dist = self.entry_price - self.stop_price
                if price >= self.entry_price + r_dist:
                    new_stop = price - 1.5 * atr
                    if new_stop > self.stop_price:
                        self.stop_price = new_stop
                if price <= self.stop_price:
                    print(f"🛑 Long stop hit at {price:.2f} 🌙")
                    self.position.close()
                    self._reset_trade()
                    return
                if price >= self.tp_price:
                    print(f"🎯 Long TP hit at {price:.2f}! Moon Dev profits! 🚀💰")
                    self.position.close()
                    self._reset_trade()
                    return
            else:
                r_dist = self.stop_price - self.entry_price
                if price <= self.entry_price - r_dist:
                    new_stop = price + 1.5 * atr
                    if new_stop < self.stop_price:
                        self.stop_price = new_stop
                if price >= self.stop_price:
                    print(f"🛑 Short stop hit at {price:.2f} 🌙")
                    self.position.close()
                    self._reset_trade()
                    return
                if price <= self.tp_price:
                    print(f"🎯 Short TP hit at {price:.2f}! Moon Dev profits! 🚀💰")
                    self.position.close()
                    self._reset_trade()
                    return
            return

        # === Entry logic ===
        if not vol_ok:
            return

        # Trend filter
        bull_trend = price > ema or price >= ema * 0.98
        bear_trend = price < ema

        # Long setup
        if bull_trend and self._bullish_divergence(i):
            if price > rng_hi:
                stop = min(self.range_low[-1], price - self.sl_atr_mult * atr)
                risk = price - stop
                if risk <= 0:
                    return
                tp = price + self.tp_rr * risk
                size = self._calc_size(price, stop)
                if size > 0:
                    print(f"🌙✨ BULLISH DIVERGENCE + BREAKOUT! Entry={price:.2f} SL={stop:.2f} TP={tp:.2f} Size={size:.4f} 🚀")
                    self.buy(size=size)
                    self.entry_bar = i
                    self.entry_price = price
                    self.stop_price = stop
                    self.tp_price = tp
                    self.trade_dir = 'long'
                    return

        # Short setup
        if bear_trend and self._bearish_divergence(i):
            if price < rng_lo:
                stop = max(self.range_high[-1], price + self.sl_atr_mult * atr)
                risk = stop - price
                if risk <= 0:
                    return
                tp = price - self.tp_rr * risk
                size = self._calc_size(price, stop)
                if size > 0:
                    print(f"🌙✨ BEARISH DIVERGENCE + BREAKOUT! Entry={price:.2f} SL={stop:.2f} TP={tp:.2f} Size={size:.4f} 🚀")
                    self.sell(size=size)
                    self.entry_bar = i
                    self.entry_price = price
                    self.stop_price = stop
                    self.tp_price = tp
                    self.trade_dir = 'short'
                    return

    def _reset_trade(self):
        self.entry_bar = None
        self.entry_price = None
        self.stop_price = None
        self.tp_price = None
        self.trade_dir = None


print("🌙 Launching Moon Dev backtest... 🚀")
bt = Backtest(data, KineticDivergence, cash=1_000_000, commission=0.0002)
stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙✨ Moon Dev backtest complete! To the moon! 🚀🌕")