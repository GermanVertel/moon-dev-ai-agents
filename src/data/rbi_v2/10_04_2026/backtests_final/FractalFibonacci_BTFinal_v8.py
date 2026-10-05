import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# ============================================================
# 🌙 MOON DEV - FRACTAL FIBONACCI STRATEGY 🌙
# ============================================================

print("🌙✨ Moon Dev is loading the data... ✨🌙")

data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

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
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

print(f"🚀 Data loaded: {len(data)} bars 🌙")


class FractalFibonacci(Strategy):
    # Strategy parameters
    fractal_lookback = 50
    fib_tolerance = 0.02  # 2% - widened to allow more trades
    atr_period = 14
    atr_mult_tp = 2.0
    atr_mult_sl = 1.0
    atr_trail_trigger = 1.5
    atr_trail_dist = 0.75
    risk_pct = 0.02
    rsi_period = 14
    ema_trend = 200
    min_rr = 1.5

    def init(self):
        print("🌙 Initializing FractalFibonacci indicators... ✨")
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close,
                          timeperiod=self.atr_period)
        self.atr_ma = self.I(talib.SMA, self.atr, timeperiod=20)
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=self.rsi_period)
        self.macd, self.macd_signal, self.macd_hist = self.I(
            talib.MACD, self.data.Close, fastperiod=12, slowperiod=26, signalperiod=9
        )
        self.ema200 = self.I(talib.EMA, self.data.Close, timeperiod=self.ema_trend)

        # Fractal highs and lows (5-bar pattern)
        self.fractal_high = self.I(self._fractal_high, self.data.High)
        self.fractal_low = self.I(self._fractal_low, self.data.Low)

        # Track trade state
        self.entry_price_val = None
        self.stop_price = None
        self.tp_price = None
        self.trail_active = False
        self.entry_bar = None
        self.trade_dir = None

    @staticmethod
    def _fractal_high(high):
        n = len(high)
        out = np.full(n, np.nan)
        for i in range(2, n - 2):
            if (high[i] > high[i-1] and high[i] > high[i-2] and
                    high[i] > high[i+1] and high[i] > high[i+2]):
                out[i] = high[i]
        return out

    @staticmethod
    def _fractal_low(low):
        n = len(low)
        out = np.full(n, np.nan)
        for i in range(2, n - 2):
            if (low[i] < low[i-1] and low[i] < low[i-2] and
                    low[i] < low[i+1] and low[i] < low[i+2]):
                out[i] = low[i]
        return out

    def _recent_swing(self):
        """Find most recent significant swing high and low within lookback."""
        i = len(self.data) - 1
        start = max(0, i - self.fractal_lookback)
        highs = self.fractal_high[start:i+1]
        lows = self.fractal_low[start:i+1]
        h_idx = None
        l_idx = None
        for j in range(len(highs) - 1, -1, -1):
            if not np.isnan(highs[j]):
                h_idx = start + j
                break
        for j in range(len(lows) - 1, -1, -1):
            if not np.isnan(lows[j]):
                l_idx = start + j
                break
        if h_idx is None or l_idx is None:
            return None, None
        return self.data.High[h_idx], self.data.Low[l_idx]

    def _near_fib(self, price, swing_high, swing_low):
        """Check if price is near a key Fibonacci level."""
        rng = swing_high - swing_low
        if rng <= 0:
            return None
        fibs = {
            '23.6': swing_high - 0.236 * rng,
            '38.2': swing_high - 0.382 * rng,
            '50.0': swing_high - 0.500 * rng,
            '61.8': swing_high - 0.618 * rng,
            '78.6': swing_high - 0.786 * rng,
        }
        for name, level in fibs.items():
            if abs(price - level) / price <= self.fib_tolerance:
                return name, level
        return None

    def _bullish_reversal(self, i):
        o, c = self.data.Open[i], self.data.Close[i]
        h, l = self.data.High[i], self.data.Low[i]
        po, pc = self.data.Open[i-1], self.data.Close[i-1]
        body = abs(c - o)
        rng = h - l
        if rng == 0:
            return False
        # Bullish engulfing
        if c > o and pc < po and c > po and o < pc:
            return True
        # Hammer / pin bar
        lower_wick = min(o, c) - l
        if body > 0 and lower_wick > 2 * body and (c - l) / rng > 0.6:
            return True
        return False

    def _bearish_reversal(self, i):
        o, c = self.data.Open[i], self.data.Close[i]
        h, l = self.data.High[i], self.data.Low[i]
        po, pc = self.data.Open[i-1], self.data.Close[i-1]
        body = abs(c - o)
        rng = h - l
        if rng == 0:
            return False
        # Bearish engulfing
        if c < o and pc > po and c < po and o > pc:
            return True
        # Shooting star
        upper_wick = h - max(o, c)
        if body > 0 and upper_wick > 2 * body and (h - c) / rng > 0.6:
            return True
        return False

    def _rsi_bullish_div(self, i):
        # Simplified: RSI rising while price near lows
        if i < 5:
            return False
        return self.rsi[i] > self.rsi[i-1] and self.rsi[i] < 45

    def _rsi_bearish_div(self, i):
        if i < 5:
            return False
        return self.rsi[i] < self.rsi[i-1] and self.rsi[i] > 55

    def next(self):
        i = len(self.data) - 1
        if i < self.ema_trend + 5:
            return

        price = self.data.Close[i]
        atr = self.atr[i]
        if np.isnan(atr) or atr <= 0:
            return

        # Manage existing position
        if self.position:
            self._manage_position(i, price, atr)
            return

        swing_high, swing_low = self._recent_swing()
        if swing_high is None or swing_low is None:
            return

        fib_hit = self._near_fib(price, swing_high, swing_low)
        if fib_hit is None:
            return
        fib_name, fib_level = fib_hit

        # Only 50-78.6% for entries
        if fib_name not in ('50.0', '61.8', '78.6'):
            return

        # Long setup
        long_ok = (
            not np.isnan(self.fractal_low[i-1])
            and self._bullish_reversal(i)
            and (self._rsi_bullish_div(i) or self.macd[i] > self.macd_signal[i])
            and price > self.ema200[i]
        )
        # Short setup
        short_ok = (
            not np.isnan(self.fractal_high[i-1])
            and self._bearish_reversal(i)
            and (self._rsi_bearish_div(i) or self.macd[i] < self.macd_signal[i])
            and price < self.ema200[i]
        )

        if long_ok:
            sl = self.fractal_low[i-1] - atr * self.atr_mult_sl
            risk = price - sl
            if risk <= 0:
                return
            tp = price + atr * self.atr_mult_tp
            if (tp - price) / risk < self.min_rr:
                return
            # Use fraction of equity for sizing (valid size parameter)
            size = 0.5
            print(f"🌙✨ LONG signal at {price:.2f} | Fib {fib_name} | SL {sl:.2f} | TP {tp:.2f} 🚀")
            self.buy(size=size, sl=sl, tp=tp)
            self.entry_price_val = price
            self.stop_price = sl
            self.tp_price = tp
            self.entry_bar = i
            self.trade_dir = 'long'
            self.trail_active = False

        elif short_ok:
            sl = self.fractal_high[i-1] + atr * self.atr_mult_sl
            risk = sl - price
            if risk <= 0:
                return
            tp = price - atr * self.atr_mult_tp
            if (price - tp) / risk < self.min_rr:
                return
            size = 0.5
            print(f"🌙✨ SHORT signal at {price:.2f} | Fib {fib_name} | SL {sl:.2f} | TP {tp:.2f} 🚀")
            self.sell(size=size, sl=sl, tp=tp)
            self.entry_price_val = price
            self.stop_price = sl
            self.tp_price = tp
            self.entry_bar = i
            self.trade_dir = 'short'
            self.trail_active = False

    def _manage_position(self, i, price, atr):
        # ATR spike exit
        if not np.isnan(self.atr_ma[i]) and self.atr[i] > 1.5 * self.atr_ma[i]:
            print(f"🌙⚡ ATR spike detected - exiting position 🚀")
            self.position.close()
            return

        # Emergency counter-fractal exit within 3 bars
        if self.entry_bar is not None and (i - self.entry_bar) <= 3:
            if self.trade_dir == 'long' and not np.isnan(self.fractal_high[i]):
                print(f"🌙🚨 Counter-fractal formed - emergency exit 🚀")
                self.position.close()
                return
            if self.trade_dir == 'short' and not np.isnan(self.fractal_low[i]):
                print(f"🌙🚨 Counter-fractal formed - emergency exit 🚀")
                self.position.close()
                return

        # Trailing stop
        if self.entry_price_val is None:
            return
        if self.trade_dir == 'long':
            move = price - self.entry_price_val
            if move >= self.atr_trail_trigger * atr:
                new_sl = price - self.atr_trail_dist * atr
                if new_sl > self.stop_price:
                    self.stop_price = new_sl
                    self.trail_active = True
        elif self.trade_dir == 'short':
            move = self.entry_price_val - price
            if move >= self.atr_trail_trigger * atr:
                new_sl = price + self.atr_trail_dist * atr
                if new_sl < self.stop_price:
                    self.stop_price = new_sl
                    self.trail_active = True


print("🌙✨ Running backtest... 🚀")
bt = Backtest(data, FractalFibonacci, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙✨ Backtest complete! 🚀")