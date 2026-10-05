import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

print("🌙✨ Moon Dev Backtest AI initializing... 🚀")
print("📊 Loading ConvergentRetracement strategy...")

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
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')

print(f"✅ Data loaded: {len(data)} bars from {data.index[0]} to {data.index[-1]}")
print(f"🌙 Moon Dev preparing indicators... ✨")


class ConvergentRetracement(Strategy):
    # Strategy parameters
    ema_fast_period = 50
    ema_slow_period = 200
    bb_period = 20
    bb_std = 2.0
    atr_period = 14
    swing_lookback = 50
    bbwp_lookback = 100
    bbwp_threshold = 25
    fib_low = 0.382
    fib_high = 0.786
    risk_pct = 0.01
    rr_min = 2.0
    max_hold_bars = 15

    def init(self):
        print("🌙 Moon Dev initializing indicators... 🚀")
        close = self.data.Close
        high = self.data.High
        low = self.data.Low

        # EMAs for trend
        self.ema_fast = self.I(talib.EMA, close, timeperiod=self.ema_fast_period)
        self.ema_slow = self.I(talib.EMA, close, timeperiod=self.ema_slow_period)

        # Bollinger Bands
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS, close, timeperiod=self.bb_period,
            nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0
        )

        # ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        # Swing highs/lows
        self.swing_high = self.I(talib.MAX, high, timeperiod=self.swing_lookback)
        self.swing_low = self.I(talib.MIN, low, timeperiod=self.swing_lookback)

        # Bollinger Band Width
        def bb_width_func(u, m, l):
            return (u - m * 0 + (u - l)) / m if False else (u - l) / m
        # Simpler: wrap as lambda over arrays
        self.bb_width = self.I(
            lambda u, m, l: (u - l) / m,
            self.bb_upper, self.bb_middle, self.bb_lower
        )

        # BBWP: percentile rank of bb_width over bbwp_lookback
        def bbwp_func(w):
            out = np.full(len(w), np.nan)
            for i in range(self.bbwp_lookback, len(w)):
                window = w[i - self.bbwp_lookback:i + 1]
                if np.any(np.isnan(window)) or np.isnan(w[i]):
                    continue
                out[i] = (np.sum(window <= w[i]) / len(window)) * 100
            return out

        self.bbwp = self.I(bbwp_func, self.bb_width)

        # Volume
        self.vol_sma = self.I(talib.SMA, self.data.Volume, timeperiod=20)

        print("✨ Moon Dev indicators ready! 🌙")

    def next(self):
        # Need enough bars
        if len(self.data) < max(self.ema_slow_period, self.bbwp_lookback) + 5:
            return

        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        open_ = self.data.Open[-1]

        ema_f = self.ema_fast[-1]
        ema_s = self.ema_slow[-1]
        atr = self.atr[-1]
        bbwp = self.bbwp[-1]
        bb_w = self.bb_width[-1]

        if np.isnan(atr) or np.isnan(ema_f) or np.isnan(ema_s) or np.isnan(bbwp):
            return

        # Trend
        uptrend = price > ema_f and price > ema_s and ema_f > ema_s
        downtrend = price < ema_f and price < ema_s and ema_f < ema_s

        # Squeeze condition
        squeeze = bbwp < self.bbwp_threshold

        # Manage existing position
        if self.position:
            self.position.bars_held = getattr(self.position, 'bars_held', 0) + 1
            bars_held = self.position.bars_held

            # Time-based exit
            if bars_held >= self.max_hold_bars:
                print(f"⏰ Moon Dev time exit after {bars_held} bars 🌙")
                self.position.close()
                return

            # Trailing stop with 1.5x ATR
            if self.position.is_long:
                trail = price - 1.5 * atr
                if trail > getattr(self.position, 'trail_stop', 0):
                    self.position.trail_stop = trail
                if price <= getattr(self.position, 'trail_stop', 0):
                    print(f"🛑 Moon Dev trailing stop hit LONG @ {price:.2f} 🌙")
                    self.position.close()
            elif self.position.is_short:
                trail = price + 1.5 * atr
                if trail < getattr(self.position, 'trail_stop', float('inf')):
                    self.position.trail_stop = trail
                if price >= getattr(self.position, 'trail_stop', float('inf')):
                    print(f"🛑 Moon Dev trailing stop hit SHORT @ {price:.2f} 🌙")
                    self.position.close()
            return

        # Entry logic
        swing_hi = self.swing_high[-1]
        swing_lo = self.swing_low[-1]
        rng = swing_hi - swing_lo
        if rng <= 0:
            return

        # Fib levels for uptrend retracement (from swing low to swing high)
        fib_382_long = swing_hi - 0.382 * rng
        fib_618_long = swing_hi - 0.618 * rng
        fib_786_long = swing_hi - 0.786 * rng

        # Fib levels for downtrend retracement (from swing high to swing low)
        fib_382_short = swing_lo + 0.382 * rng
        fib_618_short = swing_lo + 0.618 * rng
        fib_786_short = swing_lo + 0.786 * rng

        # Bullish confirmation candle
        bull_engulf = price > open_ and self.data.Close[-2] < self.data.Open[-2] and price > self.data.Open[-2]
        bull_pin = (min(open_, price) - low) > 2 * abs(price - open_) and price > open_

        # Bearish confirmation candle
        bear_engulf = price < open_ and self.data.Close[-2] > self.data.Open[-2] and price < self.data.Open[-2]
        bear_pin = (high - max(open_, price)) > 2 * abs(price - open_) and price < open_

        # LONG setup
        if uptrend and squeeze:
            in_fib_zone = fib_786_long <= price <= fib_382_long
            near_618 = abs(price - fib_618_long) < 0.5 * atr
            if in_fib_zone and (bull_engulf or bull_pin) and near_618:
                sl = min(fib_786_long, swing_lo) - 0.5 * atr
                risk = price - sl
                if risk > 0:
                    reward = (swing_hi - price) + (0.272 * rng)  # TP1 + extension
                    if reward / risk >= self.rr_min:
                        size = 0.99
                        print(f"🚀🌙 Moon Dev LONG signal @ {price:.2f} | SL {sl:.2f} | Fib618 {fib_618_long:.2f} | BBWP {bbwp:.1f}")
                        self.buy(size=size, sl=sl)
                        self.position.trail_stop = price - 1.5 * atr
                        self.position.bars_held = 0

        # SHORT setup
        if downtrend and squeeze:
            in_fib_zone = fib_382_short <= price <= fib_786_short
            near_618 = abs(price - fib_618_short) < 0.5 * atr
            if in_fib_zone and (bear_engulf or bear_pin) and near_618:
                sl = max(fib_786_short, swing_hi) + 0.5 * atr
                risk = sl - price
                if risk > 0:
                    reward = (price - swing_lo) + (0.272 * rng)
                    if reward / risk >= self.rr_min:
                        size = 0.99
                        print(f"🔻🌙 Moon Dev SHORT signal @ {price:.2f} | SL {sl:.2f} | Fib618 {fib_618_short:.2f} | BBWP {bbwp:.1f}")
                        self.sell(size=size, sl=sl)
                        self.position.trail_stop = price + 1.5 * atr
                        self.position.bars_held = 0


print("🌙 Moon Dev running backtest... 🚀")
bt = Backtest(data, ConvergentRetracement, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)
print("✨ Moon Dev backtest complete! 🌙")