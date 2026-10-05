import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Load and prepare data
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Rename to proper case for backtesting.py
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

# Set datetime index
if 'datetime' in data.columns:
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')

print("🌙✨ Moon Dev Backtest Initializing... ✨🌙")
print(f"📊 Data shape: {data.shape}")
print(f"📈 Data head:\n{data.head()}")


class FiboDivergence(Strategy):
    # Strategy parameters
    rsi_period = 14
    atr_period = 14
    swing_lookback = 20
    fib_tolerance = 0.005  # 0.5% tolerance for fib zone
    risk_pct = 0.02
    rr_ratio = 2.0
    max_bars_in_trade = 20
    divergence_lookback = 10

    def init(self):
        print("🌙 Initializing indicators...")
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=self.rsi_period)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close,
                          timeperiod=self.atr_period)
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=self.swing_lookback)
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=self.swing_lookback)
        self.entry_bar = None
        self.breakeven_set = False
        print("✅ Indicators ready!")

    def _fib_levels(self, swing_high, swing_low):
        """Calculate Fibonacci retracement levels."""
        diff = swing_high - swing_low
        return {
            '38.2': swing_high - 0.382 * diff,
            '50.0': swing_high - 0.500 * diff,
            '61.8': swing_high - 0.618 * diff,
            '78.6': swing_high - 0.786 * diff,
        }

    def _near_fib(self, price, fibs):
        """Check if price is near any fib level."""
        for level, val in fibs.items():
            if abs(price - val) / val < self.fib_tolerance:
                return level, val
        return None, None

    def _bullish_divergence(self, i):
        """Price makes lower low, RSI makes higher low."""
        lb = self.divergence_lookback
        if i < lb * 2:
            return False
        # recent window
        recent_window = self.data.Low[i - lb:i]
        if len(recent_window) == 0:
            return False
        recent_low_idx = i - lb + int(np.argmin(recent_window))
        # prior window
        prior_window = self.data.Low[i - 2 * lb:i - lb]
        if len(prior_window) == 0:
            return False
        prior_low_idx = i - 2 * lb + int(np.argmin(prior_window))

        # bounds check
        if recent_low_idx >= len(self.data.Low) or prior_low_idx >= len(self.data.Low):
            return False

        price_ll = self.data.Low[recent_low_idx] < self.data.Low[prior_low_idx]
        rsi_hl = self.rsi[recent_low_idx] > self.rsi[prior_low_idx]
        return price_ll and rsi_hl

    def _bearish_divergence(self, i):
        """Price makes higher high, RSI makes lower high."""
        lb = self.divergence_lookback
        if i < lb * 2:
            return False
        recent_window = self.data.High[i - lb:i]
        if len(recent_window) == 0:
            return False
        recent_high_idx = i - lb + int(np.argmax(recent_window))
        prior_window = self.data.High[i - 2 * lb:i - lb]
        if len(prior_window) == 0:
            return False
        prior_high_idx = i - 2 * lb + int(np.argmax(prior_window))

        # bounds check
        if recent_high_idx >= len(self.data.High) or prior_high_idx >= len(self.data.High):
            return False

        price_hh = self.data.High[recent_high_idx] > self.data.High[prior_high_idx]
        rsi_lh = self.rsi[recent_high_idx] < self.rsi[prior_high_idx]
        return price_hh and rsi_lh

    def next(self):
        i = len(self.data) - 1

        # Skip if already in a position
        if self.position:
            self._manage_position()
            return

        # Need enough bars
        if i < max(self.swing_lookback, self.atr_period, self.rsi_period) + 5:
            return

        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        atr = self.atr[-1]

        if np.isnan(atr) or atr <= 0:
            return

        sh = self.swing_high[-1]
        sl = self.swing_low[-1]
        if np.isnan(sh) or np.isnan(sl) or sh <= sl:
            return

        fibs = self._fib_levels(sh, sl)

        # ---- Long setup ----
        level, fib_price = self._near_fib(low, fibs)
        if level and self._bullish_divergence(i):
            # confirmation candle: bullish close above prior high
            confirmation = self.data.Close[-1] > self.data.High[-2]
            if confirmation:
                stop = price - 1.5 * atr
                risk = price - stop
                if risk <= 0:
                    return
                # ensure swing low isn't too far below (stop not too wide)
                if price - sl > 3 * atr:
                    return
                target = price + self.rr_ratio * risk
                size = 0.99
                print(f"🚀🌙 LONG signal @ {price:.2f} | Fib {level} ({fib_price:.2f}) | "
                      f"SL={stop:.2f} TP={target:.2f} ATR={atr:.2f}")
                self.buy(size=size, sl=stop, tp=target)
                self.entry_bar = i
                self.breakeven_set = False
                return

        # ---- Short setup ----
        level, fib_price = self._near_fib(high, fibs)
        if level and self._bearish_divergence(i):
            confirmation = self.data.Close[-1] < self.data.Low[-2]
            if confirmation:
                stop = price + 1.5 * atr
                risk = stop - price
                if risk <= 0:
                    return
                if sh - price > 3 * atr:
                    return
                target = price - self.rr_ratio * risk
                size = 0.99
                print(f"🔻🌙 SHORT signal @ {price:.2f} | Fib {level} ({fib_price:.2f}) | "
                      f"SL={stop:.2f} TP={target:.2f} ATR={atr:.2f}")
                self.sell(size=size, sl=stop, tp=target)
                self.entry_bar = i
                self.breakeven_set = False
                return

    def _manage_position(self):
        """Move stop to breakeven after 1x ATR profit; time-based exit."""
        i = len(self.data) - 1
        atr = self.atr[-1]
        if np.isnan(atr):
            return

        entry_price = self.trades[-1].entry_price if self.trades else None
        if entry_price is None:
            return

        # Breakeven move
        if not getattr(self, 'breakeven_set', False):
            if self.position.is_long and self.data.High[-1] >= entry_price + atr:
                self.position.sl = entry_price
                self.breakeven_set = True
                print(f"✨🌙 Moved LONG stop to breakeven @ {entry_price:.2f}")
            elif self.position.is_short and self.data.Low[-1] <= entry_price - atr:
                self.position.sl = entry_price
                self.breakeven_set = True
                print(f"✨🌙 Moved SHORT stop to breakeven @ {entry_price:.2f}")

        # Time-based exit
        if self.entry_bar is not None and (i - self.entry_bar) >= self.max_bars_in_trade:
            self.position.close()
            print(f"⏰🌙 Time-based exit after {self.max_bars_in_trade} bars")


print("🌙✨ Running Moon Dev FiboDivergence Backtest... ✨🌙")
bt = Backtest(data, FiboDivergence, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)