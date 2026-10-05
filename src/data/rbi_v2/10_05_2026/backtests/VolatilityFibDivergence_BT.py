import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Load data
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Ensure proper column mapping
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

print("🌙 Moon Dev: Data loaded and cleaned! Shape:", data.shape)
print("🌙 Moon Dev: Columns:", list(data.columns))
print("🌙 Moon Dev: First few rows:\n", data.head())


class VolatilityFibDivergence(Strategy):
    # Strategy parameters
    iv_lookback = 20          # rolling high lookback for IV proxy (ATR percentile)
    atr_period = 14
    rsi_period = 14
    swing_lookback = 10       # bars to look back for swing lows
    divergence_min_delta = 5  # minimum RSI delta for strong divergence
    fib_382 = 0.382
    fib_500 = 0.500
    fib_618 = 0.618
    fib_1000 = 1.000
    fib_1618 = 1.618
    risk_pct = 0.02           # 2% risk per trade
    time_stop_bars = 20
    vol_bar_mult = 1.5        # volatility bar = range > 1.5x ATR

    def init(self):
        # ATR for volatility bars and stops
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close,
                          timeperiod=self.atr_period)
        # RSI for divergence
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=self.rsi_period)
        # Rolling high of ATR as IV proxy resistance
        self.atr_high = self.I(talib.MAX, self.atr, timeperiod=self.iv_lookback)
        # Rolling low of price for swing low detection
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=self.swing_lookback)
        # Track bar count
        self.bar_count = 0
        # Trade state
        self.entry_price = None
        self.stop_price = None
        self.targets_hit = set()
        self.entry_bar = None
        self.swing_low_price = None
        self.swing_high_price = None
        print("🌙 Moon Dev: Indicators initialized! ✨")

    def next(self):
        self.bar_count += 1

        # Skip if not enough data
        if len(self.data) < max(self.iv_lookback, self.atr_period, self.rsi_period, self.swing_lookback) + 5:
            return

        price = self.data.Close[-1]
        atr_val = self.atr[-1]
        rsi_val = self.rsi[-1]
        atr_high_val = self.atr_high[-1]

        # --- Manage open position ---
        if self.position:
            self._manage_position(price, atr_val)
            return

        # --- IV Breakout Check ---
        # IV proxy = ATR percentile; breakout when ATR closes above its rolling high
        iv_breakout = atr_val >= atr_high_val and atr_val > 0

        # --- Bullish Divergence Detection ---
        # Price makes lower low, RSI makes higher low
        # Compare current swing low vs prior swing low
        lookback = self.swing_lookback
        if len(self.data) < 2 * lookback + 2:
            return

        # Current swing low region
        current_low = self.data.Low[-1]
        prior_low_idx = -lookback - 1
        prior_low = self.data.Low[prior_low_idx]
        prior_rsi = self.rsi[prior_low_idx]

        # Find local swing low in recent window
        recent_lows = self.data.Low[-lookback:]
        recent_rsi = self.rsi[-lookback:]
        min_low_idx = int(np.argmin(recent_lows))
        swing_low_price = recent_lows[min_low_idx]
        swing_low_rsi = recent_rsi[min_low_idx]

        # Bullish divergence: price lower low, RSI higher low
        price_lower_low = swing_low_price < prior_low
        rsi_higher_low = swing_low_rsi > prior_rsi
        rsi_delta = swing_low_rsi - prior_rsi
        strong_divergence = price_lower_low and rsi_higher_low and rsi_delta > self.divergence_min_delta

        # Confirmation: current candle closes bullish
        bullish_close = self.data.Close[-1] > self.data.Open[-1]

        # --- Entry ---
        if iv_breakout and strong_divergence and bullish_close:
            # Hard stop: below swing low or 1.5x ATR, whichever is tighter
            stop_swing = swing_low_price
            stop_atr = price - (1.5 * atr_val)
            stop_price = max(stop_swing, stop_atr)  # tighter = higher for long

            risk_per_unit = price - stop_price
            if risk_per_unit <= 0:
                print("🌙 Moon Dev: Invalid stop, skipping entry 🚫")
                return

            # Position sizing: risk 2% of equity
            risk_amount = self.equity * self.risk_pct
            position_size = risk_amount / risk_per_unit
            position_size = int(round(position_size))

            if position_size < 1:
                print("🌙 Moon Dev: Position size too small, skipping 🚫")
                return

            # Fibonacci targets anchored to swing low -> recent swing high
            recent_highs = self.data.High[-lookback:]
            swing_high_price = np.max(recent_highs)
            fib_range = swing_high_price - swing_low_price

            self.entry_price = price
            self.stop_price = stop_price
            self.swing_low_price = swing_low_price
            self.swing_high_price = swing_high_price
            self.fib_range = fib_range
            self.targets_hit = set()
            self.entry_bar = self.bar_count

            self.buy(size=position_size)
            print(f"🚀 Moon Dev: LONG ENTRY! Price={price:.2f} Size={position_size} "
                  f"Stop={stop_price:.2f} RSI={rsi_val:.2f} IV_Breakout={iv_breakout} "
                  f"Divergence={strong_divergence} ✨")

    def _manage_position(self, price, atr_val):
        # Time stop
        bars_in_trade = self.bar_count - self.entry_bar
        if bars_in_trade >= self.time_stop_bars and len(self.targets_hit) == 0:
            print(f"⏰ Moon Dev: TIME STOP exit at {price:.2f} after {bars_in_trade} bars")
            self.position.close()
            return

        # Hard stop
        if price <= self.stop_price:
            print(f"🛑 Moon Dev: STOP LOSS hit at {price:.2f} (stop={self.stop_price:.2f})")
            self.position.close()
            return

        # Volatility bar exit: opposing bar with range > 1.5x ATR
        bar_range = self.data.High[-1] - self.data.Low[-1]
        bearish_bar = self.data.Close[-1] < self.data.Open[-1]
        if bearish_bar and bar_range > self.vol_bar_mult * atr_val:
            print(f"📉 Moon Dev: VOLATILITY BAR exit at {price:.2f} (range={bar_range:.2f})")
            self.position.close()
            return

        # IV collapse: ATR falls back below its prior resistance
        if atr_val < self.atr_high[-2]:
            print(f"🌊 Moon Dev: IV COLLAPSE exit at {price:.2f}")
            self.position.close()
            return

        # Fibonacci scale-out targets
        entry = self.entry_price
        fib_range = self.fib_range

        # Fib levels as price targets (long: target = swing_low + fib * range)
        fib_targets = {
            '0.382': self.swing_low_price + self.fib_382 * fib_range,
            '0.500': self.swing_low_price + self.fib_500 * fib_range,
            '0.618': self.swing_low_price + self.fib_618 * fib_range,
            '1.000': self.swing_low_price + self.fib_1000 * fib_range,
            '1.618': self.swing_low_price + self.fib_1618 * fib_range,
        }

        for label, target in fib_targets.items():
            if label not in self.targets_hit and price >= target:
                self.targets_hit.add(label)
                if label in ('0.382', '0.500', '0.618'):
                    # Scale out 25% at each intermediate target
                    close_size = max(1, int(round(self.position.size * 0.25)))
                    self.position.close(portion=0.25)
                    print(f"🎯 Moon Dev: FIB {label} TARGET hit at {price:.2f}! Scaled out 25% ✨")
                else:
                    # Final targets: close everything
                    self.position.close()
                    print(f"🏆 Moon Dev: FIB {label} FINAL TARGET hit at {price:.2f}! Full exit 🚀")
                    return


# Run backtest
bt = Backtest(data, VolatilityFibDivergence, cash=1_000_000, commission=0.002)

print("🌙 Moon Dev: Starting backtest... 🚀")
stats = bt.run()
print(stats)
print(stats._strategy)