import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# ============================================================
# 🌙 Moon Dev's CompressionBurst Strategy 🌙
# ============================================================

# Load and clean data
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

print("🌙✨ Moon Dev Data Loaded Successfully! ✨🌙")
print(f"📊 Data shape: {data.shape}")
print(f"📈 Date range: {data.index[0]} to {data.index[-1]}")


class CompressionBurst(Strategy):
    # Strategy parameters
    bb_period = 20
    bb_std = 2.0
    atr_period = 20
    squeeze_multiplier = 1.0
    trail_pct = 0.65
    target_multiple = 2.0
    risk_pct = 0.01
    use_volume_filter = True
    volume_ma_period = 20
    use_ema_filter = False
    ema_fast = 50
    ema_slow = 200

    def init(self):
        print("🌙 Moon Dev initializing CompressionBurst indicators... 🚀")

        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        volume = pd.Series(self.data.Volume)

        # Bollinger Bands - capture only upper and lower bands
        def _bb_upper(close, period, std):
            u, m, l = talib.BBANDS(close, timeperiod=period, nbdevup=std, nbdevdn=std, matype=0)
            return u

        def _bb_lower(close, period, std):
            u, m, l = talib.BBANDS(close, timeperiod=period, nbdevup=std, nbdevdn=std, matype=0)
            return l

        self.bb_upper_band = self.I(_bb_upper, close, self.bb_period, self.bb_std)
        self.bb_lower_band = self.I(_bb_lower, close, self.bb_period, self.bb_std)

        # ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        # BandWidth (raw)
        self.bandwidth = self.I(lambda u, l: u - l, self.bb_upper_band, self.bb_lower_band)

        # Volume SMA
        self.vol_sma = self.I(talib.SMA, volume, timeperiod=self.volume_ma_period)

        # EMA filters
        self.ema_fast_line = self.I(talib.EMA, close, timeperiod=self.ema_fast)
        self.ema_slow_line = self.I(talib.EMA, close, timeperiod=self.ema_slow)

        # Track trade state
        self.entry_price_state = None
        self.initial_stop = None
        self.trail_high = None
        self.candle_range = None
        self.take_profit = None

        print("🌙✨ Indicators initialized! Let's catch some bursts! 🚀")

    def next(self):
        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        volume = self.data.Volume[-1]

        # Skip if indicators not ready
        min_len = max(self.bb_period, self.atr_period, self.volume_ma_period, self.ema_slow) + 2
        if len(self.data) < min_len:
            return

        # Guard against NaN indicators
        if (np.isnan(self.bandwidth[-1]) or np.isnan(self.atr[-1]) or
                np.isnan(self.bb_upper_band[-1]) or np.isnan(self.vol_sma[-1])):
            return

        # =====================================================
        # MANAGE EXISTING POSITION
        # =====================================================
        if self.position:
            # Update trailing high
            if self.trail_high is None or high > self.trail_high:
                self.trail_high = high

            # Compute new trailing stop (only ratchets up)
            if self.candle_range is not None and self.initial_stop is not None:
                new_stop = self.trail_high - (self.candle_range * self.trail_pct)
                if new_stop > self.initial_stop:
                    self.initial_stop = new_stop

            # Check trailing stop hit
            if self.initial_stop is not None and low <= self.initial_stop:
                print(f"🛑 Moon Dev EXIT (trailing stop) at {self.initial_stop:.2f} | Entry: {self.entry_price_state:.2f}")
                self.position.close()
                self._reset_state()
                return

            # Check take profit
            if self.take_profit and high >= self.take_profit:
                print(f"🎯 Moon Dev EXIT (take profit) at {self.take_profit:.2f} | Entry: {self.entry_price_state:.2f}")
                self.position.close()
                self._reset_state()
                return

            return

        # =====================================================
        # ENTRY LOGIC
        # =====================================================
        # Condition A: Squeeze (bandwidth < ATR * multiplier)
        squeeze = self.bandwidth[-1] < (self.atr[-1] * self.squeeze_multiplier)

        # Condition B: Breakout (close > upper BB)
        breakout = price > self.bb_upper_band[-1]

        # Condition C: Filters
        volume_ok = (not self.use_volume_filter) or (volume > self.vol_sma[-1])
        ema_ok = (not self.use_ema_filter) or (self.ema_fast_line[-1] > self.ema_slow_line[-1])

        if squeeze and breakout and volume_ok and ema_ok:
            entry = price
            R = high - low

            if R <= 0:
                return

            initial_stop = entry - (R * self.trail_pct)
            risk_per_unit = entry - initial_stop

            if risk_per_unit <= 0:
                return

            # Position sizing: risk_pct of equity as a fraction of equity
            equity = self.equity
            risk_amount = equity * self.risk_pct
            position_units = risk_amount / risk_per_unit

            # Convert units to a fraction of equity for percentage-based sizing
            size_fraction = (position_units * entry) / equity

            # Clamp to valid fraction range (0 < size < 1)
            if size_fraction <= 0:
                return
            if size_fraction >= 1:
                size_fraction = 0.99

            print(f"🚀🌙 Moon Dev LONG SIGNAL! Entry: {entry:.2f} | R: {R:.2f} | Stop: {initial_stop:.2f} | Size: {size_fraction:.4f}")

            self.buy(size=size_fraction)

            # Record trade state
            self.entry_price_state = entry
            self.initial_stop = initial_stop
            self.trail_high = high
            self.candle_range = R
            self.take_profit = entry + (R * self.target_multiple)

    def _reset_state(self):
        self.entry_price_state = None
        self.initial_stop = None
        self.trail_high = None
        self.candle_range = None
        self.take_profit = None


# ============================================================
# RUN BACKTEST
# ============================================================
print("🌙✨ Starting Moon Dev CompressionBurst Backtest... 🚀")
bt = Backtest(
    data,
    CompressionBurst,
    cash=1_000_000,
    commission=0.002,
    exclusive_orders=True
)

stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙✨ Moon Dev Backtest Complete! ✨🌙")