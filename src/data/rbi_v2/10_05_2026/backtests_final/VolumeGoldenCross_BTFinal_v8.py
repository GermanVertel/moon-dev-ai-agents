import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

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
if 'datetime' in data.columns:
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')

# 🌙 Ensure all OHLCV columns are float64 for talib compatibility
for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    if col in data.columns:
        data[col] = data[col].astype('float64')

# Drop any NaN rows that would break talib
data = data.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])

print("🌙 Moon Dev Backtest AI - VolumeGoldenCross Strategy 🚀")
print(f"📊 Data loaded: {len(data)} bars")
print(f"📅 Date range: {data.index[0]} to {data.index[-1]}")
print("=" * 60)


class VolumeGoldenCross(Strategy):
    # Strategy parameters
    ema_fast_period = 50
    ema_slow_period = 200
    volume_sma_period = 20
    volume_multiplier = 1.5
    rsi_period = 14
    rsi_exit_level = 70
    atr_period = 14
    atr_multiplier = 1.5
    risk_reward_ratio = 2.0
    risk_pct = 0.02  # 2% risk per trade
    adx_period = 14
    adx_threshold = 20

    def init(self):
        print("🌙 Initializing Moon Dev indicators... ✨")

        # 🌙 Wrap talib calls so inputs are guaranteed float64 arrays
        def _sma(arr, timeperiod):
            return talib.SMA(np.asarray(arr, dtype=np.float64), timeperiod=timeperiod)

        def _ema(arr, timeperiod):
            return talib.EMA(np.asarray(arr, dtype=np.float64), timeperiod=timeperiod)

        def _rsi(arr, timeperiod):
            return talib.RSI(np.asarray(arr, dtype=np.float64), timeperiod=timeperiod)

        def _atr(high, low, close, timeperiod):
            return talib.ATR(
                np.asarray(high, dtype=np.float64),
                np.asarray(low, dtype=np.float64),
                np.asarray(close, dtype=np.float64),
                timeperiod=timeperiod,
            )

        def _adx(high, low, close, timeperiod):
            return talib.ADX(
                np.asarray(high, dtype=np.float64),
                np.asarray(low, dtype=np.float64),
                np.asarray(close, dtype=np.float64),
                timeperiod=timeperiod,
            )

        def _min(arr, timeperiod):
            return talib.MIN(np.asarray(arr, dtype=np.float64), timeperiod=timeperiod)

        # EMA indicators
        self.ema_fast = self.I(_ema, self.data.Close, timeperiod=self.ema_fast_period)
        self.ema_slow = self.I(_ema, self.data.Close, timeperiod=self.ema_slow_period)

        # Volume SMA
        self.volume_sma = self.I(_sma, self.data.Volume, timeperiod=self.volume_sma_period)

        # RSI
        self.rsi = self.I(_rsi, self.data.Close, timeperiod=self.rsi_period)

        # ATR for stop sizing
        self.atr = self.I(_atr, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period)

        # ADX for trend filter
        self.adx = self.I(_adx, self.data.High, self.data.Low, self.data.Close, timeperiod=self.adx_period)

        # Swing low for stop placement
        self.swing_low = self.I(_min, self.data.Low, timeperiod=20)

        # Track entry price, stop, target
        self.entry_price = None
        self.stop_price = None
        self.target_price = None

        print("🚀 Indicators ready! Let's ride the golden cross! 🌙")

    def next(self):
        # Skip if not enough data
        if len(self.data) < self.ema_slow_period + 5:
            return

        # Current values
        price = self.data.Close[-1]
        ema_fast_now = self.ema_fast[-1]
        ema_slow_now = self.ema_slow[-1]
        ema_fast_prev = self.ema_fast[-2]
        ema_slow_prev = self.ema_slow[-2]
        volume_now = self.data.Volume[-1]
        volume_avg = self.volume_sma[-1]
        rsi_now = self.rsi[-1]
        rsi_prev = self.rsi[-2]
        adx_now = self.adx[-1]
        atr_now = self.atr[-1]
        swing_low_now = self.swing_low[-1]

        # Check for NaN
        if (np.isnan(ema_fast_now) or np.isnan(ema_slow_now) or
            np.isnan(ema_fast_prev) or np.isnan(ema_slow_prev) or
            np.isnan(volume_avg) or np.isnan(rsi_now) or np.isnan(atr_now) or
            np.isnan(adx_now) or np.isnan(swing_low_now)):
            return

        # ==================== ENTRY LOGIC ====================
        if not self.position:
            # Golden cross: fast EMA crosses above slow EMA
            golden_cross = (ema_fast_prev <= ema_slow_prev) and (ema_fast_now > ema_slow_now)

            # Volume spike confirmation
            volume_spike = volume_now >= (self.volume_multiplier * volume_avg)

            # Trend filter: ADX > threshold and price above slow EMA
            trend_ok = adx_now > self.adx_threshold
            price_above_slow = price > ema_slow_now

            if golden_cross and volume_spike and trend_ok and price_above_slow:
                # Calculate stop: 1.5x ATR below entry, or swing low (whichever is higher/closer)
                atr_stop = price - (self.atr_multiplier * atr_now)
                swing_stop = swing_low_now
                stop = max(atr_stop, swing_stop)  # Higher = closer to entry = tighter stop

                risk_per_unit = price - stop

                if risk_per_unit > 0:
                    # 🌙 Moon Dev FIX: use fractional sizing (percentage of equity)
                    # instead of unit-based sizing so trades actually execute.
                    risk_amount = self.equity * self.risk_pct
                    units_needed = risk_amount / risk_per_unit
                    size_fraction = (units_needed * price) / self.equity

                    # Clamp to valid fraction (0 < size < 1)
                    size_fraction = min(max(size_fraction, 0.0001), 0.99)

                    target = price + (self.risk_reward_ratio * risk_per_unit)

                    self.entry_price = price
                    self.stop_price = stop
                    self.target_price = target

                    print(f"🌙✨ GOLDEN CROSS DETECTED! ✨🌙")
                    print(f"   💰 Entry: {price:.2f}")
                    print(f"   🛑 Stop: {stop:.2f} (risk: {risk_per_unit:.2f})")
                    print(f"   🎯 Target: {target:.2f} (R:R = {self.risk_reward_ratio}:1)")
                    print(f"   📊 Volume: {volume_now:.2f} vs avg {volume_avg:.2f} ({volume_now/volume_avg:.2f}x)")
                    print(f"   📈 RSI: {rsi_now:.2f} | ADX: {adx_now:.2f}")
                    print(f"   🚀 Size fraction: {size_fraction:.4f} of equity")

                    self.buy(size=size_fraction)
                    return

        # ==================== EXIT LOGIC ====================
        if self.position:
            # Primary exit: RSI crosses below 70 from above
            rsi_exit = (rsi_prev >= self.rsi_exit_level) and (rsi_now < self.rsi_exit_level)

            # Stop loss hit
            stop_hit = price <= self.stop_price if self.stop_price is not None else False

            # Take profit hit
            target_hit = price >= self.target_price if self.target_price is not None else False

            if rsi_exit:
                print(f"🌙 RSI EXIT! RSI crossed below {self.rsi_exit_level} ({rsi_prev:.2f} -> {rsi_now:.2f})")
                print(f"   💰 Exit at: {price:.2f} | Entry: {self.entry_price:.2f}")
                self.position.close()
                self.entry_price = None
                self.stop_price = None
                self.target_price = None
            elif stop_hit:
                print(f"🛑 STOP LOSS HIT! Price {price:.2f} <= Stop {self.stop_price:.2f}")
                self.position.close()
                self.entry_price = None
                self.stop_price = None
                self.target_price = None
            elif target_hit:
                print(f"🎯 TAKE PROFIT HIT! Price {price:.2f} >= Target {self.target_price:.2f}")
                self.position.close()
                self.entry_price = None
                self.stop_price = None
                self.target_price = None


# Run backtest
print("🌙 Starting Moon Dev backtest... 🚀")
bt = Backtest(
    data,
    VolumeGoldenCross,
    cash=1_000_000,
    commission=0.002,
    exclusive_orders=True
)

stats = bt.run()
print(stats)
print(stats._strategy)