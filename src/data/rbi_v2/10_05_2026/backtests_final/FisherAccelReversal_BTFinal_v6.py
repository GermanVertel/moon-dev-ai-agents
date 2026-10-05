import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# ── Moon Dev Data Loader ──────────────────────────────────────────────
print("🌙 Moon Dev: Loading BTC-USD 15m data...")
data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper mapping
data = data.rename(columns={
    'datetime': 'Datetime',
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})
data['Datetime'] = pd.to_datetime(data['Datetime'])
data = data.set_index('Datetime')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

# Ensure all numeric columns are float64 (talib requires double)
data['Open'] = data['Open'].astype(np.float64)
data['High'] = data['High'].astype(np.float64)
data['Low'] = data['Low'].astype(np.float64)
data['Close'] = data['Close'].astype(np.float64)
data['Volume'] = data['Volume'].astype(np.float64)

print(f"✨ Moon Dev: Data loaded — {len(data)} bars 🚀")


# ── Fisher Transform Helper ───────────────────────────────────────────
def fisher_transform(high, low, period=10):
    """Compute Fisher Transform using talib primitives."""
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)

    hl2 = (high + low) / 2.0

    # Rolling max/min over period
    max_high = talib.MAX(hl2, timeperiod=period)
    min_low = talib.MIN(hl2, timeperiod=period)

    # Normalize to [-1, 1]
    rng = max_high - min_low
    rng = np.where(rng == 0, 1e-10, rng)
    val = 2.0 * ((hl2 - min_low) / rng - 0.5)

    # Smooth and clip
    val = pd.Series(val).ewm(span=5, adjust=False).mean().values
    val = np.clip(val, -0.999, 0.999)

    # Fisher transform
    fisher = 0.5 * np.log((1.0 + val) / (1.0 - val))

    # Recursive smoothing (standard Fisher)
    fisher_smooth = np.zeros_like(fisher)
    for i in range(1, len(fisher)):
        fisher_smooth[i] = 0.33 * fisher[i] + 0.67 * fisher_smooth[i - 1]

    return fisher_smooth


# ── Volume Acceleration Helper ────────────────────────────────────────
def volume_acceleration(volume, period=1):
    """2nd derivative of volume via chained ROC."""
    vol = np.asarray(volume, dtype=np.float64)
    roc1 = talib.ROC(vol, timeperiod=period)
    roc1 = np.nan_to_num(roc1, nan=0.0)
    accel = talib.ROC(roc1, timeperiod=period)
    accel = np.nan_to_num(accel, nan=0.0)
    return accel


# ── Strategy ──────────────────────────────────────────────────────────
class FisherAccelReversal(Strategy):
    fisher_period = 10
    adx_period = 14
    atr_period = 14
    adx_threshold = 25
    fisher_extreme = 2.0
    risk_pct = 0.02
    time_stop_bars = 20

    def init(self):
        print("🌙 Moon Dev: Initializing FisherAccelReversal indicators...")

        high = self.data.High
        low = self.data.Low
        close = self.data.Close
        volume = self.data.Volume

        # Fisher Transform
        self.fisher = self.I(
            fisher_transform,
            high, low,
            self.fisher_period,
            name="Fisher"
        )

        # ADX
        self.adx = self.I(talib.ADX, high, low, close,
                          timeperiod=self.adx_period, name="ADX")

        # ATR
        self.atr = self.I(talib.ATR, high, low, close,
                          timeperiod=self.atr_period, name="ATR")

        # Volume acceleration
        self.vol_accel = self.I(volume_acceleration, volume, 1, name="VolAccel")

        # Swing highs/lows for divergence
        self.swing_high = self.I(talib.MAX, high, timeperiod=5, name="SwingHigh")
        self.swing_low = self.I(talib.MIN, low, timeperiod=5, name="SwingLow")

        # Track entry bar for time stop
        self.entry_bar = None

        # Moon Dev debug counters
        self._sig_long = 0
        self._sig_short = 0
        self._fisher_hook_long = 0
        self._fisher_hook_short = 0
        self._adx_block = 0

        print("✨ Moon Dev: Indicators ready 🚀")

    def next(self):
        i = len(self.data) - 1
        if i < self.fisher_period + 10:
            return

        price = self.data.Close[-1]
        fisher_now = self.fisher[-1]
        fisher_prev = self.fisher[-2]
        adx_now = self.adx[-1]
        atr_now = self.atr[-1]
        vol_accel_now = self.vol_accel[-1]
        vol_accel_prev = self.vol_accel[-2]

        # Guard against NaN
        if (np.isnan(fisher_now) or np.isnan(fisher_prev) or
                np.isnan(adx_now) or np.isnan(atr_now) or
                np.isnan(vol_accel_now) or np.isnan(vol_accel_prev)):
            return

        # ── Time stop check ──────────────────────────────────────────
        if self.position and self.entry_bar is not None:
            if i - self.entry_bar >= self.time_stop_bars:
                print(f"⏰ Moon Dev: Time stop hit at bar {i} — exiting 🌙")
                self.position.close()
                self.entry_bar = None
                return

        # ── In-position exits (Fisher opposing extreme) ──────────────
        if self.position.is_long:
            if fisher_now > self.fisher_extreme and fisher_prev <= self.fisher_extreme:
                print(f"🎯 Moon Dev: LONG exit — Fisher hit +{self.fisher_extreme} at {price:.2f} 🌙")
                self.position.close()
                self.entry_bar = None
                return

        if self.position.is_short:
            if fisher_now < -self.fisher_extreme and fisher_prev >= -self.fisher_extreme:
                print(f"🎯 Moon Dev: SHORT exit — Fisher hit -{self.fisher_extreme} at {price:.2f} 🌙")
                self.position.close()
                self.entry_bar = None
                return

        # ── No position: look for entries ────────────────────────────
        if not self.position:
            # ADX filter (mean-reverting strategy → skip trending markets)
            if adx_now >= self.adx_threshold:
                self._adx_block += 1
                return

            # ── LONG: Fisher hooks up from below -2.0 ────────────────
            fisher_hook_up = (fisher_prev < -self.fisher_extreme) and (fisher_now >= -self.fisher_extreme)
            if fisher_hook_up:
                self._fisher_hook_long += 1

            recent_low = self.swing_low[-1]
            prev_low = self.swing_low[-5] if i >= 5 else recent_low
            price_lower_low = recent_low < prev_low
            vol_accel_positive = vol_accel_now > 0 and vol_accel_prev > 0

            if fisher_hook_up and price_lower_low and vol_accel_positive:
                equity = self.equity
                risk_amount = equity * self.risk_pct
                stop_distance = 1.5 * atr_now
                if stop_distance <= 0:
                    return
                # Convert risk-based units to fraction of equity for backtesting.py
                position_units = risk_amount / stop_distance
                position_size = position_units * price / equity
                position_size = max(0.001, min(position_size, 0.95))
                if position_size <= 0:
                    return

                sl_price = price - stop_distance
                self._sig_long += 1
                print(f"🚀 Moon Dev: LONG entry at {price:.2f} | Fisher={fisher_now:.2f} | ADX={adx_now:.1f} | VolAccel={vol_accel_now:.2f} | size={position_size:.4f} 🌙")
                self.buy(size=position_size, sl=sl_price)
                self.entry_bar = i

            # ── SHORT: Fisher hooks down from above +2.0 ─────────────
            fisher_hook_down = (fisher_prev > self.fisher_extreme) and (fisher_now <= self.fisher_extreme)
            if fisher_hook_down:
                self._fisher_hook_short += 1

            recent_high = self.swing_high[-1]
            prev_high = self.swing_high[-5] if i >= 5 else recent_high
            price_higher_high = recent_high > prev_high
            vol_accel_negative = vol_accel_now < 0 and vol_accel_prev < 0

            if fisher_hook_down and price_higher_high and vol_accel_negative:
                equity = self.equity
                risk_amount = equity * self.risk_pct
                stop_distance = 1.5 * atr_now
                if stop_distance <= 0:
                    return
                position_units = risk_amount / stop_distance
                position_size = position_units * price / equity
                position_size = max(0.001, min(position_size, 0.95))
                if position_size <= 0:
                    return

                sl_price = price + stop_distance
                self._sig_short += 1
                print(f"🔻 Moon Dev: SHORT entry at {price:.2f} | Fisher={fisher_now:.2f} | ADX={adx_now:.1f} | VolAccel={vol_accel_now:.2f} | size={position_size:.4f} 🌙")
                self.sell(size=position_size, sl=sl_price)
                self.entry_bar = i


# ── Run Backtest ──────────────────────────────────────────────────────
print("🌙 Moon Dev: Starting FisherAccelReversal backtest...")
bt = Backtest(data, FisherAccelReversal, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)
print("✨ Moon Dev: Backtest complete 🚀🌙")