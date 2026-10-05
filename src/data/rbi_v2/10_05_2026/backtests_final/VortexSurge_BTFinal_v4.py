import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Load data
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Rename to proper case
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

print("🌙 VortexSurge Backtest Initializing... ✨")
print(f"📊 Data loaded: {len(data)} bars")
print(f"📅 Date range: {data.index[0]} to {data.index[-1]}")
print(f"🚀 Starting backtest...\n")


class VortexSurge(Strategy):
    # Strategy parameters
    vi_period = 14
    cmo_period = 14
    cmo_threshold = 60
    kc_ema_period = 20
    kc_atr_period = 10
    kc_multiplier = 2.0
    risk_pct = 0.02
    time_stop_bars = 20

    def init(self):
        print("🌙 Initializing VortexSurge indicators... ✨")

        high = pd.Series(self.data.High, index=self.data.index)
        low = pd.Series(self.data.Low, index=self.data.index)
        close = pd.Series(self.data.Close, index=self.data.index)

        # === Vortex Indicator ===
        high_prev = high.shift(1)
        low_prev = low.shift(1)

        vm_plus = (high - low_prev).abs()
        vm_minus = (low - high_prev).abs()

        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)

        vm_plus_sum = vm_plus.rolling(self.vi_period).sum()
        vm_minus_sum = vm_minus.rolling(self.vi_period).sum()
        tr_sum = tr.rolling(self.vi_period).sum()

        vi_plus = (vm_plus_sum / tr_sum).values
        vi_minus = (vm_minus_sum / tr_sum).values

        self.vi_plus = self.I(lambda: vi_plus, name='VI+')
        self.vi_minus = self.I(lambda: vi_minus, name='VI-')

        # === Chande Momentum Oscillator ===
        delta = close.diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)

        sum_up = up.rolling(self.cmo_period).sum()
        sum_down = down.rolling(self.cmo_period).sum()

        cmo = (100 * (sum_up - sum_down) / (sum_up + sum_down)).values
        self.cmo = self.I(lambda: cmo, name='CMO')

        # === Keltner Channels ===
        kc_ema = talib.EMA(close.values, timeperiod=self.kc_ema_period)
        kc_atr = talib.ATR(high.values, low.values, close.values, timeperiod=self.kc_atr_period)

        kc_upper = kc_ema + self.kc_multiplier * kc_atr
        kc_lower = kc_ema - self.kc_multiplier * kc_atr
        kc_width = kc_upper - kc_lower

        self.kc_upper = self.I(lambda: kc_upper, name='KC_Upper')
        self.kc_lower = self.I(lambda: kc_lower, name='KC_Lower')
        self.kc_width = self.I(lambda: kc_width, name='KC_Width')
        self.kc_ema = self.I(lambda: kc_ema, name='KC_EMA')

        # === ATR for risk management ===
        self.atr = self.I(lambda: kc_atr, name='ATR')

        # === Volume SMA for filter ===
        volume = pd.Series(self.data.Volume, index=self.data.index)
        vol_sma = volume.rolling(20).mean().values
        self.vol_sma = self.I(lambda: vol_sma, name='Vol_SMA')

        # Internal state
        self.trade_direction = 0
        self.entry_bar = 0
        self.highest_since_entry = 0
        self.lowest_since_entry = 0
        self.prev_width = 0
        self.trailing_stop = 0

        print("🌙 All indicators initialized! 🚀")

    def next(self):
        if len(self.data) < 30:
            return

        if np.isnan(self.vi_plus[-1]) or np.isnan(self.cmo[-1]) or np.isnan(self.kc_width[-1]):
            return

        if self.position:
            self._update_trailing_stop()
        else:
            self._check_entry()

    def _check_entry(self):
        price = self.data.Close[-1]
        cmo_val = self.cmo[-1]
        vi_plus = self.vi_plus[-1]
        vi_minus = self.vi_minus[-1]
        vi_plus_prev = self.vi_plus[-2]
        vi_minus_prev = self.vi_minus[-2]
        width = self.kc_width[-1]
        atr_val = self.atr[-1]
        vol = self.data.Volume[-1]
        vol_avg = self.vol_sma[-1]

        if np.isnan(width) or width <= 0:
            return

        volume_ok = vol > vol_avg if not np.isnan(vol_avg) else True

        # === Long Entry ===
        bullish_cross = vi_plus_prev <= vi_minus_prev and vi_plus > vi_minus
        strong_bull_momentum = cmo_val > self.cmo_threshold

        if bullish_cross and strong_bull_momentum and volume_ok:
            stop_distance = 2.0 * atr_val
            if stop_distance <= 0 or np.isnan(stop_distance):
                return

            risk_amount = self.equity * self.risk_pct
            position_size = int(round(risk_amount / stop_distance))
            if position_size < 1:
                position_size = 1

            max_size = int(self.equity * 0.95 / price)
            if max_size < 1:
                max_size = 1
            if position_size > max_size:
                position_size = max_size

            print(f"🚀🌙 LONG ENTRY! VI+ crossed VI- | CMO={cmo_val:.2f} > {self.cmo_threshold} | Price={price:.2f} | Size={position_size}")

            self.buy(size=position_size)
            self.trade_direction = 1
            self.entry_bar = len(self.data)
            self.highest_since_entry = self.data.High[-1]
            self.lowest_since_entry = self.data.Low[-1]
            self.prev_width = width
            self.trailing_stop = price - stop_distance
            return

        # === Short Entry ===
        bearish_cross = vi_minus_prev <= vi_plus_prev and vi_minus > vi_plus
        strong_bear_momentum = cmo_val < -self.cmo_threshold

        if bearish_cross and strong_bear_momentum and volume_ok:
            stop_distance = 2.0 * atr_val
            if stop_distance <= 0 or np.isnan(stop_distance):
                return

            risk_amount = self.equity * self.risk_pct
            position_size = int(round(risk_amount / stop_distance))
            if position_size < 1:
                position_size = 1

            max_size = int(self.equity * 0.95 / price)
            if max_size < 1:
                max_size = 1
            if position_size > max_size:
                position_size = max_size

            print(f"🔻🌙 SHORT ENTRY! VI- crossed VI+ | CMO={cmo_val:.2f} < -{self.cmo_threshold} | Price={price:.2f} | Size={position_size}")

            self.sell(size=position_size)
            self.trade_direction = -1
            self.entry_bar = len(self.data)
            self.highest_since_entry = self.data.High[-1]
            self.lowest_since_entry = self.data.Low[-1]
            self.prev_width = width
            self.trailing_stop = price + stop_distance

    def _update_trailing_stop(self):
        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        width = self.kc_width[-1]
        cmo_val = self.cmo[-1]

        if np.isnan(width):
            return

        if high > self.highest_since_entry:
            self.highest_since_entry = high
        if low < self.lowest_since_entry:
            self.lowest_since_entry = low

        width_expanding = width > self.prev_width if self.prev_width > 0 else True
        width_mult = 1.5 if width_expanding else 0.8

        if self.trade_direction == 1:  # Long
            new_stop = self.highest_since_entry - width_mult * width
            if new_stop > self.trailing_stop:
                self.trailing_stop = new_stop

            if price < self.trailing_stop:
                print(f"🌙✨ TRAILING STOP EXIT (LONG) | Price={price:.2f} < Stop={self.trailing_stop:.2f}")
                self.position.close()
                self._reset_state()
                return

            if cmo_val < self.cmo_threshold:
                print(f"🌙✨ CMO EXIT (LONG) | CMO={cmo_val:.2f} < {self.cmo_threshold}")
                self.position.close()
                self._reset_state()
                return

            if len(self.data) - self.entry_bar >= self.time_stop_bars:
                if price <= self.data.Close[self.entry_bar - 1]:
                    print(f"🌙⏰ TIME STOP EXIT (LONG) | {self.time_stop_bars} bars without profit")
                    self.position.close()
                    self._reset_state()
                    return

        elif self.trade_direction == -1:  # Short
            new_stop = self.lowest_since_entry + width_mult * width
            if new_stop < self.trailing_stop:
                self.trailing_stop = new_stop

            if price > self.trailing_stop:
                print(f"🌙✨ TRAILING STOP EXIT (SHORT) | Price={price:.2f} > Stop={self.trailing_stop:.2f}")
                self.position.close()
                self._reset_state()
                return

            if cmo_val > -self.cmo_threshold:
                print(f"🌙✨ CMO EXIT (SHORT) | CMO={cmo_val:.2f} > -{self.cmo_threshold}")
                self.position.close()
                self._reset_state()
                return

            if len(self.data) - self.entry_bar >= self.time_stop_bars:
                if price >= self.data.Close[self.entry_bar - 1]:
                    print(f"🌙⏰ TIME STOP EXIT (SHORT) | {self.time_stop_bars} bars without profit")
                    self.position.close()
                    self._reset_state()
                    return

        self.prev_width = width

    def _reset_state(self):
        self.trade_direction = 0
        self.entry_bar = 0
        self.highest_since_entry = 0
        self.lowest_since_entry = 0
        self.prev_width = 0
        self.trailing_stop = 0


print("🌙🚀 Running VortexSurge Backtest...\n")
bt = Backtest(data, VortexSurge, cash=1_000_000, commission=0.001)

stats = bt.run()
print(stats)
print(stats._strategy)