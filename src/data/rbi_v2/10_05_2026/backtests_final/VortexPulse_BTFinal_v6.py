import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

print("🌙 Moon Dev Backtest AI spinning up... ✨🚀")

# Load data
data_path = '/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv'
data = pd.read_csv(data_path)

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Set datetime index BEFORE renaming
if 'datetime' in data.columns:
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')

# Map to proper case
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

# Ensure required columns
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
data = data.dropna()
print(f"🌙 Data loaded: {len(data)} bars ✨")
print(f"🌙 Columns: {list(data.columns)} ✨")


class VortexPulse(Strategy):
    vi_period = 14
    cmo_period = 14
    atr_period = 14
    ema_period = 200
    cmo_entry_long = 50
    cmo_entry_short = -50
    cmo_exit_long = 30
    cmo_exit_short = -30
    risk_pct = 0.02
    atr_stop_mult = 2.0
    trail_atr_mult = 1.5
    trail_trigger_atr = 1.0
    time_stop_bars = 20

    def init(self):
        h = pd.Series(self.data.High, index=np.arange(len(self.data)))
        l = pd.Series(self.data.Low, index=np.arange(len(self.data)))
        c = pd.Series(self.data.Close, index=np.arange(len(self.data)))

        # True Range
        prev_close = c.shift(1)
        tr = pd.concat([(h - l), (h - prev_close).abs(), (l - prev_close).abs()], axis=1).max(axis=1)
        tr_vals = tr.values
        self.tr = self.I(lambda: tr_vals, name='TR')

        # Vortex
        vm_plus = (h - l.shift(1)).abs()
        vm_minus = (l - h.shift(1)).abs()

        tr_sum = tr.rolling(self.vi_period).sum()
        tr_sum_safe = tr_sum.replace(0, np.nan)
        vi_plus = (vm_plus.rolling(self.vi_period).sum() / tr_sum_safe).values
        vi_minus = (vm_minus.rolling(self.vi_period).sum() / tr_sum_safe).values

        self.vi_plus = self.I(lambda: vi_plus, name='VI+')
        self.vi_minus = self.I(lambda: vi_minus, name='VI-')

        # CMO
        diff = c.diff()
        up = diff.clip(lower=0)
        down = (-diff).clip(lower=0)
        sum_up = up.rolling(self.cmo_period).sum()
        sum_down = down.rolling(self.cmo_period).sum()
        denom = (sum_up + sum_down).replace(0, np.nan)
        cmo = (100 * (sum_up - sum_down) / denom).values
        self.cmo = self.I(lambda: cmo, name='CMO')

        # ATR
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period, name='ATR')

        # EMA200
        self.ema200 = self.I(talib.EMA, self.data.Close, timeperiod=self.ema_period, name='EMA200')

        self.entry_bar = None
        self.stop_price = None
        self.take_profit_price = None
        self.trail_active = False
        self.highest_since_entry = None
        self.lowest_since_entry = None

        print("🌙 VortexPulse indicators initialized ✨🚀")

    def next(self):
        if len(self.data) < max(self.vi_period, self.cmo_period, self.ema_period, self.atr_period) + 2:
            return

        price = self.data.Close[-1]
        vi_plus = self.vi_plus[-1]
        vi_minus = self.vi_minus[-1]
        vi_plus_prev = self.vi_plus[-2]
        vi_minus_prev = self.vi_minus[-2]
        cmo = self.cmo[-1]
        cmo_prev = self.cmo[-2]
        atr = self.atr[-1]
        ema200 = self.ema200[-1]

        if np.isnan(vi_plus) or np.isnan(vi_minus) or np.isnan(cmo) or np.isnan(atr) or np.isnan(ema200):
            return
        if np.isnan(vi_plus_prev) or np.isnan(vi_minus_prev) or np.isnan(cmo_prev):
            return

        # Manage existing position
        if self.position:
            if self.entry_bar is None:
                self.entry_bar = len(self.data)

            # Time stop
            if len(self.data) - self.entry_bar >= self.time_stop_bars:
                print(f"🌙⏰ Time stop hit at {price:.2f} - signal decay exit ✨")
                self.position.close()
                self._reset()
                return

            if self.position.is_long:
                self.highest_since_entry = max(self.highest_since_entry or price, self.data.High[-1])

                # Trailing stop
                if not self.trail_active and self.highest_since_entry - self.trades[-1].entry_price >= self.trail_trigger_atr * atr:
                    self.trail_active = True
                    self.stop_price = self.highest_since_entry - self.trail_atr_mult * atr
                    print(f"🌙📈 Trailing stop activated at {self.stop_price:.2f} ✨")

                if self.trail_active:
                    new_stop = self.highest_since_entry - self.trail_atr_mult * atr
                    if new_stop > self.stop_price:
                        self.stop_price = new_stop

                # Exit conditions
                vi_reversal = vi_minus > vi_plus and vi_minus_prev <= vi_plus_prev
                cmo_exhaust = cmo < self.cmo_exit_long

                if vi_reversal or cmo_exhaust:
                    reason = "VI reversal" if vi_reversal else "CMO exhaustion"
                    print(f"🌙🔻 Long exit ({reason}) at {price:.2f} 🚀")
                    self.position.close()
                    self._reset()
                    return

                # Stop loss check
                if self.stop_price is not None and self.data.Low[-1] <= self.stop_price:
                    print(f"🌙🛑 Long stop-loss hit at {self.stop_price:.2f} 💫")
                    self.position.close()
                    self._reset()
                    return

            elif self.position.is_short:
                self.lowest_since_entry = min(self.lowest_since_entry or price, self.data.Low[-1])

                if not self.trail_active and self.trades[-1].entry_price - self.lowest_since_entry >= self.trail_trigger_atr * atr:
                    self.trail_active = True
                    self.stop_price = self.lowest_since_entry + self.trail_atr_mult * atr
                    print(f"🌙📉 Trailing stop activated at {self.stop_price:.2f} ✨")

                if self.trail_active:
                    new_stop = self.lowest_since_entry + self.trail_atr_mult * atr
                    if new_stop < self.stop_price:
                        self.stop_price = new_stop

                vi_reversal = vi_plus > vi_minus and vi_plus_prev <= vi_minus_prev
                cmo_exhaust = cmo > self.cmo_exit_short

                if vi_reversal or cmo_exhaust:
                    reason = "VI reversal" if vi_reversal else "CMO exhaustion"
                    print(f"🌙🔺 Short exit ({reason}) at {price:.2f} 🚀")
                    self.position.close()
                    self._reset()
                    return

                if self.stop_price is not None and self.data.High[-1] >= self.stop_price:
                    print(f"🌙🛑 Short stop-loss hit at {self.stop_price:.2f} 💫")
                    self.position.close()
                    self._reset()
                    return
            return

        # Entry logic
        long_cross = vi_plus > vi_minus and vi_plus_prev <= vi_minus_prev
        short_cross = vi_minus > vi_plus and vi_minus_prev <= vi_plus_prev

        if long_cross and cmo > self.cmo_entry_long and cmo > cmo_prev and price > ema200:
            risk_per_unit = self.atr_stop_mult * atr
            if risk_per_unit <= 0:
                return
            size_frac = self.risk_pct * price / risk_per_unit
            if size_frac <= 0:
                return
            if size_frac > 0.95:
                size_frac = 0.95
            self.stop_price = price - risk_per_unit
            self.buy(size=size_frac)
            self.entry_bar = len(self.data)
            self.trail_active = False
            self.highest_since_entry = price
            print(f"🌙🚀 LONG entry at {price:.2f} | VI+={vi_plus:.3f} VI-={vi_minus:.3f} CMO={cmo:.1f} size={size_frac:.4f} ✨")

        elif short_cross and cmo < self.cmo_entry_short and cmo < cmo_prev and price < ema200:
            risk_per_unit = self.atr_stop_mult * atr
            if risk_per_unit <= 0:
                return
            size_frac = self.risk_pct * price / risk_per_unit
            if size_frac <= 0:
                return
            if size_frac > 0.95:
                size_frac = 0.95
            self.stop_price = price + risk_per_unit
            self.sell(size=size_frac)
            self.entry_bar = len(self.data)
            self.trail_active = False
            self.lowest_since_entry = price
            print(f"🌙🔻 SHORT entry at {price:.2f} | VI+={vi_plus:.3f} VI-={vi_minus:.3f} CMO={cmo:.1f} size={size_frac:.4f} ✨")

    def _reset(self):
        self.entry_bar = None
        self.stop_price = None
        self.trail_active = False
        self.highest_since_entry = None
        self.lowest_since_entry = None


bt = Backtest(data, VortexPulse, cash=1_000_000, commission=0.001, exclusive_orders=True)
stats = bt.run()
print(stats)
print(stats._strategy)