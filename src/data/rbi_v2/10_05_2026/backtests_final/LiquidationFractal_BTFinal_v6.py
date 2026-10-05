import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# 🌙 Moon Dev's LiquidationFractal Backtest 🌙
print("🚀 Initializing Moon Dev's LiquidationFractal Strategy...")
print("✨ Loading cosmic data from the moon base...")

data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# 🧹 Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# 📅 Parse datetime BEFORE renaming
if 'datetime' in data.columns:
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')

# 🗺️ Map to backtesting.py required columns
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume',
})

data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

# 🔧 Ensure all OHLCV columns are float64 (talib requires double)
data['Open'] = data['Open'].astype(np.float64)
data['High'] = data['High'].astype(np.float64)
data['Low'] = data['Low'].astype(np.float64)
data['Close'] = data['Close'].astype(np.float64)
data['Volume'] = data['Volume'].astype(np.float64)

print(f"🌙 Data loaded: {len(data)} lunar cycles of price action ✨")


class LiquidationFractal(Strategy):
    """
    🌙 LiquidationFractal Strategy 🌙
    Catches forced-selling/buying cascades and rides the mean-reversion
    back to the pre-liquidation origin using Fibonacci retracement triggers.
    """

    # ⚙️ Strategy parameters
    vol_ma_period = 20
    vol_spike_mult = 1.5
    atr_period = 14
    atr_spike_mult = 1.0
    fib_entry = 0.618
    fib_tp1 = 0.382
    risk_pct = 0.015
    time_stop_bars = 20
    trail_fib = 0.50
    spike_lookback = 6

    def init(self):
        print("🌙 Initializing Moon Dev indicators...")
        vol = np.asarray(self.data.Volume, dtype=np.float64)
        high = np.asarray(self.data.High, dtype=np.float64)
        low = np.asarray(self.data.Low, dtype=np.float64)
        close = np.asarray(self.data.Close, dtype=np.float64)

        self.vol_ma = self.I(talib.SMA, vol, timeperiod=self.vol_ma_period)
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        # State tracking
        self.entry_bar = None
        self.spike_low = None
        self.spike_high = None
        self.spike_range = None
        self.tp1_hit = False
        self.stop_price = None
        print("✨ Indicators ready! Let's hunt liquidations 🚀")

    def _detect_spike(self, i):
        if i < self.spike_lookback + 1:
            return None

        # Use indicator array values (backtesting.py self.I wraps them)
        vol_ma_val = self.vol_ma[-1]
        atr_val = self.atr[-1]

        if vol_ma_val == 0 or np.isnan(vol_ma_val):
            return None
        vol_ratio = self.data.Volume[-1] / vol_ma_val
        if vol_ratio < self.vol_spike_mult:
            return None

        # Use last N bars via arrays
        highs = np.asarray(self.data.High)[-self.spike_lookback:]
        lows = np.asarray(self.data.Low)[-self.spike_lookback:]

        spike_high = np.max(highs)
        spike_low = np.min(lows)
        spike_range = spike_high - spike_low

        if np.isnan(atr_val) or spike_range < self.atr_spike_mult * atr_val:
            return None

        open_p = self.data.Open[-self.spike_lookback]
        close_p = self.data.Close[-1]

        if close_p < open_p and (open_p - close_p) > 0.3 * spike_range:
            return ('down', spike_high, spike_low)
        elif close_p > open_p and (close_p - open_p) > 0.3 * spike_range:
            return ('up', spike_high, spike_low)

        return None

    def next(self):
        # ⏰ Time stop check
        if self.position and self.entry_bar is not None:
            bars_held = len(self.data) - 1 - self.entry_bar
            if bars_held >= self.time_stop_bars and not self.tp1_hit:
                print(f"⏰ Time stop hit at bar {len(self.data)-1} — cascade too slow, exiting 🌙")
                self.position.close()
                self._reset_state()
                return

        # 🎯 Manage open position
        if self.position:
            self._manage_position()
            return

        # 🔍 Look for new spike
        spike = self._detect_spike(len(self.data) - 1)
        if spike is None:
            return

        direction, spike_high, spike_low = spike
        spike_range = spike_high - spike_low

        prev_close = self.data.Close[-2]
        cur_close = self.data.Close[-1]

        # 🎯 LONG reversal after downward cascade
        if direction == 'down':
            fib_618 = spike_low + self.fib_entry * spike_range
            if cur_close > fib_618 and prev_close <= fib_618:
                self.spike_low = spike_low
                self.spike_high = spike_high
                self.spike_range = spike_range

                entry = cur_close
                stop = spike_low * 0.999
                risk_per_unit = entry - stop
                if risk_per_unit <= 0:
                    return

                risk_amount = self.equity * self.risk_pct
                size_frac = (risk_amount / risk_per_unit) / self.equity
                if size_frac <= 0:
                    return
                if size_frac > 0.95:
                    size_frac = 0.95

                print(f"🌙✨ LONG LIQUIDATION REVERSAL! Bar {len(self.data)-1}")
                print(f"   Spike Low: {spike_low:.2f} | Spike High: {spike_high:.2f}")
                print(f"   Entry: {entry:.2f} | Stop: {stop:.2f} | Size: {size_frac:.4f}")
                print(f"   Fib 61.8%: {fib_618:.2f} 🚀")

                self.buy(size=size_frac, sl=stop)
                self.entry_bar = len(self.data) - 1
                self.stop_price = stop
                self.tp1_hit = False

        # 🎯 SHORT reversal after upward cascade
        elif direction == 'up':
            fib_618 = spike_high - self.fib_entry * spike_range
            if cur_close < fib_618 and prev_close >= fib_618:
                self.spike_low = spike_low
                self.spike_high = spike_high
                self.spike_range = spike_range

                entry = cur_close
                stop = spike_high * 1.001
                risk_per_unit = stop - entry
                if risk_per_unit <= 0:
                    return

                risk_amount = self.equity * self.risk_pct
                size_frac = (risk_amount / risk_per_unit) / self.equity
                if size_frac <= 0:
                    return
                if size_frac > 0.95:
                    size_frac = 0.95

                print(f"🌙✨ SHORT LIQUIDATION REVERSAL! Bar {len(self.data)-1}")
                print(f"   Spike High: {spike_high:.2f} | Spike Low: {spike_low:.2f}")
                print(f"   Entry: {entry:.2f} | Stop: {stop:.2f} | Size: {size_frac:.4f}")
                print(f"   Fib 61.8%: {fib_618:.2f} 🚀")

                self.sell(size=size_frac, sl=stop)
                self.entry_bar = len(self.data) - 1
                self.stop_price = stop
                self.tp1_hit = False

    def _manage_position(self):
        """Handle stops, TP1, TP2, and trailing."""
        price_high = self.data.High[-1]
        price_low = self.data.Low[-1]

        if self.position.is_long:
            if price_low <= self.stop_price:
                print(f"🛑 LONG STOP HIT at {self.stop_price:.2f} 🌙")
                self.position.close()
                self._reset_state()
                return

            tp1_price = self.spike_low + self.fib_tp1 * self.spike_range
            tp2_price = self.spike_high

            if not self.tp1_hit and price_high >= tp1_price:
                print(f"🎯 TP1 HIT (Long) at {tp1_price:.2f} ✨")
                self.position.close()
                self.tp1_hit = True
                self.stop_price = self.spike_low + self.trail_fib * self.spike_range
                print(f"   Trailing stop moved to 50% Fib: {self.stop_price:.2f} 🌙")
                return

            if self.tp1_hit and price_high >= tp2_price:
                print(f"🚀 TP2 HIT (Long) at {tp2_price:.2f} 🌙✨")
                self.position.close()
                self._reset_state()
                return

        elif self.position.is_short:
            if price_high >= self.stop_price:
                print(f"🛑 SHORT STOP HIT at {self.stop_price:.2f} 🌙")
                self.position.close()
                self._reset_state()
                return

            tp1_price = self.spike_high - self.fib_tp1 * self.spike_range
            tp2_price = self.spike_low

            if not self.tp1_hit and price_low <= tp1_price:
                print(f"🎯 TP1 HIT (Short) at {tp1_price:.2f} ✨")
                self.position.close()
                self.tp1_hit = True
                self.stop_price = self.spike_high - self.trail_fib * self.spike_range
                print(f"   Trailing stop moved to 50% Fib: {self.stop_price:.2f} 🌙")
                return

            if self.tp1_hit and price_low <= tp2_price:
                print(f"🚀 TP2 HIT (Short) at {tp2_price:.2f} 🌙✨")
                self.position.close()
                self._reset_state()
                return

    def _reset_state(self):
        self.entry_bar = None
        self.spike_low = None
        self.spike_high = None
        self.spike_range = None
        self.tp1_hit = False
        self.stop_price = None


print("🚀 Launching Moon Dev backtest engine...")
bt = Backtest(
    data,
    LiquidationFractal,
    cash=1_000_000,
    commission=0.001,
    exclusive_orders=True,
)

stats = bt.run()
print("\n" + "=" * 60)
print("🌙 MOON DEV LIQUIDATIONFRACTAL — FINAL STATS 🌙")
print("=" * 60)
print(stats)
print("\n" + "=" * 60)
print("🔍 STRATEGY DETAILS 🔍")
print("=" * 60)
print(stats._strategy)
print("✨ Moon Dev out — happy hunting! 🚀🌙")