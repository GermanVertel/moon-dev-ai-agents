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

# 🗺️ Map to backtesting.py required columns
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume',
})

# 📅 Parse datetime
if 'datetime' in data.columns:
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')

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
    vol_ma_period = 20              # Volume MA period for spike detection
    vol_spike_mult = 2.0            # Volume spike threshold (2x avg)
    atr_period = 14                 # ATR period
    atr_spike_mult = 1.5            # Spike range must be >= 1.5x ATR
    fib_entry = 0.618               # Fibonacci entry level
    fib_tp1 = 0.382                 # TP1 Fibonacci level
    risk_pct = 0.015                # 1.5% risk per trade
    time_stop_bars = 20             # Time stop (bars)
    trail_fib = 0.50                # Trail stop at 50% Fib once TP1 hit
    spike_lookback = 6              # Bars to look back for spike detection

    def init(self):
        print("🌙 Initializing Moon Dev indicators...")
        # 🔧 Convert to float64 arrays explicitly for talib
        vol = np.asarray(self.data.Volume, dtype=np.float64)
        high = np.asarray(self.data.High, dtype=np.float64)
        low = np.asarray(self.data.Low, dtype=np.float64)
        close = np.asarray(self.data.Close, dtype=np.float64)

        self.vol_ma = self.I(talib.SMA, vol, timeperiod=self.vol_ma_period)
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        # State tracking
        self.trade_state = None
        self.entry_bar = None
        self.spike_low = None
        self.spike_high = None
        self.spike_range = None
        self.tp1_hit = False
        self.entry_price = None
        self.stop_price = None
        print("✨ Indicators ready! Let's hunt liquidations 🚀")

    def _detect_spike(self, i):
        """
        Detect a liquidation spike: a large impulse move with volume spike.
        Returns (direction, spike_high, spike_low) or None.
        """
        if i < self.spike_lookback + 1:
            return None

        # Volume spike check
        if self.vol_ma[i] == 0 or np.isnan(self.vol_ma[i]):
            return None
        vol_ratio = self.data.Volume[i] / self.vol_ma[i]
        if vol_ratio < self.vol_spike_mult:
            return None

        # Look at recent bars for the impulse
        window = slice(i - self.spike_lookback + 1, i + 1)
        highs = self.data.High[window]
        lows = self.data.Low[window]

        spike_high = np.max(highs)
        spike_low = np.min(lows)
        spike_range = spike_high - spike_low

        # Must be >= 1.5x ATR
        if np.isnan(self.atr[i]) or spike_range < self.atr_spike_mult * self.atr[i]:
            return None

        # Determine direction: which way was the impulse?
        open_p = self.data.Open[i - self.spike_lookback + 1]
        close_p = self.data.Close[i]

        # Downward spike (longs liquidated) if price dropped a lot
        if close_p < open_p and (open_p - close_p) > 0.5 * spike_range:
            return ('down', spike_high, spike_low)
        # Upward spike (shorts liquidated)
        elif close_p > open_p and (close_p - open_p) > 0.5 * spike_range:
            return ('up', spike_high, spike_low)

        return None

    def _fib_level(self, level):
        """Compute a fib price level on the spike range (0% = spike end, 100% = spike start)."""
        return self.spike_low + level * self.spike_range

    def _fib_level_short(self, level):
        """For short reversal (upward spike): 0 = high, 100 = low."""
        return self.spike_high - level * self.spike_range

    def next(self):
        i = len(self.data) - 1

        # ⏰ Time stop check
        if self.position and self.entry_bar is not None:
            bars_held = i - self.entry_bar
            if bars_held >= self.time_stop_bars and not self.tp1_hit:
                print(f"⏰ Time stop hit at bar {i} — cascade too slow, exiting 🌙")
                self.position.close()
                self._reset_state()
                return

        # 🎯 Manage open position
        if self.position:
            self._manage_position(i)
            return

        # 🔍 Look for new spike
        spike = self._detect_spike(i)
        if spike is None:
            return

        direction, spike_high, spike_low = spike
        spike_range = spike_high - spike_low

        # 🎯 LONG reversal after downward cascade
        if direction == 'down':
            fib_618 = spike_low + self.fib_entry * spike_range
            # Price must close above 61.8% retracement
            if self.data.Close[i] > fib_618 and self.data.Close[i - 1] <= fib_618:
                self.spike_low = spike_low
                self.spike_high = spike_high
                self.spike_range = spike_range

                entry = self.data.Close[i]
                stop = spike_low * 0.999  # just below spike low
                risk_per_unit = entry - stop
                if risk_per_unit <= 0:
                    return

                risk_amount = self.equity * self.risk_pct
                size = int(round(risk_amount / risk_per_unit))
                if size < 1:
                    size = 1

                print(f"🌙✨ LONG LIQUIDATION REVERSAL! Bar {i}")
                print(f"   Spike Low: {spike_low:.2f} | Spike High: {spike_high:.2f}")
                print(f"   Entry: {entry:.2f} | Stop: {stop:.2f} | Size: {size}")
                print(f"   Fib 61.8%: {fib_618:.2f} 🚀")

                self.buy(size=size)
                self.entry_bar = i
                self.entry_price = entry
                self.stop_price = stop
                self.tp1_hit = False

        # 🎯 SHORT reversal after upward cascade
        elif direction == 'up':
            fib_618 = spike_high - self.fib_entry * spike_range
            if self.data.Close[i] < fib_618 and self.data.Close[i - 1] >= fib_618:
                self.spike_low = spike_low
                self.spike_high = spike_high
                self.spike_range = spike_range

                entry = self.data.Close[i]
                stop = spike_high * 1.001  # just above spike high
                risk_per_unit = stop - entry
                if risk_per_unit <= 0:
                    return

                risk_amount = self.equity * self.risk_pct
                size = int(round(risk_amount / risk_per_unit))
                if size < 1:
                    size = 1

                print(f"🌙✨ SHORT LIQUIDATION REVERSAL! Bar {i}")
                print(f"   Spike High: {spike_high:.2f} | Spike Low: {spike_low:.2f}")
                print(f"   Entry: {entry:.2f} | Stop: {stop:.2f} | Size: {size}")
                print(f"   Fib 61.8%: {fib_618:.2f} 🚀")

                self.sell(size=size)
                self.entry_bar = i
                self.entry_price = entry
                self.stop_price = stop
                self.tp1_hit = False

    def _manage_position(self, i):
        """Handle stops, TP1, TP2, and trailing."""
        price_high = self.data.High[i]
        price_low = self.data.Low[i]
        price_close = self.data.Close[i]

        if self.position.is_long:
            # 🛑 Stop loss check
            if price_low <= self.stop_price:
                print(f"🛑 LONG STOP HIT at {self.stop_price:.2f} — cascade wasn't done 🌙")
                self.position.close()
                self._reset_state()
                return

            # 🎯 TP1 = 38.2% fib
            tp1_price = self.spike_low + self.fib_tp1 * self.spike_range
            tp2_price = self.spike_high  # 100% retracement to origin

            if not self.tp1_hit and price_high >= tp1_price:
                print(f"🎯 TP1 HIT (Long) at {tp1_price:.2f} — taking 50% profit ✨")
                self.position.close()
                self.tp1_hit = True
                # Trail stop at 50% fib
                self.stop_price = self.spike_low + self.trail_fib * self.spike_range
                print(f"   Trailing stop moved to 50% Fib: {self.stop_price:.2f} 🌙")
                return

            if self.tp1_hit and price_high >= tp2_price:
                print(f"🚀 TP2 HIT (Long) at {tp2_price:.2f} — full origin retracement! 🌙✨")
                self.position.close()
                self._reset_state()
                return

        elif self.position.is_short:
            if price_high >= self.stop_price:
                print(f"🛑 SHORT STOP HIT at {self.stop_price:.2f} — cascade wasn't done 🌙")
                self.position.close()
                self._reset_state()
                return

            tp1_price = self.spike_high - self.fib_tp1 * self.spike_range
            tp2_price = self.spike_low

            if not self.tp1_hit and price_low <= tp1_price:
                print(f"🎯 TP1 HIT (Short) at {tp1_price:.2f} — taking 50% profit ✨")
                self.position.close()
                self.tp1_hit = True
                self.stop_price = self.spike_high - self.trail_fib * self.spike_range
                print(f"   Trailing stop moved to 50% Fib: {self.stop_price:.2f} 🌙")
                return

            if self.tp1_hit and price_low <= tp2_price:
                print(f"🚀 TP2 HIT (Short) at {tp2_price:.2f} — full origin retracement! 🌙✨")
                self.position.close()
                self._reset_state()
                return

    def _reset_state(self):
        self.trade_state = None
        self.entry_bar = None
        self.spike_low = None
        self.spike_high = None
        self.spike_range = None
        self.tp1_hit = False
        self.entry_price = None
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