```python
import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

print("🌙✨ Moon Dev FibBand Reversal Backtest Initializing... 🚀")

# ============================================================
# DATA LOADING & CLEANING
# ============================================================
data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
print(f"📂 Loading data from: {data_path}")

data = pd.read_csv(data_path)

# Clean column names
data.columns = data.columns.str.strip().str.lower()
print(f"🧹 Cleaned columns: {list(data.columns)}")

# Drop unnamed columns
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

data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
print(f"✅ Data ready! Shape: {data.shape}, Range: {data.index[0]} → {data.index[-1]}")


# ============================================================
# STRATEGY CLASS
# ============================================================
class FibBandReversal(Strategy):
    """
    🌙 FibBand Reversal Strategy
    Combines Fibonacci retracement levels with Bollinger Bands
    to identify high-probability reversal zones.
    """

    # --- Strategy Parameters ---
    bb_period = 20
    bb_std = 2.0
    swing_lookback = 60          # lookback for swing high/low (50-100)
    fib_tolerance = 0.005        # ±0.5% proximity to fib level
    atr_period = 14
    atr_buffer = 0.4             # ATR buffer for stop loss
    risk_pct = 0.01              # 1% risk per trade
    rr_min = 2.0                 # minimum 1:2 R:R
    time_stop_bars = 10          # exit if no +1R within 10 bars
    vix_ma_short = 10
    vix_ma_long = 20
    vix_high = 35
    vix_extreme = 40
    vix_low = 12
    min_bb_width = 0.005         # skip trades in extreme squeeze

    def init(self):
        print("🌙 Initializing FibBand Reversal indicators...")

        close = self.data.Close
        high = self.data.High
        low = self.data.Low

        # --- Bollinger Bands ---
        self.bb_mid = self.I(talib.SMA, close, timeperiod=self.bb_period, name="BB_Mid")
        self.bb_stddev = self.I(talib.STDDEV, close, timeperiod=self.bb_period, nbdev=1, name="BB_Std")
        self.bb_upper = self.I(lambda: self.bb_mid + self.bb_std * self.bb_stddev, name="BB_Upper")
        self.bb_lower = self.I(lambda: self.bb_mid - self.bb_std * self.bb_stddev, name="BB_Lower")

        # --- ATR ---
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period, name="ATR")

        # --- Swing High / Low (for Fib levels) ---
        self.swing_high = self.I(talib.MAX, high, timeperiod=self.swing_lookback, name="SwingHigh")
        self.swing_low = self.I(talib.MIN, low, timeperiod=self.swing_lookback, name="SwingLow")

        # --- VIX proxy: use rolling volatility of returns (since we only have BTC data) ---
        # We approximate VIX regime using realized volatility
        log_ret = np.log(close / np.roll(close, 1))
        log_ret[0] = 0.0
        self.vix_proxy = self.I(
            lambda: pd.Series(log_ret).rolling(14).std().values * np.sqrt(252 * 96) * 100,
            name="VIX_Proxy"
        )
        self.vix_ma10 = self.I(talib.SMA, self.vix_proxy, timeperiod=self.vix_ma_short, name="VIX_MA10")
        self.vix_ma20 = self.I(talib.SMA, self.vix_proxy, timeperiod=self.vix_ma_long, name="VIX_MA20")
        self.vix_min10 = self.I(talib.MIN, self.vix_proxy, timeperiod=self.vix_ma_short, name="VIX_Min10")
        self.vix_max10 = self.I(talib.MAX, self.vix_proxy, timeperiod=self.vix_ma_short, name="VIX_Max10")

        # --- Volume MA ---
        self.vol_ma = self.I(talib.SMA, self.data.Volume, timeperiod=20, name="Vol_MA")

        # --- Trade state tracking ---
        self.trade_meta = {}  # track entry bar, stop, target, R

        print("✨ Indicators ready! Moon Dev is watching the charts 🌙")

    # ------------------------------------------------------------
    # Helper: compute Fibonacci levels from swing high/low
    # ------------------------------------------------------------
    def _fib_levels(self, i):
        sh = self.swing_high[i]
        sl = self.swing_low[i]
        if np.isnan(sh) or np.isnan(sl) or sh <= sl:
            return None
        diff = sh - sl
        # Retracement levels measured from swing low upward (for uptrend)
        # and from swing high downward (for downtrend). We return both sets.
        levels_up = {  # moves down from high (retracement of up move)
            '0.236': sh - 0.236 * diff,
            '0.382': sh - 0.382 * diff,
            '0.500': sh - 0.500 * diff,
            '0.618': sh - 0.618 * diff,
            '0.786': sh - 0.786 * diff,
        }
        return levels_up, sh, sl

    def _near_fib(self, price, levels, tolerance):
        """Check if price is within tolerance of any key fib level."""
        for key, lvl in levels.items():
            if abs(price - lvl) / price <= tolerance:
                return key, lvl
        return None, None

    # ------------------------------------------------------------
    # NEXT BAR LOGIC
    # ------------------------------------------------------------
    def next(self):
        i = len(self.data) - 1
        if i < self.swing_lookback + 5:
            return

        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]

        # Skip if indicators not ready
        if (np.isnan(self.bb_upper[-1]) or np.isnan(self.bb_lower[-1]) or
                np.isnan(self.atr[-1]) or np.isnan(self.vix_proxy[-1])):
            return

        # --- Manage open trades first ---
        if self.position:
            self._manage_position(i)
            return

        # --- Bollinger Band width filter (skip squeeze) ---
        bb_width = (self.bb_upper[-1] - self.bb_lower[-1]) / self.bb_mid[-1]
        if bb_width < self.min_bb_width:
            return

        fib_data = self._fib_levels(i)
        if fib_data is None:
            return
        fib_levels, sh, sl = fib_data

        vix = self.vix_proxy[-1]
        vix_ma20 = self.vix_ma20[-1]

        if np.isnan(vix_ma20):
            return

        prev_close = self.data.Close[-2]
        prev_low = self.data.Low[-2]
        prev_high = self.data.High[-2]

        # ========================================================
        # LONG ENTRY (Bullish Reversal)
        # ========================================================
        # Condition: previous candle closed BELOW lower BB
        prev_below_lower = prev_close < self.bb_lower[-2]
        # Confirmation: current candle closes back INSIDE lower BB
        now_inside_lower = price > self.bb_lower[-1]
        # Fib proximity (use prev low or prev close)
        fib_key, fib_lvl = self._near_fib(prev_low, fib_levels, self.fib_tolerance)
        if fib_key is None:
            fib_key, fib_lvl = self._near_fib(prev_close, fib_levels, self.fib_tolerance)
        # Prefer golden zone 0.618 / 0.786
        golden = fib_key in ('0.618', '0.786') if fib_key else False
        # VIX elevated above 20MA but not violently spiking
        vix_ok = vix > vix_ma20 and vix < self.vix_extreme

        if prev_below_lower and now_inside_lower and fib_key is not None and vix_ok:
            # Stop below fib level + ATR buffer
            stop = fib_lvl - self.atr_buffer * self.atr[-1]
            risk = price - stop
            if risk <= 0:
                return
            target = price + self.rr_min * risk
            # Ensure R:R >= 2 (target must be at least 2R)
            if (target - price) / risk < self.rr_min:
                return

            size = self._calc_size(risk, vix)
            if size <= 0:
                return

            print(f"🌙🚀 LONG SIGNAL! Fib {fib_key}{' (GOLDEN)' if golden else ''} @ {fib_lvl:.2f} | "
                  f"Price: {price:.2f} | Stop: {stop:.2f} | Target: {target:.2f} | "
                  f"VIX: {vix:.2f} | Size: {size}")

            self.buy(size=size)
            self.trade_meta['entry_bar'] = i
            self.trade_meta['stop'] = stop
            self.trade_meta['target'] = target
            self.trade_meta['entry'] = price
            self.trade_meta['risk'] = risk
            self.trade_meta['side'] = 'long'
            self.trade_meta['reached_1r'] = False

        # ========================================================
        # SHORT ENTRY (Bearish Reversal)
        # ========================================================
        prev_above_upper = prev_close > self.bb_upper[-2]
        now_inside_upper = price < self.bb_upper[-1]
        fib_key_s, fib_lvl_s = self._near_fib(prev_high, fib_levels, self.fib_tolerance)
        if fib_key_s is None:
            fib_key_s, fib_lvl_s = self._near_fib(prev_close, fib_levels, self.fib_tolerance)
        golden_s = fib_key_s in ('0.618', '0.786') if fib_key_s else False
        # VIX low or declining
        vix_ok_s = (vix <= vix_ma20) or (vix < self.vix_ma10[-1])

        if prev_above_upper and now_inside_upper and fib_key_s is not None and vix_ok_s:
            stop = fib_lvl_s + self.atr_buffer * self.atr[-1]
            risk = stop - price
            if risk <= 0:
                return
            target = price - self.rr_min * risk
            if (price - target) / risk < self.rr_min:
                return

            size = self._calc_size(risk, vix)
            if size <= 0:
                return

            print(f"🌙🔻 SHORT SIGNAL! Fib {fib_key_s}{' (GOLDEN)' if golden_s else ''} @ {fib_lvl_s:.2f} | "
                  f"Price: {price:.2f} | Stop: {stop:.2f} | Target: {target:.2f} | "
                  f"VIX: {vix:.2f} | Size: {size}")

            self.sell(size=size)
            self.trade_meta['entry_bar'] = i
            self.trade_meta['stop'] = stop
            self.trade_meta['target'] = target
            self.trade_meta['entry'] = price
            self.trade_meta['risk'] = risk
            self.trade_meta['side'] = 'short'
            self.trade_meta['reached_1r'] = False

    # ------------------------------------------------------------
    # Position sizing: risk 1% of equity, adjusted for VIX
    # ------------------------------------------------------------
    def _calc_size(self, risk_per_unit, vix):
        equity = self.equity
        risk_amount = equity * self.risk_pct
        # Reduce size when VIX > 30
        if vix > 30:
            risk_amount *= 0.5
        size = risk_amount / risk_per_unit
        size = int(round(size))
        # Cap by available cash (approx)
        max_units = int(self.equity / max(self.data.Close[-1], 1e-9))
        size = min(size, max_units)
        return max(size, 0)

    # ------------------------------------------------------------
    # Trade management: stops, targets, VIX exits, time stop
    # ------------------------------------------------------------
    def _manage_position(self, i):
        price = self.data.Close[-1]
        meta = self.trade_meta
        if not meta:
            return

        side = meta.get('side')
        entry = meta.get('entry')
        stop = meta.get('stop')
        target = meta.get('target')
        risk = meta.get('risk')
        entry_bar = meta.get('entry_bar')

        if side == 'long':
            # Update 1R tracking
            if price >= entry + risk:
                meta['reached_1r'] = True

            # Stop loss
            if price <= stop:
                print(f"🌙💀 LONG STOP HIT @ {price:.2f}")
                self.position.close()
                self.trade_meta = {}
                return
            # Profit target
            if price >= target:
                print(f"🌙🎯 LONG TARGET HIT @ {price:.2f}")
                self.position.close()
                self.trade_meta = {}
                return
            # VIX-based exit
            vix = self.vix_proxy[-1]
            vix_min10 = self.vix_min10[-1]
            if not np.isnan(vix_min10) and vix < vix_min10:
                print(f"🌙📉 VIX fade exit (long) @ {price:.2f} | VIX={vix:.2f}")
                self.position.close()
                self.trade_meta = {}
                return
            if vix > self.vix_high:
                print(f"🌙⚠️ VIX panic exit (long) @ {price:.2f} | VIX={vix:.2f}")
                self.position.close()
                self.trade_meta = {}
                return
            # Time stop: no +1R within N bars
            if (i - entry_bar) >= self.time_stop_bars and not meta.get('reached_1r'):
                print(f"🌙⏰ TIME STOP (long) @ {price:.2f}")
                self.position.close()
                self.trade_meta = {}
                return

        elif side == 'short':
            if price <= entry - risk:
                meta['reached_1r'] = True

            if price >= stop:
                print(f"🌙💀 SHORT STOP HIT @ {price:.2f}")
                self.position.close()
                self.trade_meta = {}
                return
            if price <= target:
                print(f"🌙🎯 SHORT TARGET HIT @ {price:.2f}")
                self.position.close()
                self.trade_meta = {}
                return
            vix = self.vix_proxy[-1]
            vix_max10 = self.vix_max10[-1]
            if not np.isnan(vix_max10) and vix > vix_max10:
                print(f"🌙📈 VIX spike exit (short) @ {price:.2f} | VIX={vix:.2f}")
                self.position.close()
                self.trade_meta = {}
                return
            if vix < self.vix_low:
                print(f"🌙⚠️ Complacency exit (short) @ {price:.2f} | VIX={vix:.2f}")
                self.position.close()
                self.trade_meta = {}
                return
            if (i - entry_bar) >= self.time_stop_bars and not meta.get('reached_1r'):
                print(f"🌙⏰ TIME STOP (short) @ {price:.2f}")
                self.position.close()
                self.trade_meta = {}
                return


# ============================================================
# RUN BACKTEST
# ============================================================
print("🌙✨ Starting Moon Dev Backtest Engine... 🚀")

bt = Backtest(
    data,
    FibBandReversal,
    cash=1_000_000,
    commission=0.001,
    exclusive=False,
)

stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙✨ Moon Dev Backtest Complete!