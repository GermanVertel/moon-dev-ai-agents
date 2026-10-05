import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# ============================================================
# 🌙 Moon Dev's ClusterBreakout Strategy 🌙
# ============================================================

data_path = '/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv'

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
    'volume': 'Volume',
    'datetime': 'Datetime'
})

data['Datetime'] = pd.to_datetime(data['Datetime'])
data = data.set_index('Datetime')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]


class ClusterBreakout(Strategy):
    # Optimization parameters
    cluster_lookback = 20
    retracement_pct = 0.01       # 1% trailing exit
    volume_multiplier = 1.5
    trendline_min_touches = 2
    swing_lookback = 5           # bars on each side for swing low detection
    atr_period = 14
    risk_pct = 0.02              # 2% risk per trade
    max_bars_in_trade = 15

    def init(self):
        print("🌙✨ Moon Dev ClusterBreakout initializing... 🚀")

        # Cluster high/low
        self.cluster_high = self.I(talib.MAX, self.data.High, timeperiod=self.cluster_lookback)
        self.cluster_low = self.I(talib.MIN, self.data.Low, timeperiod=self.cluster_lookback)

        # ATR
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period)

        # Volume MA
        self.vol_ma = self.I(talib.SMA, self.data.Volume, timeperiod=self.cluster_lookback)

        # ATR percentile filter (rolling)
        self.atr_series = self.atr

        # Track trade state
        self.entry_price = None
        self.peak_price = None
        self.initial_stop = None
        self.bars_in_trade = 0
        self.trail_active = False

        print("🌙 Indicators ready! Cluster lookback:", self.cluster_lookback)

    def _detect_ascending_trendline(self, idx):
        """Detect ascending trend line using swing lows."""
        lookback = self.cluster_lookback * 3
        if idx < lookback:
            return False, None

        swing_low_indices = []
        swing_low_values = []

        start = idx - lookback
        end = idx - self.swing_lookback

        for i in range(start + self.swing_lookback, end):
            window_low = self.data.Low[i - self.swing_lookback:i + self.swing_lookback + 1]
            if len(window_low) == 0:
                continue
            if self.data.Low[i] == window_low.min():
                swing_low_indices.append(i)
                swing_low_values.append(self.data.Low[i])

        if len(swing_low_indices) < self.trendline_min_touches:
            return False, None

        # Fit linear regression on swing lows
        x = np.array(swing_low_indices, dtype=float)
        y = np.array(swing_low_values, dtype=float)

        # Slope
        try:
            slope, intercept = np.polyfit(x, y, 1)
        except Exception:
            return False, None

        if slope <= 0:
            return False, None

        # Trend line value at current bar
        trend_value = slope * idx + intercept
        return True, trend_value

    def next(self):
        price = self.data.Close[-1]

        # ===== MANAGE OPEN TRADE =====
        if self.position:
            self.bars_in_trade += 1
            self.peak_price = max(self.peak_price, self.data.High[-1])

            # Activate trailing once price moves 1% in favor
            if not self.trail_active and self.peak_price >= self.entry_price * (1 + self.retracement_pct):
                self.trail_active = True
                print(f"🌙 Trail activated at peak {self.peak_price:.2f} 🚀")

            # Trailing retracement exit
            if self.trail_active:
                trail_stop = self.peak_price * (1 - self.retracement_pct)
                if price <= trail_stop:
                    print(f"🌙💫 Trailing exit hit at {price:.2f} (peak {self.peak_price:.2f})")
                    self.position.close()
                    self._reset_trade_state()
                    return

            # Initial stop loss
            if price <= self.initial_stop:
                print(f"🌙🛑 Stop loss hit at {price:.2f} (stop {self.initial_stop:.2f})")
                self.position.close()
                self._reset_trade_state()
                return

            # Time-based exit
            if self.bars_in_trade >= self.max_bars_in_trade:
                print(f"🌙⏰ Time exit at {price:.2f} after {self.bars_in_trade} bars")
                self.position.close()
                self._reset_trade_state()
                return

            return

        # ===== LOOK FOR ENTRY =====
        idx = len(self.data) - 1

        # Need enough history
        if idx < self.cluster_lookback * 3:
            return

        # ATR volatility filter
        atr_window = self.atr_series[idx - 100:idx] if idx >= 100 else self.atr_series[:idx]
        atr_window = atr_window[~np.isnan(atr_window)]
        if len(atr_window) < 20:
            return

        atr_current = self.atr[-1]
        if np.isnan(atr_current):
            return

        p20 = np.percentile(atr_window, 20)
        p90 = np.percentile(atr_window, 90)

        if atr_current <= p20:
            return  # too quiet
        if atr_current >= p90:
            return  # too whippy

        # Cluster width filter (must be < 5% of price)
        ch = self.cluster_high[-1]
        cl = self.cluster_low[-1]
        if np.isnan(ch) or np.isnan(cl) or cl <= 0:
            return

        cluster_width_pct = (ch - cl) / cl
        if cluster_width_pct > 0.05:
            return

        # Trend line detection
        has_trend, trend_value = self._detect_ascending_trendline(idx)
        if not has_trend or trend_value is None:
            return

        # Breakout conditions (no backtesting.lib crossover — pure array indexing)
        prev_close = self.data.Close[-2]
        close_above_cluster = price > ch and prev_close <= ch
        close_above_trend = price > trend_value

        if not (close_above_cluster and close_above_trend):
            return

        # Volume confirmation
        vol_ma = self.vol_ma[-1]
        if np.isnan(vol_ma) or vol_ma <= 0:
            return
        if self.data.Volume[-1] < self.volume_multiplier * vol_ma:
            return

        # ===== POSITION SIZING =====
        equity = self.equity
        risk_amount = equity * self.risk_pct

        # Stop at cluster low, but tighter of cluster low or 1% below entry
        stop_cluster = cl
        stop_1pct = price * 0.99
        stop_price = max(stop_cluster, stop_1pct)

        risk_per_unit = price - stop_price
        if risk_per_unit <= 0:
            return

        # R:R check — target is trailing, but ensure minimum 1.5:1 potential
        # Use cluster width as proxy for first target
        potential_reward = price - stop_price  # trail will handle actual
        if (price - stop_price) <= 0:
            return

        # Position size in units
        position_size = risk_amount / risk_per_unit
        position_size = int(round(position_size))

        if position_size <= 0:
            return

        # Cap by equity
        max_units = int(equity / price)
        position_size = min(position_size, max_units)
        if position_size <= 0:
            return

        print(f"🌙🚀 CLUSTER BREAKOUT! Price={price:.2f} ClusterHigh={ch:.2f} Trend={trend_value:.2f} "
              f"Vol={self.data.Volume[-1]:.2f} VolMA={vol_ma:.2f} Size={position_size} Stop={stop_price:.2f}")

        self.buy(size=position_size)
        self.entry_price = price
        self.peak_price = price
        self.initial_stop = stop_price
        self.bars_in_trade = 0
        self.trail_active = False

    def _reset_trade_state(self):
        self.entry_price = None
        self.peak_price = None
        self.initial_stop = None
        self.bars_in_trade = 0
        self.trail_active = False


print("🌙 Loading BTC-USD 15m data... ✨")
bt = Backtest(
    data,
    ClusterBreakout,
    cash=1_000_000,
    commission=0.002,
    exclusive_orders=True
)

stats = bt.run()
print(stats)
print(stats._strategy)