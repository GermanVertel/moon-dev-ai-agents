import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

print("🌙 Moon Dev's CompressionSqueeze Backtest Initializing... ✨")

# Load data
data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Set datetime index BEFORE renaming
if 'datetime' in data.columns:
    data = data.set_index(pd.to_datetime(data['datetime']))
    data = data.drop(columns=['datetime'])

# Proper column mapping
data = data.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})

# Ensure required columns
required = ['Open', 'High', 'Low', 'Close', 'Volume']
for col in required:
    if col not in data.columns:
        raise ValueError(f"Missing required column: {col}")

data = data[required].astype(float)
print(f"🌙 Data loaded: {len(data)} bars 🚀")


class CompressionSqueeze(Strategy):
    # Parameters
    bb_period = 20
    bb_std = 2.0
    kc_ema_period = 20
    kc_atr_period = 10
    kc_mult = 1.5
    bbwp_lookback = 252
    bbwp_threshold = 20
    squeeze_ratio_min = 0.0
    squeeze_consecutive = 5
    composite_threshold = 70
    vol_mult = 1.5
    vol_avg_period = 20
    adx_threshold = 20
    atr_stop_period = 14
    chandelier_mult = 2.5
    time_stop_bars = 15
    risk_pct = 0.01
    max_positions = 3

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        # Bollinger Bands
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS, close, timeperiod=self.bb_period,
            nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0
        )

        # Keltner Channels
        self.kc_ema = self.I(talib.EMA, close, timeperiod=self.kc_ema_period)
        self.kc_atr = self.I(talib.ATR, high, low, close, timeperiod=self.kc_atr_period)
        self.kc_upper = self.I(lambda e, a: e + self.kc_mult * a, self.kc_ema, self.kc_atr)
        self.kc_lower = self.I(lambda e, a: e - self.kc_mult * a, self.kc_ema, self.kc_atr)

        # ATR 14 for stops
        self.atr14 = self.I(talib.ATR, high, low, close, timeperiod=self.atr_stop_period)

        # EMA 20
        self.ema20 = self.I(talib.EMA, close, timeperiod=20)

        # ADX
        self.adx = self.I(talib.ADX, high, low, close, timeperiod=14)

        # Volume average
        self.vol_avg = self.I(talib.SMA, volume, timeperiod=self.vol_avg_period)

        # BB Width
        self.bb_width = self.I(
            lambda u, l, m: (u - l) / np.where(m == 0, np.nan, m),
            self.bb_upper, self.bb_lower, self.bb_middle
        )

        # KC Width
        self.kc_width = self.I(
            lambda u, l: u - l, self.kc_upper, self.kc_lower
        )

        # Squeeze ratio
        self.squeeze_ratio = self.I(
            lambda kw, bw: (kw - bw) / np.where(kw == 0, np.nan, kw),
            self.kc_width, self.bb_width
        )

        # BBWP - rolling percentile rank of bb_width
        def bbwp_func(bw):
            out = np.full(len(bw), np.nan)
            for i in range(self.bbwp_lookback, len(bw)):
                window = bw[i - self.bbwp_lookback:i + 1]
                window = window[~np.isnan(window)]
                if len(window) < 50:
                    continue
                cur = bw[i]
                if np.isnan(cur):
                    continue
                out[i] = (np.sum(window <= cur) / len(window)) * 100.0
            return out

        self.bbwp = self.I(bbwp_func, self.bb_width)

        # Composite score
        self.composite = self.I(
            lambda bbwp, sr: (100 - bbwp) * 0.5 + (np.nan_to_num(sr) * 100) * 0.5,
            self.bbwp, self.squeeze_ratio
        )

        # TTM momentum histogram proxy: close - SMA(close,20)
        self.sma20 = self.I(talib.SMA, close, timeperiod=20)
        self.ttm_hist = self.I(lambda c, s: c - s, close, self.sma20)

        # Track consecutive squeeze bars
        self.squeeze_streak = self.I(
            lambda sr: self._compute_streak(sr), self.squeeze_ratio
        )

        # State
        self.entry_bar = None
        self.entry_price = None
        self.stop_price = None
        self.target_price = None
        self.highest_since_entry = None

    def _compute_streak(self, sr):
        out = np.zeros(len(sr))
        count = 0
        for i in range(len(sr)):
            if not np.isnan(sr[i]) and sr[i] > 0:
                count += 1
            else:
                count = 0
            out[i] = count
        return out

    def next(self):
        i = len(self.data) - 1
        if i < self.bbwp_lookback + 5:
            return

        price = self.data.Close[-1]

        # Manage existing position
        if self.position:
            self.highest_since_entry = max(self.highest_since_entry, self.data.High[-1])

            # Chandelier trailing stop
            chandelier = self.highest_since_entry - self.chandelier_mult * self.atr14[-1]
            if chandelier > self.stop_price:
                self.stop_price = chandelier

            # Time stop
            bars_held = i - self.entry_bar
            if bars_held >= self.time_stop_bars:
                print(f"🌙 Time stop hit after {bars_held} bars — exiting at {price:.2f} ⏰")
                self.position.close()
                self._reset_state()
                return

            # Stop loss
            if self.data.Low[-1] <= self.stop_price:
                print(f"🛑 Stop hit at {self.stop_price:.2f} — exiting ✨")
                self.position.close()
                self._reset_state()
                return

            # Target
            if self.data.High[-1] >= self.target_price:
                print(f"🎯 Target hit at {self.target_price:.2f} — taking profit 🚀")
                self.position.close()
                self._reset_state()
                return

            # Volatility exit: BBWP > 80
            if not np.isnan(self.bbwp[-1]) and self.bbwp[-1] > 80:
                print(f"🌪️ BBWP > 80 ({self.bbwp[-1]:.1f}) — volatility exhausted, exiting 🌙")
                self.position.close()
                self._reset_state()
                return
            return

        # Entry logic (no position)
        if len(self.trades) >= self.max_positions:
            return

        # Previous bar composite score
        prev_composite = self.composite[-2] if len(self.composite) > 1 else np.nan
        prev_bbwp = self.bbwp[-2] if len(self.bbwp) > 1 else np.nan
        prev_squeeze_ratio = self.squeeze_ratio[-2] if len(self.squeeze_ratio) > 1 else np.nan
        prev_streak = self.squeeze_streak[-2] if len(self.squeeze_streak) > 1 else 0

        if np.isnan(prev_composite) or np.isnan(prev_bbwp):
            return

        # 1. Regime gate
        if prev_composite < self.composite_threshold:
            return

        # Both sub-conditions
        if prev_bbwp > self.bbwp_threshold:
            return
        if prev_squeeze_ratio <= self.squeeze_ratio_min:
            return
        if prev_streak < self.squeeze_consecutive:
            return

        # 2. Expansion trigger: close > upper BB and close > prior high
        if price <= self.bb_upper[-1]:
            return
        if price <= self.data.High[-2]:
            return

        # 3. Momentum filter: close > EMA20
        if price <= self.ema20[-1]:
            return

        # 4. Volume confirmation
        if self.data.Volume[-1] < self.vol_mult * self.vol_avg[-1]:
            return

        # 5. TTM histogram flip positive
        if self.ttm_hist[-1] <= 0:
            return

        # Market context: ADX > 20
        if np.isnan(self.adx[-1]) or self.adx[-1] <= self.adx_threshold:
            return

        # Risk management: stop distance
        atr_stop = price - 1.5 * self.atr14[-1]
        bb_stop = self.bb_lower[-1]
        stop = max(atr_stop, bb_stop)  # tighter (higher) stop for long
        risk = price - stop
        if risk <= 0:
            return

        # Position sizing: risk 1% of equity
        equity = self.equity
        risk_amount = equity * self.risk_pct
        size = int(round(risk_amount / risk))
        if size <= 0:
            return
        # Cap size to available cash
        max_size = int(equity / price)
        size = min(size, max_size)
        if size <= 0:
            return

        # Target: KC width measured move (2R)
        kc_width_val = self.kc_width[-1]
        target = price + 2 * risk

        print(f"🌙✨ SQUEEZE RELEASE DETECTED! Composite={prev_composite:.1f} BBWP={prev_bbwp:.1f} "
              f"Streak={prev_streak:.0f} ADX={self.adx[-1]:.1f} 🚀")
        print(f"   Entry={price:.2f} Stop={stop:.2f} Target={target:.2f} Size={size}")

        self.buy(size=size)
        self.entry_bar = i
        self.entry_price = price
        self.stop_price = stop
        self.target_price = target
        self.highest_since_entry = self.data.High[-1]

    def _reset_state(self):
        self.entry_bar = None
        self.entry_price = None
        self.stop_price = None
        self.target_price = None
        self.highest_since_entry = None


bt = Backtest(
    data, CompressionSqueeze,
    cash=1_000_000,
    commission=0.001
)

stats = bt.run()
print(stats)
print(stats._strategy)