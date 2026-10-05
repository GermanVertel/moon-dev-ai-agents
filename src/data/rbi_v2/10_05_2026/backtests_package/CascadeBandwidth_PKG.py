import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# 🌙 Moon Dev CascadeBandwidth Strategy ✨🚀

class CascadeBandwidth(Strategy):
    # Strategy parameters
    bb_period = 20
    bb_std = 2
    bw_lookback = 100
    bw_percentile_entry = 20
    bw_percentile_exit = 80
    atr_period = 14
    atr_percentile_max = 90
    risk_per_trade = 0.01
    time_stop_bars = 25
    swing_lookback = 10
    cascade_lookback = 10
    cascade_std_mult = 2.0
    flip_window = 20
    flip_threshold = 3
    ratio_neutral_band = 0.05

    def init(self):
        # 🌙 Moon Dev indicator initialization
        print("🌙✨ Initializing CascadeBandwidth strategy indicators...")

        # Bollinger Bands
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS, self.data.Close, timeperiod=self.bb_period,
            nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0
        )
        # Bandwidth
        self.bandwidth = self.I(
            lambda u, m, l: (u - l) / m,
            self.bb_upper, self.bb_middle, self.bb_lower
        )
        # Bandwidth percentile rank (rolling)
        self.bw_pct = self.I(
            lambda bw: pd.Series(bw).rolling(self.bw_lookback).rank(pct=True),
            self.bandwidth
        )
        # ATR
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period)
        # ATR percentile rank
        self.atr_pct = self.I(
            lambda atr: pd.Series(atr).rolling(self.bw_lookback).rank(pct=True),
            self.atr
        )
        # Swing high/low
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=self.swing_lookback)
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=self.swing_lookback)

        # Simulated liquidation data (since real feed not available)
        # Use volume as proxy for liquidation cascade
        self.liq_vol = self.I(lambda v: v * np.random.uniform(0.5, 1.5, len(v)), self.data.Volume)
        self.liq_avg = self.I(talib.SMA, self.liq_vol, timeperiod=self.cascade_lookback)
        self.liq_std = self.I(talib.STDDEV, self.liq_vol, timeperiod=self.cascade_lookback)

        # Simulated long/short ratio (random walk around 1.0)
        np.random.seed(42)
        ratio = np.cumsum(np.random.randn(len(self.data)) * 0.02) + 1.0
        ratio = np.clip(ratio, 0.7, 1.3)
        self.ls_ratio = self.I(lambda: ratio)

        # Track state
        self.entry_price = None
        self.stop_price = None
        self.take_profit_1 = None
        self.take_profit_2 = None
        self.bars_in_trade = 0
        self.anchor_vwap = None
        self.anchor_idx = None

        print("🌙✨ Indicators ready! Let's cascade some bandwidth! 🚀")

    def next(self):
        # 🌙 Moon Dev main logic
        if len(self.data) < self.bw_lookback + self.bb_period:
            return

        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]

        # --- Detect cascade ---
        liq_spike = False
        liq_dir = 0  # 1 = short liquidation cascade (bullish), -1 = long liquidation cascade (bearish)
        if self.liq_std[-1] > 0:
            z = (self.liq_vol[-1] - self.liq_avg[-1]) / self.liq_std[-1]
            if z > self.cascade_std_mult:
                liq_spike = True
                # Direction: if price is falling, likely long liquidations
                if price < self.data.Close[-2]:
                    liq_dir = -1  # long cascade -> short entry
                else:
                    liq_dir = 1   # short cascade -> long entry

        # --- Clustered ratio flips ---
        flips = 0
        if len(self.ls_ratio) > self.flip_window:
            ratio_slice = self.ls_ratio[-self.flip_window:]
            for i in range(1, len(ratio_slice)):
                prev = ratio_slice[i-1]
                curr = ratio_slice[i]
                if (prev - 1.0) * (curr - 1.0) < 0:
                    flips += 1
        flip_signal = flips >= self.flip_threshold

        # --- Bandwidth extreme ---
        bw_low = self.bw_pct[-1] <= (self.bw_percentile_entry / 100.0)
        bw_high = self.bw_pct[-1] >= (self.bw_percentile_exit / 100.0)

        # --- Volatility filter ---
        atr_too_high = self.atr_pct[-1] >= (self.atr_percentile_max / 100.0)

        # --- Entry conditions ---
        if not self.position:
            if liq_spike and flip_signal and bw_low and not atr_too_high:
                # Long entry: short liquidation cascade + close above middle band
                if liq_dir == 1 and price > self.bb_middle[-1]:
                    # Risk management
                    stop = self.swing_low[-1] - 0.5 * self.atr[-1]
                    risk = price - stop
                    if risk > 0:
                        size = int(round((self.equity * self.risk_per_trade) / risk))
                        size = max(1, min(size, int(self.equity / price)))
                        self.buy(size=size)
                        self.entry_price = price
                        self.stop_price = stop
                        self.take_profit_1 = price + risk
                        self.take_profit_2 = price + 2 * risk
                        self.bars_in_trade = 0
                        # Anchor VWAP at swing low
                        self.anchor_idx = len(self.data) - self.swing_lookback
                        self.anchor_vwap = self._calc_anchored_vwap(self.anchor_idx)
                        print(f"🌙🚀 LONG ENTRY! Price: {price:.2f}, Size: {size}, Stop: {stop:.2f}, TP1: {self.take_profit_1:.2f}, TP2: {self.take_profit_2:.2f}")

                # Short entry: long liquidation cascade + close below middle band
                elif liq_dir == -1 and price < self.bb_middle[-1]:
                    stop = self.swing_high[-1] + 0.5 * self.atr[-1]
                    risk = stop - price
                    if risk > 0:
                        size = int(round((self.equity * self.risk_per_trade) / risk))
                        size = max(1, min(size, int(self.equity / price)))
                        self.sell(size=size)
                        self.entry_price = price
                        self.stop_price = stop
                        self.take_profit_1 = price - risk
                        self.take_profit_2 = price - 2 * risk
                        self.bars_in_trade = 0
                        self.anchor_idx = len(self.data) - self.swing_lookback
                        self.anchor_vwap = self._calc_anchored_vwap(self.anchor_idx)
                        print(f"🌙🔻 SHORT ENTRY! Price: {price:.2f}, Size: {size}, Stop: {stop:.2f}, TP1: {self.take_profit_1:.2f}, TP2: {self.take_profit_2:.2f}")

        # --- Exit management ---
        else:
            self.bars_in_trade += 1
            # Update anchored VWAP
            if self.anchor_idx is not None:
                self.anchor_vwap = self._calc_anchored_vwap(self.anchor_idx)

            if self.position.is_long:
                # Stop loss
                if low <= self.stop_price:
                    self.position.close()
                    print(f"🌙💥 LONG STOP LOSS at {self.stop_price:.2f}")
                # Take profit 1 (partial)
                elif high >= self.take_profit_1 and self.position.size > 0:
                    # Scale out half
                    half_size = max(1, int(self.position.size // 2))
                    self.position.close(half_size)
                    print(f"🌙💰 LONG TP1 hit at {self.take_profit_1:.2f}, scaled out {half_size}")
                # Take profit 2 (full)
                elif high >= self.take_profit_2:
                    self.position.close()
                    print(f"🌙💰 LONG TP2 hit at {self.take_profit_2:.2f}")
                # Anchored VWAP exit
                elif self.anchor_vwap is not None and high >= self.anchor_vwap:
                    # Check slope flattening
                    if self._vwap_slope_flattening(self.anchor_idx):
                        self.position.close()
                        print(f"🌙⚓ LONG VWAP EXIT at {self.anchor_vwap:.2f}")
                # Bandwidth expansion exit
                elif bw_high and price < self.data.Open[-1]:
                    self.position.close()
                    print(f"🌙📉 LONG BW EXIT at {price:.2f}")
                # Time stop
                elif self.bars_in_trade >= self.time_stop_bars:
                    self.position.close()
                    print(f"🌙⏰ LONG TIME STOP at {price:.2f}")

            elif self.position.is_short:
                if high >= self.stop_price:
                    self.position.close()
                    print(f"🌙💥 SHORT STOP LOSS at {self.stop_price:.2f}")
                elif low <= self.take_profit_1 and self.position.size < 0:
                    half_size = max(1, int(abs(self.position.size) // 2))
                    self.position.close(half_size)
                    print(f"🌙💰 SHORT TP1 hit at {self.take_profit_1:.2f}, scaled out {half_size}")
                elif low <= self.take_profit_2:
                    self.position.close()
                    print(f"🌙💰 SHORT TP2 hit at {self.take_profit_2:.2f}")
                elif self.anchor_vwap is not None and low <= self.anchor_vwap:
                    if self._vwap_slope_flattening(self.anchor_idx):
                        self.position.close()
                        print(f"🌙⚓ SHORT VWAP EXIT at {self.anchor_vwap:.2f}")
                elif bw_high and price > self.data.Open[-1]:
                    self.position.close()
                    print(f"🌙📈 SHORT BW EXIT at {price:.2f}")
                elif self.bars_in_trade >= self.time_stop_bars:
                    self.position.close()
                    print(f"🌙⏰ SHORT TIME STOP at {price:.2f}")

    def _calc_anchored_vwap(self, anchor_idx):
        """Calculate anchored VWAP from anchor index to current bar."""
        if anchor_idx is None or anchor_idx >= len(self.data):
            return None
        try:
            prices = (self.data.High[anchor_idx:] + self.data.Low[anchor_idx:] + self.data.Close[anchor_idx:]) / 3
            volumes = self.data.Volume[anchor_idx:]
            if volumes.sum() == 0:
                return None
            vwap = (prices * volumes).sum() / volumes.sum()
            return vwap
        except Exception:
            return None

    def _vwap_slope_flattening(self, anchor_idx):
        """Check if VWAP slope is flattening."""
        if anchor_idx is None or len(self.data) - anchor_idx < 5:
            return False
        try:
            vwap_now = self._calc_anchored_vwap(anchor_idx)
            vwap_prev = self._calc_anchored_vwap(anchor_idx)
            if vwap_now is None or vwap_prev is None:
                return False
            # Simple slope check
            slope = (vwap_now - vwap_prev) / max(abs(vwap_prev), 1e-9)
            return abs(slope) < 0.001
        except Exception:
            return False


# 🌙 Load and prepare data
print("🌙✨ Loading BTC-USD 15m data...")
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Ensure proper column mapping
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

# Set datetime index if present
if 'datetime' in data.columns:
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')

print(f"🌙✨ Data loaded: {len(data)} bars")
print(f"🌙✨ Columns: {list(data.columns)}")

# 🚀 Run backtest
bt = Backtest(data, CascadeBandwidth, cash=1_000_000, commission=0.001, exclusive_orders=True)
stats = bt.run()
print(stats)
print(stats._strategy)