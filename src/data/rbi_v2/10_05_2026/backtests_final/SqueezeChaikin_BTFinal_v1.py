import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# ============================================================
# 🌙 Moon Dev's SqueezeChaikin Backtest 🚀
# ============================================================

print("🌙 Moon Dev initializing SqueezeChaikin strategy... ✨")

data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper case mapping
data = data.rename(columns={
    'datetime': 'datetime',
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')

# 🌙 Ensure numeric dtype for OHLCV (fixes "input array type is not double")
for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    data[col] = pd.to_numeric(data[col], errors='coerce').astype(float)

data = data.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])

print(f"🌙 Data loaded: {len(data)} bars ✨")
print(f"🚀 Columns: {list(data.columns)}")


class SqueezeChaikin(Strategy):
    # Parameters
    bb_period = 20
    bb_std = 2.0
    squeeze_lookback = 125
    squeeze_pct = 0.20
    vol_period = 20
    vol_mult = 1.5
    atr_period = 14
    atr_mult = 1.5
    risk_pct = 0.01
    time_stop = 30
    div_window = 30

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        # 🌙 Cast to float64 numpy arrays for talib compatibility
        close_arr = np.asarray(close, dtype=np.float64)
        high_arr = np.asarray(high, dtype=np.float64)
        low_arr = np.asarray(low, dtype=np.float64)
        volume_arr = np.asarray(volume, dtype=np.float64)

        # Bollinger Bands
        self.mid = self.I(talib.SMA, close_arr, timeperiod=self.bb_period)
        self.std = self.I(talib.STDDEV, close_arr, timeperiod=self.bb_period, nbdev=1)
        upper = self.mid + self.bb_std * self.std
        lower = self.mid - self.bb_std * self.std
        self.upper = self.I(lambda: upper, name='UpperBand')
        self.lower = self.I(lambda: lower, name='LowerBand')

        # BandWidth
        bw = (upper - lower) / self.mid
        self.bw = self.I(lambda: bw, name='BandWidth')

        # Squeeze threshold: rolling 20th percentile
        bw_series = pd.Series(bw)
        sq_thresh = bw_series.rolling(self.squeeze_lookback).quantile(self.squeeze_pct)
        self.sq_thresh = self.I(lambda: sq_thresh.values, name='SqueezeThresh')

        # Volume SMA
        self.vol_sma = self.I(talib.SMA, volume_arr, timeperiod=self.vol_period)

        # ATR
        self.atr = self.I(talib.ATR, high_arr, low_arr, close_arr, timeperiod=self.atr_period)

        # Chaikin Oscillator
        adl = self.I(self._adl, high_arr, low_arr, close_arr, volume_arr, name='ADL')
        ema3 = self.I(talib.EMA, adl, timeperiod=3)
        ema10 = self.I(talib.EMA, adl, timeperiod=10)
        chaikin = ema3 - ema10
        self.chaikin = self.I(lambda: chaikin, name='Chaikin')

        # Tracking
        self.entry_bar = 0
        self.entry_price = 0
        self.stop_price = 0
        self.initial_risk = 0

        print("🌙 Indicators initialized ✨")

    @staticmethod
    def _adl(high, low, close, volume):
        high = np.asarray(high, dtype=np.float64)
        low = np.asarray(low, dtype=np.float64)
        close = np.asarray(close, dtype=np.float64)
        volume = np.asarray(volume, dtype=np.float64)
        rng = high - low
        mfm = np.where(rng > 0, ((close - low) - (high - close)) / np.where(rng == 0, 1, rng), 0.0)
        mfv = mfm * volume
        return np.cumsum(mfv)

    def _detect_bullish_divergence(self, i):
        """Price LL while Chaikin HL within squeeze window."""
        window = self.div_window
        if i < window + 5:
            return False
        start = i - window
        # Find local min of price in first half, current low region
        prices = self.data.Low[start:i]
        chaik = self.chaikin[start:i]
        if len(prices) < 10:
            return False
        # Recent low (last 5 bars)
        recent_low = np.min(prices[-5:])
        recent_chaik = np.min(chaik[-5:])
        # Prior low (first part of window)
        prior_low = np.min(prices[:len(prices)//2])
        prior_chaik = np.min(chaik[:len(chaik)//2])
        # Price made lower low, Chaikin made higher low
        return recent_low <= prior_low and recent_chaik > prior_chaik

    def _detect_bearish_divergence(self, i):
        """Price HH while Chaikin LH within squeeze window."""
        window = self.div_window
        if i < window + 5:
            return False
        start = i - window
        prices = self.data.High[start:i]
        chaik = self.chaikin[start:i]
        if len(prices) < 10:
            return False
        recent_high = np.max(prices[-5:])
        recent_chaik = np.max(chaik[-5:])
        prior_high = np.max(prices[:len(prices)//2])
        prior_chaik = np.max(chaik[:len(chaik)//2])
        return recent_high >= prior_high and recent_chaik < prior_chaik

    def next(self):
        i = len(self.data) - 1
        if i < self.squeeze_lookback + 5:
            return

        price = self.data.Close[-1]
        upper = self.upper[-1]
        lower = self.lower[-1]
        mid = self.mid[-1]
        bw = self.bw[-1]
        sq = self.sq_thresh[-1]
        vol = self.data.Volume[-1]
        vol_sma = self.vol_sma[-1]
        chaik = self.chaikin[-1]
        atr = self.atr[-1]

        if np.isnan([upper, lower, mid, bw, sq, vol_sma, chaik, atr]).any():
            return

        # Manage open position
        if self.position:
            bars_held = i - self.entry_bar
            # Time stop
            if bars_held >= self.time_stop:
                print(f"⏰ Moon Dev time stop hit at bar {i} — closing 🌙")
                self.position.close()
                return

            if self.position.is_long:
                # Trailing: if moved 1R in favor, trail to mid
                if price > self.entry_price + self.initial_risk:
                    if mid > self.stop_price:
                        self.stop_price = mid
                # Stop loss
                if price <= self.stop_price:
                    print(f"🛑 Long stop hit at {price:.2f} (stop {self.stop_price:.2f}) 🌙")
                    self.position.close()
                    return
                # Midline exit
                if price < mid:
                    print(f"✅ Long exit: close {price:.2f} < mid {mid:.2f} 🚀")
                    self.position.close()
                    return
            else:
                if price < self.entry_price - self.initial_risk:
                    if mid < self.stop_price:
                        self.stop_price = mid
                if price >= self.stop_price:
                    print(f"🛑 Short stop hit at {price:.2f} (stop {self.stop_price:.2f}) 🌙")
                    self.position.close()
                    return
                if price > mid:
                    print(f"✅ Short exit: close {price:.2f} > mid {mid:.2f} 🚀")
                    self.position.close()
                    return
            return

        # Entry logic
        squeeze_active = bw < sq
        vol_confirm = vol > self.vol_mult * vol_sma

        # Long
        if squeeze_active and price > upper and vol_confirm:
            if self._detect_bullish_divergence(i):
                stop = min(lower, price - self.atr_mult * atr)
                risk = price - stop
                if risk > 0:
                    # 🌙 Use fractional equity sizing (valid 0 < size < 1)
                    size = 0.95
                    print(f"🌙🚀 LONG Squeeze Breakout @ {price:.2f} | upper={upper:.2f} bw={bw:.4f} sq={sq:.4f} chaikin={chaik:.2f} ✨")
                    self.buy(size=size)
                    self.entry_bar = i
                    self.entry_price = price
                    self.stop_price = stop
                    self.initial_risk = risk
                    return

        # Short
        if squeeze_active and price < lower and vol_confirm:
            if self._detect_bearish_divergence(i):
                stop = max(upper, price + self.atr_mult * atr)
                risk = stop - price
                if risk > 0:
                    # 🌙 Use fractional equity sizing (valid 0 < size < 1)
                    size = 0.95
                    print(f"🌙🔻 SHORT Squeeze Breakout @ {price:.2f} | lower={lower:.2f} bw={bw:.4f} sq={sq:.4f} chaikin={chaik:.2f} ✨")
                    self.sell(size=size)
                    self.entry_bar = i
                    self.entry_price = price
                    self.stop_price = stop
                    self.initial_risk = risk
                    return


print("🌙 Running Moon Dev SqueezeChaikin backtest... 🚀")
bt = Backtest(data, SqueezeChaikin, cash=1_000_000, commission=0.001, exclusive_orders=True)
stats = bt.run()
print(stats)
print(stats._strategy)