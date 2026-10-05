import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# 🌙 Moon Dev Inverse Volume Divergence Strategy ✨

def compute_rsi(close, period=14):
    return talib.RSI(close, timeperiod=period)

def compute_stoch(high, low, close, k=14, d=3, smooth=3):
    slowk, slowd = talib.STOCH(high, low, close,
                               fastk_period=k,
                               slowk_period=smooth,
                               slowk_matype=0,
                               slowd_period=d,
                               slowd_matype=0)
    return slowk, slowd

def compute_atr(high, low, close, period=14):
    return talib.ATR(high, low, close, timeperiod=period)

def compute_vol_sma(volume, period=20):
    return talib.SMA(volume, timeperiod=period)

class InverseVolumeDivergence(Strategy):
    # Parameters
    rsi_period = 14
    stoch_k = 14
    stoch_d = 3
    stoch_smooth = 3
    atr_period = 14
    vol_ma_period = 20
    swing_lookback = 20
    risk_pct = 0.02
    rr_ratio = 2.0
    atr_stop_mult = 1.5
    trailing_atr_mult = 1.0

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        vol = self.data.Volume

        # 🌙 Indicators
        self.rsi = self.I(compute_rsi, close, self.rsi_period)
        self.stoch_k, self.stoch_d = self.I(compute_stoch, high, low, close,
                                            self.stoch_k, self.stoch_d, self.stoch_smooth)
        self.atr = self.I(compute_atr, high, low, close, self.atr_period)
        self.vol_ma = self.I(compute_vol_sma, vol, self.vol_ma_period)

        # Swing highs/lows for divergence detection
        self.swing_high = self.I(talib.MAX, high, timeperiod=self.swing_lookback)
        self.swing_low = self.I(talib.MIN, low, timeperiod=self.swing_lookback)

        # Track trade state for trailing stop
        self.entry_price = None
        self.entry_atr = None
        self.stop_price = None
        self.target_price = None
        self.trade_dir = None

        print("🌙✨ Moon Dev Inverse Volume Divergence initialized! 🚀")

    def _bullish_candle(self, i):
        o = self.data.Open[i]
        c = self.data.Close[i]
        h = self.data.High[i]
        l = self.data.Low[i]
        body = abs(c - o)
        rng = h - l if h > l else 1e-9
        # Hammer: small body, long lower wick
        lower_wick = min(o, c) - l
        upper_wick = h - max(o, c)
        hammer = (lower_wick > 2 * body) and (upper_wick < body) and (body / rng < 0.4)
        # Bullish engulfing
        if i > 0:
            prev_o = self.data.Open[i-1]
            prev_c = self.data.Close[i-1]
            engulf = (c > o) and (prev_c < prev_o) and (c >= prev_o) and (o <= prev_c)
        else:
            engulf = False
        return hammer or engulf

    def _bearish_candle(self, i):
        o = self.data.Open[i]
        c = self.data.Close[i]
        h = self.data.High[i]
        l = self.data.Low[i]
        body = abs(c - o)
        rng = h - l if h > l else 1e-9
        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l
        # Shooting star
        shooting = (upper_wick > 2 * body) and (lower_wick < body) and (body / rng < 0.4)
        # Bearish engulfing
        if i > 0:
            prev_o = self.data.Open[i-1]
            prev_c = self.data.Close[i-1]
            engulf = (c < o) and (prev_c > prev_o) and (c <= prev_o) and (o >= prev_c)
        else:
            engulf = False
        return shooting or engulf

    def _bullish_divergence(self, i, lookback=15):
        if i < lookback + 2:
            return False
        # Price lower low over lookback
        recent_low = min(self.data.Low[i-lookback:i+1])
        prev_low = min(self.data.Low[i-2*lookback:i-lookback+1]) if i >= 2*lookback else None
        if prev_low is None:
            return False
        price_ll = recent_low < prev_low
        # RSI higher low
        recent_rsi = min(self.rsi[i-lookback:i+1])
        prev_rsi = min(self.rsi[i-2*lookback:i-lookback+1])
        rsi_hl = recent_rsi > prev_rsi
        # Stoch higher low
        recent_k = min(self.stoch_k[i-lookback:i+1])
        prev_k = min(self.stoch_k[i-2*lookback:i-lookback+1])
        stoch_hl = recent_k > prev_k
        return price_ll and rsi_hl and stoch_hl

    def _bearish_divergence(self, i, lookback=15):
        if i < lookback + 2:
            return False
        recent_high = max(self.data.High[i-lookback:i+1])
        prev_high = max(self.data.High[i-2*lookback:i-lookback+1]) if i >= 2*lookback else None
        if prev_high is None:
            return False
        price_hh = recent_high > prev_high
        recent_rsi = max(self.rsi[i-lookback:i+1])
        prev_rsi = max(self.rsi[i-2*lookback:i-lookback+1])
        rsi_lh = recent_rsi < prev_rsi
        recent_k = max(self.stoch_k[i-lookback:i+1])
        prev_k = max(self.stoch_k[i-2*lookback:i-lookback+1])
        stoch_lh = recent_k < prev_k
        return price_hh and rsi_lh and stoch_lh

    def _inverse_volume_breakdown(self, i, lookback=10):
        # Price breaks below recent swing low but volume decreased
        if i < lookback + 1:
            return False
        recent_low = min(self.data.Low[i-lookback:i])
        breakdown = self.data.Low[i] < recent_low
        vol_decreasing = self.data.Volume[i] < self.vol_ma[i]
        return breakdown and vol_decreasing

    def _inverse_volume_breakout(self, i, lookback=10):
        if i < lookback + 1:
            return False
        recent_high = max(self.data.High[i-lookback:i])
        breakout = self.data.High[i] > recent_high
        vol_decreasing = self.data.Volume[i] < self.vol_ma[i]
        return breakout and vol_decreasing

    def next(self):
        i = len(self.data) - 1
        if i < 50:
            return

        price = self.data.Close[i]

        # Manage open trades
        if self.position:
            if self.trade_dir == 'long':
                # Trailing stop after 1x ATR profit
                if price > self.entry_price + self.entry_atr:
                    new_stop = price - self.trailing_atr_mult * self.atr[i]
                    if new_stop > self.stop_price:
                        self.stop_price = new_stop
                # Exit on divergence failure: new low without new RSI low
                if self.data.Low[i] < self.data.Low[i-1] and self.rsi[i] < self.rsi[i-1]:
                    if price < self.entry_price:
                        print(f"🌙💥 Long divergence failure exit at {price:.2f}")
                        self.position.close()
                        self.trade_dir = None
                        return
                if price <= self.stop_price:
                    print(f"🌙🛑 Long stop hit at {price:.2f}")
                    self.position.close()
                    self.trade_dir = None
                    return
                if price >= self.target_price:
                    print(f"🌙🎯 Long target hit at {price:.2f}")
                    self.position.close()
                    self.trade_dir = None
                    return
            elif self.trade_dir == 'short':
                if price < self.entry_price - self.entry_atr:
                    new_stop = price + self.trailing_atr_mult * self.atr[i]
                    if new_stop < self.stop_price:
                        self.stop_price = new_stop
                if self.data.High[i] > self.data.High[i-1] and self.rsi[i] > self.rsi[i-1]:
                    if price > self.entry_price:
                        print(f"🌙💥 Short divergence failure exit at {price:.2f}")
                        self.position.close()
                        self.trade_dir = None
                        return
                if price >= self.stop_price:
                    print(f"🌙🛑 Short stop hit at {price:.2f}")
                    self.position.close()
                    self.trade_dir = None
                    return
                if price <= self.target_price:
                    print(f"🌙🎯 Short target hit at {price:.2f}")
                    self.position.close()
                    self.trade_dir = None
                    return
            return

        # --- Entry Logic ---
        # Long: inverse volume breakdown + bullish candle + bullish RSI/Stoch divergence
        if (self._inverse_volume_breakdown(i) and
                self._bullish_candle(i) and
                self._bullish_divergence(i)):
            atr_val = self.atr[i]
            if atr_val <= 0:
                return
            stop = self.data.Low[i] - self.atr_stop_mult * atr_val
            risk = price - stop
            if risk <= 0:
                return
            target = price + self.rr_ratio * risk
            size = int(round(1000000 / price))
            if size < 1:
                size = 1
            print(f"🌙🚀 LONG signal! Price={price:.2f} Stop={stop:.2f} Target={target:.2f} Size={size}")
            self.buy(size=size)
            self.entry_price = price
            self.entry_atr = atr_val
            self.stop_price = stop
            self.target_price = target
            self.trade_dir = 'long'

        # Short: inverse volume breakout + bearish candle + bearish RSI/Stoch divergence
        elif (self._inverse_volume_breakout(i) and
              self._bearish_candle(i) and
              self._bearish_divergence(i)):
            atr_val = self.atr[i]
            if atr_val <= 0:
                return
            stop = self.data.High[i] + self.atr_stop_mult * atr_val
            risk = stop - price
            if risk <= 0:
                return
            target = price - self.rr_ratio * risk
            size = int(round(1000000 / price))
            if size < 1:
                size = 1
            print(f"🌙🔻 SHORT signal! Price={price:.2f} Stop={stop:.2f} Target={target:.2f} Size={size}")
            self.sell(size=size)
            self.entry_price = price
            self.entry_atr = atr_val
            self.stop_price = stop
            self.target_price = target
            self.trade_dir = 'short'


# 🌙 Load and clean data
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

print(f"🌙✨ Loaded {len(data)} bars of data 🚀")

bt = Backtest(data, InverseVolumeDivergence, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)