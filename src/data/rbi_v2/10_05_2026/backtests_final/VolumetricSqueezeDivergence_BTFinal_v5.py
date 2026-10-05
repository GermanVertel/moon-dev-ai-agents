import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# 🌙 Moon Dev's Volumetric SqueezeDivergence Backtest 🚀

def vw_rsi(close, volume, period=14, vol_lookback=20):
    """Volume-Weighted RSI 🌙"""
    close = pd.Series(np.asarray(close, dtype=np.float64))
    volume = pd.Series(np.asarray(volume, dtype=np.float64))
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_vol = volume.rolling(vol_lookback).mean()
    vol_weight = (volume / avg_vol).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    vw_gain = gain * vol_weight
    vw_loss = loss * vol_weight
    avg_gain = vw_gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = vw_loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50).values


class VolumetricSqueezeDivergence(Strategy):
    bb_period = 20
    bb_std = 2.0
    bbw_lookback = 125
    bbw_pct_threshold = 0.20
    vol_sma_period = 20
    vol_mult = 1.3
    atr_period = 14
    div_lookback = 15
    risk_pct = 0.01
    time_stop_bars = 18
    swing_window = 5

    def init(self):
        # 🌙 Bollinger Bands
        self.bb_mid = self.I(talib.SMA, self.data.Close, timeperiod=self.bb_period)
        self.bb_stddev = self.I(talib.STDDEV, self.data.Close, timeperiod=self.bb_period, nbdev=1)
        self.bb_upper = self.I(lambda c, s: c + self.bb_std * s, self.bb_mid, self.bb_stddev)
        self.bb_lower = self.I(lambda c, s: c - self.bb_std * s, self.bb_mid, self.bb_stddev)

        # 🌙 Bollinger Bandwidth
        def bbw_calc(upper, lower, mid):
            upper = np.asarray(upper, dtype=np.float64)
            lower = np.asarray(lower, dtype=np.float64)
            mid = np.asarray(mid, dtype=np.float64)
            mid = np.where(mid == 0, np.nan, mid)
            return (upper - lower) / mid
        self.bbw = self.I(bbw_calc, self.bb_upper, self.bb_lower, self.bb_mid)

        # 🌙 BBW percentile rank over lookback
        def bbw_pct(arr):
            s = pd.Series(np.asarray(arr, dtype=np.float64))
            return s.rolling(self.bbw_lookback).apply(
                lambda x: (x.iloc[-1] <= x).mean(), raw=False
            ).values
        self.bbw_pct = self.I(bbw_pct, self.bbw)

        # 🌙 Volume SMA
        self.vol_sma = self.I(talib.SMA, self.data.Volume.astype(np.float64), timeperiod=self.vol_sma_period)

        # 🌙 ATR
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period)

        # 🌙 Volume-Weighted RSI
        self.vwrsi = self.I(vw_rsi, self.data.Close, self.data.Volume, period=14, vol_lookback=20)

        # 🌙 Swing lows / highs
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=self.swing_window)
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=self.swing_window)

        self.entry_bar = None
        self.stop_price = None
        self.tp_price = None

    def next(self):
        i = len(self.data) - 1
        if i < max(self.bbw_lookback, self.vol_sma_period, self.atr_period, self.div_lookback) + 5:
            return

        price = self.data.Close[-1]
        upper = self.bb_upper[-1]
        lower = self.bb_lower[-1]
        mid = self.bb_mid[-1]
        bbw_pct = self.bbw_pct[-1]
        vol = self.data.Volume[-1]
        vol_avg = self.vol_sma[-1]
        atr = self.atr[-1]
        vwrsi_now = self.vwrsi[-1]

        # 🌙 Manage open position
        if self.position:
            bars_held = i - self.entry_bar

            # Time stop
            if bars_held >= self.time_stop_bars:
                print(f"🌙⏰ Time stop hit after {bars_held} bars — exiting")
                self.position.close()
                return

            if self.position.is_long:
                # Exit at middle band
                if price >= mid:
                    print(f"🌙✨ Long exit at middle band @ {price:.2f}")
                    self.position.close()
                    return
                # VW-RSI cross below 50
                if self.vwrsi[-2] >= 50 and vwrsi_now < 50:
                    print(f"🌙📉 VW-RSI cross below 50 — closing long")
                    self.position.close()
                    return
                # Stop loss
                if self.stop_price and price <= self.stop_price:
                    print(f"🌙🛑 Long stop hit @ {price:.2f}")
                    self.position.close()
                    return

            if self.position.is_short:
                if price <= mid:
                    print(f"🌙✨ Short exit at middle band @ {price:.2f}")
                    self.position.close()
                    return
                if self.vwrsi[-2] <= 50 and vwrsi_now > 50:
                    print(f"🌙📈 VW-RSI cross above 50 — closing short")
                    self.position.close()
                    return
                if self.stop_price and price >= self.stop_price:
                    print(f"🌙🛑 Short stop hit @ {price:.2f}")
                    self.position.close()
                    return
            return

        # 🌙 Entry conditions
        if np.isnan(bbw_pct) or bbw_pct > self.bbw_pct_threshold:
            return
        if np.isnan(vol_avg) or vol < self.vol_mult * vol_avg:
            return

        # 🌙 Bullish divergence: price lower low, VW-RSI higher low
        lookback = self.div_lookback
        recent_low_idx = i - 1
        prior_low_idx = None
        for j in range(i - lookback, i - 2):
            if j < 1:
                continue
            if self.data.Low[j] == self.swing_low[j] and self.data.Low[j] < self.data.Low[recent_low_idx]:
                prior_low_idx = j
                break

        bullish_div = False
        if prior_low_idx is not None:
            if self.data.Low[recent_low_idx] < self.data.Low[prior_low_idx] and \
               self.vwrsi[recent_low_idx] > self.vwrsi[prior_low_idx]:
                bullish_div = True

        # 🌙 Bearish divergence: price higher high, VW-RSI lower high
        recent_high_idx = i - 1
        prior_high_idx = None
        for j in range(i - lookback, i - 2):
            if j < 1:
                continue
            if self.data.High[j] == self.swing_high[j] and self.data.High[j] > self.data.High[recent_high_idx]:
                prior_high_idx = j
                break

        bearish_div = False
        if prior_high_idx is not None:
            if self.data.High[recent_high_idx] > self.data.High[prior_high_idx] and \
               self.vwrsi[recent_high_idx] < self.vwrsi[prior_high_idx]:
                bearish_div = True

        # 🌙 Long entry
        if bullish_div and price > upper:
            stop = min(lower, self.swing_low[-1]) - 0.2 * atr
            risk = price - stop
            if risk <= 0:
                return
            size = 0.95  # 🌙 fraction of equity (valid sizing)
            print(f"🌙🚀 LONG ENTRY @ {price:.2f} | BBW%={bbw_pct:.2f} | Stop={stop:.2f} | Size={size}")
            self.buy(size=size)
            self.entry_bar = i
            self.stop_price = stop
            return

        # 🌙 Short entry
        if bearish_div and price < lower:
            stop = max(upper, self.swing_high[-1]) + 0.2 * atr
            risk = stop - price
            if risk <= 0:
                return
            size = 0.95  # 🌙 fraction of equity (valid sizing)
            print(f"🌙🔻 SHORT ENTRY @ {price:.2f} | BBW%={bbw_pct:.2f} | Stop={stop:.2f} | Size={size}")
            self.sell(size=size)
            self.entry_bar = i
            self.stop_price = stop
            return


# 🌙 Load data
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'datetime': 'Date',
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume',
})
data['Date'] = pd.to_datetime(data['Date'])
data = data.set_index('Date')

# 🌙 Ensure numeric dtypes for talib (must be double/float64)
for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    data[col] = pd.to_numeric(data[col], errors='coerce').astype(np.float64)

data = data.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])

print("🌙 Moon Dev's Volumetric SqueezeDivergence Backtest 🚀✨")
print(f"📊 Data loaded: {len(data)} bars")

bt = Backtest(data, VolumetricSqueezeDivergence, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)