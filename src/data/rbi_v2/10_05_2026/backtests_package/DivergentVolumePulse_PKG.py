import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Load and clean data
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)

print("🌙✨ Moon Dev Backtest System Initialized ✨🌙")
print(f"📊 Data loaded: {len(data)} bars from {data.index[0]} to {data.index[-1]}")


def vwma(values, volume, period):
    """Volume Weighted Moving Average"""
    values = np.asarray(values, dtype=float)
    volume = np.asarray(volume, dtype=float)
    out = np.full(len(values), np.nan)
    for i in range(period - 1, len(values)):
        v = values[i - period + 1:i + 1]
        w = volume[i - period + 1:i + 1]
        if w.sum() > 0:
            out[i] = np.sum(v * w) / np.sum(w)
    return out


def vw_ema(values, volume, period):
    """Volume-weighted EMA approximation: EMA of (price * volume) / EMA of volume"""
    values = np.asarray(values, dtype=float)
    volume = np.asarray(volume, dtype=float)
    pv = values * volume
    ema_pv = talib.EMA(pv, timeperiod=period)
    ema_v = talib.EMA(volume, timeperiod=period)
    with np.errstate(divide='ignore', invalid='ignore'):
        result = ema_pv / ema_v
    return result


class DivergentVolumePulse(Strategy):
    # Parameters
    fast = 12
    slow = 26
    signal = 9
    cmo_period = 14
    bb_period = 20
    bb_std = 2.0
    atr_period = 14
    atr_mult = 1.5
    risk_pct = 0.02
    div_lookback = 10
    div_min_gap = 3
    time_stop = 10
    bbw_pct_lookback = 100
    bbw_pct_floor = 0.10

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        # VW-MACD
        vw_fast = self.I(vw_ema, close, volume, self.fast, name='VW_EMA_Fast')
        vw_slow = self.I(vw_ema, close, volume, self.slow, name='VW_EMA_Slow')
        self.vw_macd = self.I(lambda: vw_fast - vw_slow, name='VW_MACD')
        self.vw_signal = self.I(talib.EMA, self.vw_macd, timeperiod=self.signal, name='VW_MACD_Signal')

        # CMO
        self.cmo = self.I(talib.CMO, close, timeperiod=self.cmo_period, name='CMO')

        # Bollinger Bands
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS, close, timeperiod=self.bb_period, nbdevup=self.bb_std,
            nbdevdn=self.bb_std, matype=0, name='BB'
        )
        self.bbw = self.I(
            lambda: (self.bb_upper - self.bb_lower) / self.bb_middle,
            name='BBW'
        )

        # ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period, name='ATR')

        # Trend filter
        self.vwma50 = self.I(vwma, close, volume, 50, name='VWMA50')
        self.ema200 = self.I(talib.EMA, close, timeperiod=200, name='EMA200')

        # Swing highs/lows for divergence detection
        self.swing_high = self.I(talib.MAX, high, timeperiod=5, name='SwingHigh')
        self.swing_low = self.I(talib.MIN, low, timeperiod=5, name='SwingLow')

        print("🌙 Indicators initialized! Let's ride the volume pulse! 🚀")

    def _bullish_divergence(self, i):
        """Price lower low, CMO higher low within lookback window."""
        if i < self.div_lookback + 5:
            return False
        recent_low_idx = i
        recent_low = self.data.Low[recent_low_idx]
        recent_cmo = self.cmo[recent_low_idx]

        start = max(0, i - self.div_lookback)
        for j in range(start, i - self.div_min_gap + 1):
            if self.data.Low[j] < recent_low and self.cmo[j] < recent_cmo:
                return True
        return False

    def _bearish_divergence(self, i):
        """Price higher high, CMO lower high within lookback window."""
        if i < self.div_lookback + 5:
            return False
        recent_high = self.data.High[i]
        recent_cmo = self.cmo[i]

        start = max(0, i - self.div_lookback)
        for j in range(start, i - self.div_min_gap + 1):
            if self.data.High[j] > recent_high and self.cmo[j] > recent_cmo:
                return True
        return False

    def _bbw_squeeze_exit(self, i):
        """BBW contracted 2 consecutive periods AND price in middle 50% of bands."""
        if i < 3:
            return False
        bbw_now = self.bbw[i]
        bbw_prev = self.bbw[i - 1]
        bbw_prev2 = self.bbw[i - 2]
        if np.isnan(bbw_now) or np.isnan(bbw_prev) or np.isnan(bbw_prev2):
            return False
        contracted = (bbw_now < bbw_prev) and (bbw_prev < bbw_prev2)
        if not contracted:
            return False
        upper = self.bb_upper[i]
        lower = self.bb_lower[i]
        close = self.data.Close[i]
        if np.isnan(upper) or np.isnan(lower) or upper == lower:
            return False
        pos = (close - lower) / (upper - lower)
        return 0.25 <= pos <= 0.75

    def _bbw_low_percentile(self, i):
        """Skip entries when BBW in bottom 10th percentile of 100-period range."""
        if i < self.bbw_pct_lookback:
            return False
        window = self.bbw[i - self.bbw_pct_lookback + 1:i + 1]
        window = window[~np.isnan(window)]
        if len(window) < 20:
            return False
        return self.bbw[i] <= np.percentile(window, self.bbw_pct_floor * 100)

    def next(self):
        i = len(self.data) - 1
        if i < 210:
            return

        price = self.data.Close[-1]
        macd_now = self.vw_macd[-1]
        macd_prev = self.vw_macd[-2]
        cmo_now = self.cmo[-1]
        cmo_prev = self.cmo[-2]
        atr_now = self.atr[-1]

        if np.isnan(macd_now) or np.isnan(macd_prev) or np.isnan(cmo_now) or np.isnan(cmo_prev) or np.isnan(atr_now):
            return

        # ===== EXIT LOGIC =====
        if self.position:
            if self._bbw_squeeze_exit(i):
                print(f"🌙 BBW squeeze exit at {price:.2f} — volatility compressing! 💤")
                self.position.close()
                return

            if len(self.trades) > 0 and self.trades[-1].size != 0:
                bars_held = i - self.trades[-1].entry_bar
                if bars_held >= self.time_stop:
                    print(f"⏰ Time stop hit at {price:.2f} after {bars_held} bars")
                    self.position.close()
                    return

        # ===== ENTRY LOGIC =====
        if self.position:
            return

        if self._bbw_low_percentile(i):
            return

        # Long entry
        vw_macd_cross_up = (macd_prev <= 0) and (macd_now > 0)
        cmo_rising = cmo_now > cmo_prev
        if vw_macd_cross_up and cmo_rising and self._bullish_divergence(i):
            trend_ok = price > self.ema200[-1] if not np.isnan(self.ema200[-1]) else True
            if trend_ok:
                stop = price - self.atr_mult * atr_now
                risk = price - stop
                if risk > 0:
                    risk_amount = self.equity * self.risk_pct
                    size = int(round(risk_amount / risk))
                    if size > 0:
                        print(f"🚀🌙 LONG SIGNAL! Price={price:.2f} VW-MACD cross up, CMO div bullish, size={size}")
                        self.buy(size=size, sl=stop, tp=price + 2 * risk)
                        return

        # Short entry
        vw_macd_cross_dn = (macd_prev >= 0) and (macd_now < 0)
        cmo_falling = cmo_now < cmo_prev
        if vw_macd_cross_dn and cmo_falling and self._bearish_divergence(i):
            trend_ok = price < self.ema200[-1] if not np.isnan(self.ema200[-1]) else True
            if trend_ok:
                stop = price + self.atr_mult * atr_now
                risk = stop - price
                if risk > 0:
                    risk_amount = self.equity * self.risk_pct
                    size = int(round(risk_amount / risk))
                    if size > 0:
                        print(f"🔻🌙 SHORT SIGNAL! Price={price:.2f} VW-MACD cross down, CMO div bearish, size={size}")
                        self.sell(size=size, sl=stop, tp=price - 2 * risk)
                        return


print("🌙✨ Starting DivergentVolumePulse Backtest ✨🌙")
bt = Backtest(data, DivergentVolumePulse, cash=1_000_000, commission=0.001, exclusive_orders=True)
stats = bt.run()
print(stats)
print(stats._strategy)