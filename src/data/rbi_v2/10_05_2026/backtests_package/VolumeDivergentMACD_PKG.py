import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

print("🌙 Moon Dev's VolumeDivergentMACD Backtest Loading... ✨🚀")

data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
print(f"🌙 Data loaded: {len(data)} bars ✨")


class VolumeDivergentMACD(Strategy):
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    vol_ma_period = 20
    bb_period = 20
    bb_std = 2.0
    atr_period = 14
    pivot_window = 5
    lookback = 30
    min_pivot_gap = 5
    max_pivot_gap = 20
    vol_decline_pct = 0.10
    risk_pct = 0.02
    max_hold = 40
    sd_expansion_mult = 2.0

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        self.macd, self.macd_signal, self.macd_hist = self.I(
            talib.MACD, close,
            fastperiod=self.macd_fast,
            slowperiod=self.macd_slow,
            signalperiod=self.macd_signal
        )
        self.vol_ma = self.I(talib.SMA, volume, timeperiod=self.vol_ma_period)
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS, close,
            timeperiod=self.bb_period,
            nbdevup=self.bb_std,
            nbdevdn=self.bb_std
        )
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        # Bollinger Band width and its rolling mean/std for expansion detection
        bb_upper = np.asarray(self.bb_upper)
        bb_lower = np.asarray(self.bb_lower)
        bb_middle = np.asarray(self.bb_middle)
        bb_width = (bb_upper - bb_lower) / np.where(bb_middle == 0, 1, bb_middle)
        self.bb_width = self.I(lambda: bb_width)
        self.bb_width_ma = self.I(talib.SMA, bb_width, timeperiod=self.bb_period)
        self.bb_width_std = self.I(talib.STDDEV, bb_width, timeperiod=self.bb_period)

        # Pivot detection (rolling max/min)
        self.swing_high = self.I(talib.MAX, high, timeperiod=self.pivot_window * 2 + 1)
        self.swing_low = self.I(talib.MIN, low, timeperiod=self.pivot_window * 2 + 1)

        self.entry_bar = None
        self.stop_price = None
        self.tp_price = None
        self.trade_direction = None

        print("🌙 Indicators initialized! MACD, BB, ATR, Volume MA ready ✨")

    def _find_pivots_low(self, i):
        """Find two recent swing lows for bullish divergence."""
        pivots = []
        for k in range(i - self.lookback, i - self.min_pivot_gap):
            if k < self.pivot_window:
                continue
            if self.data.Low[k] == self.swing_low[k] and self.data.Low[k] < self.data.Low[k - 1] and self.data.Low[k] < self.data.Low[k + 1]:
                pivots.append(k)
        return pivots

    def _find_pivots_high(self, i):
        pivots = []
        for k in range(i - self.lookback, i - self.min_pivot_gap):
            if k < self.pivot_window:
                continue
            if self.data.High[k] == self.swing_high[k] and self.data.High[k] > self.data.High[k - 1] and self.data.High[k] > self.data.High[k + 1]:
                pivots.append(k)
        return pivots

    def _bullish_divergence(self, i):
        pivots = self._find_pivots_low(i)
        if len(pivots) < 2:
            return False
        p2 = pivots[-1]
        for p1 in pivots[:-1]:
            gap = p2 - p1
            if gap < self.min_pivot_gap or gap > self.max_pivot_gap:
                continue
            # price lower low
            if self.data.Low[p2] < self.data.Low[p1]:
                # macd higher low
                if self.macd[p2] > self.macd[p1]:
                    # volume declining
                    v1 = self.data.Volume[p1]
                    v2 = self.data.Volume[p2]
                    if v1 > 0 and v2 < v1 * (1 - self.vol_decline_pct):
                        return True
        return False

    def _bearish_divergence(self, i):
        pivots = self._find_pivots_high(i)
        if len(pivots) < 2:
            return False
        p2 = pivots[-1]
        for p1 in pivots[:-1]:
            gap = p2 - p1
            if gap < self.min_pivot_gap or gap > self.max_pivot_gap:
                continue
            if self.data.High[p2] > self.data.High[p1]:
                if self.macd[p2] < self.macd[p1]:
                    v1 = self.data.Volume[p1]
                    v2 = self.data.Volume[p2]
                    if v1 > 0 and v2 < v1 * (1 - self.vol_decline_pct):
                        return True
        return False

    def next(self):
        i = len(self.data) - 1
        if i < max(self.lookback + self.pivot_window, self.bb_period + 5, self.macd_slow + self.macd_signal):
            return

        price = self.data.Close[-1]

        # Manage open position
        if self.position:
            bars_held = i - self.entry_bar if self.entry_bar else 0

            # Stop loss / take profit
            if self.trade_direction == 'long':
                if self.data.Low[-1] <= self.stop_price:
                    print(f"🛑 Long stop-loss hit at {self.stop_price:.2f} 🌙")
                    self.position.close()
                    return
                if self.tp_price and self.data.High[-1] >= self.tp_price:
                    print(f"🎯 Long take-profit hit at {self.tp_price:.2f} ✨")
                    self.position.close()
                    return
            else:
                if self.data.High[-1] >= self.stop_price:
                    print(f"🛑 Short stop-loss hit at {self.stop_price:.2f} 🌙")
                    self.position.close()
                    return
                if self.tp_price and self.data.Low[-1] <= self.tp_price:
                    print(f"🎯 Short take-profit hit at {self.tp_price:.2f} ✨")
                    self.position.close()
                    return

            # BB SD expansion exit
            if not np.isnan(self.bb_width_std[-1]) and not np.isnan(self.bb_width_ma[-1]):
                threshold = self.bb_width_ma[-1] + self.sd_expansion_mult * self.bb_width_std[-1]
                if self.bb_width[-1] > threshold:
                    print(f"🌊 BB volatility expansion exit! width={self.bb_width[-1]:.4f} > {threshold:.4f} 🚀")
                    self.position.close()
                    return

            # Max hold
            if bars_held >= self.max_hold:
                print(f"⏰ Max hold period ({self.max_hold}) reached, closing 🕐")
                self.position.close()
                return
            return

        # Entry logic
        hist = self.macd_hist[-1]
        hist_prev = self.macd_hist[-2]

        # Long entry: bullish divergence + hist turning positive
        if self._bullish_divergence(i) and hist_prev <= 0 and hist > 0:
            stop = price - 1.5 * self.atr[-1]
            tp = self.bb_middle[-1] if self.bb_middle[-1] > price else price + 2 * self.atr[-1]
            self.stop_price = stop
            self.tp_price = tp
            self.entry_bar = i
            self.trade_direction = 'long'
            print(f"🌟 LONG ENTRY @ {price:.2f} | Bullish Divergence + Volume Decline | SL={stop:.2f} TP={tp:.2f} 🌙🚀")
            self.buy(size=1000000)
            return

        # Short entry: bearish divergence + hist turning negative
        if self._bearish_divergence(i) and hist_prev >= 0 and hist < 0:
            stop = price + 1.5 * self.atr[-1]
            tp = self.bb_middle[-1] if self.bb_middle[-1] < price else price - 2 * self.atr[-1]
            self.stop_price = stop
            self.tp_price = tp
            self.entry_bar = i
            self.trade_direction = 'short'
            print(f"🌟 SHORT ENTRY @ {price:.2f} | Bearish Divergence + Volume Decline | SL={stop:.2f} TP={tp:.2f} 🌙🚀")
            self.sell(size=1000000)
            return


bt = Backtest(data, VolumeDivergentMACD, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)