import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# 🌙 Moon Dev's VolatilityDivergence Backtest 🚀

def load_data():
    data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')
    data.columns = data.columns.str.strip().str.lower()
    data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
    data = data.rename(columns={
        'datetime': 'Date',
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    })
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.set_index('Date')
    return data


class VolatilityDivergence(Strategy):
    rsi_period = 14
    atr_period = 14
    vol_ma_period = 20
    bb_period = 20
    bb_std = 2
    swing_window = 5
    risk_pct = 0.01
    time_exit_bars = 12

    def init(self):
        print("🌙✨ Initializing VolatilityDivergence indicators...")
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=self.rsi_period)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period)
        self.vol_ma = self.I(talib.SMA, self.data.Volume, timeperiod=self.vol_ma_period)
        self.bb_upper, self.bb_mid, self.bb_lower = self.I(
            talib.BBANDS, self.data.Close, timeperiod=self.bb_period,
            nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0
        )
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=self.swing_window * 2 + 1)
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=self.swing_window * 2 + 1)
        self.entry_bar = None
        self.stop_price = None
        self.tp_price = None
        print("🌙✨ Indicators ready! Let's ride the volatility waves! 🚀")

    def _bullish_divergence(self, i):
        if i < self.swing_window * 2 + 2:
            return False
        # current swing low vs prior swing low
        c_low = self.data.Low[i]
        p_low = self.data.Low[i - self.swing_window]
        c_rsi = self.rsi[i]
        p_rsi = self.rsi[i - self.swing_window]
        # price lower low, rsi higher low
        if c_low < p_low and c_rsi > p_rsi and c_rsi < 45:
            return True
        return False

    def _bearish_divergence(self, i):
        if i < self.swing_window * 2 + 2:
            return False
        c_high = self.data.High[i]
        p_high = self.data.High[i - self.swing_window]
        c_rsi = self.rsi[i]
        p_rsi = self.rsi[i - self.swing_window]
        if c_high > p_high and c_rsi < p_rsi and c_rsi > 55:
            return True
        return False

    def _volume_capitulation(self, i):
        if i < self.vol_ma_period + 3:
            return False
        v = self.data.Volume[i]
        vma = self.vol_ma[i]
        # spike then dry up: recent volume > 1.5x ma and current < previous
        prev_v = self.data.Volume[i - 1]
        if vma > 0 and prev_v > 1.5 * vma and v < prev_v:
            return True
        return False

    def _volume_climax(self, i):
        if i < self.vol_ma_period + 3:
            return False
        v = self.data.Volume[i]
        vma = self.vol_ma[i]
        prev_v = self.data.Volume[i - 1]
        if vma > 0 and prev_v > 1.5 * vma and v < prev_v:
            return True
        return False

    def _atr_ok(self, i):
        if i < 50:
            return True
        atr_hist = self.atr[-50:]
        atr_hist = atr_hist[~np.isnan(atr_hist)]
        if len(atr_hist) < 10:
            return True
        top5 = np.percentile(atr_hist, 95)
        # avoid extreme chaos unless volume capitulation
        if self.atr[i] > top5 and not self._volume_capitulation(i):
            return False
        return True

    def next(self):
        i = len(self.data) - 1
        if i < self.swing_window * 2 + 5:
            return

        price = self.data.Close[i]
        atr = self.atr[i]
        if np.isnan(atr) or atr <= 0:
            return

        # ---------- MANAGE OPEN POSITION ----------
        if self.position:
            bars_held = i - self.entry_bar if self.entry_bar is not None else 0
            # time-based exit
            if bars_held >= self.time_exit_bars:
                print(f"⏰🌙 Time-based exit after {bars_held} bars. Closing position.")
                self.position.close()
                self.entry_bar = None
                return

            if self.position.is_long:
                # trailing stop: once 1x ATR in favor, trail at 2x ATR
                if price > self.position.entry_price + atr:
                    new_stop = price - 2 * atr
                    if new_stop > self.stop_price:
                        self.stop_price = new_stop
                        print(f"📈🌙 Trailing long stop to {self.stop_price:.2f}")
                # take profit
                if price >= self.tp_price:
                    print(f"🎯🚀 Long TP hit at {price:.2f}")
                    self.position.close()
                    self.entry_bar = None
                    return
                # stop loss
                if price <= self.stop_price:
                    print(f"🛑🌙 Long SL hit at {price:.2f}")
                    self.position.close()
                    self.entry_bar = None
                    return
                # RSI overbought
                if self.rsi[i] >= 70:
                    print(f"💫🌙 Long exit: RSI overbought at {self.rsi[i]:.1f}")
                    self.position.close()
                    self.entry_bar = None
                    return

            elif self.position.is_short:
                if price < self.position.entry_price - atr:
                    new_stop = price + 2 * atr
                    if new_stop < self.stop_price:
                        self.stop_price = new_stop
                        print(f"📉🌙 Trailing short stop to {self.stop_price:.2f}")
                if price <= self.tp_price:
                    print(f"🎯🚀 Short TP hit at {price:.2f}")
                    self.position.close()
                    self.entry_bar = None
                    return
                if price >= self.stop_price:
                    print(f"🛑🌙 Short SL hit at {price:.2f}")
                    self.position.close()
                    self.entry_bar = None
                    return
                if self.rsi[i] <= 30:
                    print(f"💫🌙 Short exit: RSI oversold at {self.rsi[i]:.1f}")
                    self.position.close()
                    self.entry_bar = None
                    return
            return

        # ---------- ENTRY LOGIC ----------
        if not self._atr_ok(i):
            return

        # Bullish reversal
        if self._bullish_divergence(i) and self._volume_capitulation(i):
            # confirmation: close breaks above prior swing high
            prior_swing_high = self.swing_high[i - 1]
            if price > prior_swing_high:
                stop = self.data.Low[i - self.swing_window] - 0.5 * atr
                risk = price - stop
                if risk <= 0:
                    return
                tp = price + 2 * atr
                size = int(round((self.equity * self.risk_pct) / risk))
                if size <= 0:
                    return
                print(f"🌙🚀 BULLISH DIVERGENCE! Long entry at {price:.2f} | SL {stop:.2f} | TP {tp:.2f} | size {size}")
                self.buy(size=size)
                self.entry_bar = i
                self.stop_price = stop
                self.tp_price = tp

        # Bearish reversal
        elif self._bearish_divergence(i) and self._volume_climax(i):
            prior_swing_low = self.swing_low[i - 1]
            if price < prior_swing_low:
                stop = self.data.High[i - self.swing_window] + 0.5 * atr
                risk = stop - price
                if risk <= 0:
                    return
                tp = price - 2 * atr
                size = int(round((self.equity * self.risk_pct) / risk))
                if size <= 0:
                    return
                print(f"🌙🚀 BEARISH DIVERGENCE! Short entry at {price:.2f} | SL {stop:.2f} | TP {tp:.2f} | size {size}")
                self.sell(size=size)
                self.entry_bar = i
                self.stop_price = stop
                self.tp_price = tp


if __name__ == '__main__':
    print("🌙✨ Moon Dev's VolatilityDivergence Backtest Starting... 🚀")
    data = load_data()
    print(f"🌙 Data loaded: {len(data)} bars")
    bt = Backtest(data, VolatilityDivergence, cash=1_000_000, commission=0.001)
    stats = bt.run()
    print(stats)
    print(stats._strategy)