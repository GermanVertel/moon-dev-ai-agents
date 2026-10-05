import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

print("🌙✨ SqueezePulse Backtest Initializing... 🚀")

# Load and clean data
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data.columns = [c.capitalize() for c in data.columns]
data['Datetime'] = pd.to_datetime(data['Datetime'])
data = data.set_index('Datetime')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
print(f"🌙 Data loaded: {len(data)} bars ✨")


class SqueezePulse(Strategy):
    ema_fast = 50
    ema_slow = 200
    bb_period = 20
    bb_std = 2.0
    vol_period = 20
    vol_mult = 2.0
    squeeze_lookback = 100
    squeeze_pct = 0.20
    atr_period = 14
    atr_mult = 1.5
    rr_ratio = 2.0
    time_stop = 15
    risk_pct = 0.02

    def init(self):
        print("🌙 Setting up SqueezePulse indicators... ✨")
        self.ema_fast_i = self.I(talib.EMA, self.data.Close, timeperiod=self.ema_fast)
        self.ema_slow_i = self.I(talib.EMA, self.data.Close, timeperiod=self.ema_slow)

        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS, self.data.Close,
            timeperiod=self.bb_period, nbdevup=self.bb_std, nbdevdn=self.bb_std
        )

        self.band_width = self.I(
            lambda u, m, l: (u - l) / np.where(m == 0, np.nan, m),
            self.bb_upper, self.bb_middle, self.bb_lower
        )

        self.vol_sma = self.I(talib.SMA, self.data.Volume, timeperiod=self.vol_period)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period)

        self.squeeze_thresh = self.I(
            lambda bw: pd.Series(bw).rolling(self.squeeze_lookback).quantile(self.squeeze_pct).values,
            self.band_width
        )

        self.bar_range = self.I(lambda h, l: h - l, self.data.High, self.data.Low)
        self.avg_range = self.I(talib.SMA, self.bar_range, timeperiod=20)

        self.entry_bar = None
        self.entry_price = None
        self.stop_price = None
        self.tp_price = None
        print("🚀 Indicators ready!")

    def next(self):
        price = self.data.Close[-1]

        # Manage open position
        if self.position:
            bars_held = len(self.data) - 1 - self.entry_bar

            if self.position.is_long:
                if self.data.Low[-1] <= self.stop_price:
                    print(f"🛑 LONG Stop hit @ {self.stop_price:.2f} | price={price:.2f} 🌙")
                    self.position.close()
                    self._reset()
                    return
                if self.data.High[-1] >= self.tp_price:
                    print(f"🎯 LONG TP hit @ {self.tp_price:.2f} | price={price:.2f} ✨")
                    self.position.close()
                    self._reset()
                    return
                if price < self.bb_middle[-1]:
                    print(f"📉 LONG exit: close back inside BB @ {price:.2f} 🌙")
                    self.position.close()
                    self._reset()
                    return
                if self.ema_fast_i[-1] < self.ema_slow_i[-1]:
                    print(f"⚠️ LONG exit: EMA cross reversed @ {price:.2f} 🚀")
                    self.position.close()
                    self._reset()
                    return
                if bars_held >= self.time_stop:
                    print(f"⏰ LONG time stop @ {price:.2f} after {bars_held} bars 🌙")
                    self.position.close()
                    self._reset()
                    return

            elif self.position.is_short:
                if self.data.High[-1] >= self.stop_price:
                    print(f"🛑 SHORT Stop hit @ {self.stop_price:.2f} | price={price:.2f} 🌙")
                    self.position.close()
                    self._reset()
                    return
                if self.data.Low[-1] <= self.tp_price:
                    print(f"🎯 SHORT TP hit @ {self.tp_price:.2f} | price={price:.2f} ✨")
                    self.position.close()
                    self._reset()
                    return
                if price > self.bb_middle[-1]:
                    print(f"📈 SHORT exit: close back inside BB @ {price:.2f} 🌙")
                    self.position.close()
                    self._reset()
                    return
                if self.ema_fast_i[-1] > self.ema_slow_i[-1]:
                    print(f"⚠️ SHORT exit: EMA cross reversed @ {price:.2f} 🚀")
                    self.position.close()
                    self._reset()
                    return
                if bars_held >= self.time_stop:
                    print(f"⏰ SHORT time stop @ {price:.2f} after {bars_held} bars 🌙")
                    self.position.close()
                    self._reset()
                    return
            return

        # Check for entries
        if len(self.data) < max(self.ema_slow, self.squeeze_lookback) + 5:
            return

        bw = self.band_width[-1]
        sq_thresh = self.squeeze_thresh[-1]
        if np.isnan(bw) or np.isnan(sq_thresh):
            return

        squeeze = bw <= sq_thresh
        vol_ok = self.data.Volume[-1] >= self.vol_mult * self.vol_sma[-1]
        range_exp = self.bar_range[-1] > self.avg_range[-1]

        if not (squeeze and vol_ok and range_exp):
            return

        bullish = self.ema_fast_i[-1] > self.ema_slow_i[-1]
        bearish = self.ema_fast_i[-1] < self.ema_slow_i[-1]

        atr = self.atr[-1]
        if np.isnan(atr) or atr <= 0:
            return

        equity = self.equity
        risk_amount = equity * self.risk_pct

        # LONG
        if bullish and price > self.bb_upper[-1]:
            stop = min(self.bb_middle[-1], price - self.atr_mult * atr)
            risk = price - stop
            if risk <= 0:
                return
            size = int(round(risk_amount / risk))
            if size <= 0:
                return
            tp = price + self.rr_ratio * risk
            print(f"🚀🌙 LONG ENTRY @ {price:.2f} | SL={stop:.2f} TP={tp:.2f} size={size} ✨")
            self.buy(size=size)
            self.entry_bar = len(self.data) - 1
            self.entry_price = price
            self.stop_price = stop
            self.tp_price = tp

        # SHORT
        elif bearish and price < self.bb_lower[-1]:
            stop = max(self.bb_middle[-1], price + self.atr_mult * atr)
            risk = stop - price
            if risk <= 0:
                return
            size = int(round(risk_amount / risk))
            if size <= 0:
                return
            tp = price - self.rr_ratio * risk
            print(f"🚀🌙 SHORT ENTRY @ {price:.2f} | SL={stop:.2f} TP={tp:.2f} size={size} ✨")
            self.sell(size=size)
            self.entry_bar = len(self.data) - 1
            self.entry_price = price
            self.stop_price = stop
            self.tp_price = tp

    def _reset(self):
        self.entry_bar = None
        self.entry_price = None
        self.stop_price = None
        self.tp_price = None


print("🌙 Running SqueezePulse backtest... 🚀✨")
bt = Backtest(data, SqueezePulse, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙✨ SqueezePulse backtest complete! 🚀")