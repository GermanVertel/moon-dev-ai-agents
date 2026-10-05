import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# 🌙 Moon Dev's CompressionIgnition Backtest 🚀

data_path = '/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv'

data = pd.read_csv(data_path)
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
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
print(f"🌙 Moon Dev data loaded: {len(data)} bars ✨")


class CompressionIgnition(Strategy):
    bb_period = 20
    bb_std = 2.0
    squeeze_lookback = 30
    vol_period = 20
    vol_mult = 2.0
    atr_period = 14
    risk_pct = 0.01
    time_stop = 15

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        vol = self.data.Volume

        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS, close, timeperiod=self.bb_period,
            nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0
        )
        self.bbw = self.I(
            lambda u, l, m: (u - l) / m,
            self.bb_upper, self.bb_lower, self.bb_middle
        )
        self.bbw_min = self.I(
            lambda x: pd.Series(x).rolling(self.squeeze_lookback).min().values,
            self.bbw
        )
        self.vol_sma = self.I(talib.SMA, vol, timeperiod=self.vol_period)
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        self.armed_long = False
        self.armed_short = False
        self.entry_bar = None
        self.entry_price = None
        self.stop_price = None
        self.target_price = None
        self.initial_band_width = None
        print("🌙✨ CompressionIgnition indicators initialized 🚀")

    def next(self):
        if len(self.data) < self.squeeze_lookback + 5:
            return

        price = self.data.Close[-1]
        upper = self.bb_upper[-1]
        lower = self.bb_lower[-1]
        middle = self.bb_middle[-1]
        bbw = self.bbw[-1]
        bbw_min = self.bbw_min[-1]
        vol = self.data.Volume[-1]
        vol_avg = self.vol_sma[-1]
        atr = self.atr[-1]

        if np.isnan(bbw_min) or np.isnan(vol_avg) or np.isnan(atr) or atr <= 0:
            return

        # 🎯 Manage open position
        if self.position:
            bars_held = len(self.data) - self.entry_bar
            is_long = self.position.is_long

            # Exit: price closes back inside BB
            if is_long and price < upper:
                print(f"🌙💥 LONG exit: price closed back inside BB @ {price:.2f}")
                self.position.close()
                self._reset()
                return
            if not is_long and price > lower:
                print(f"🌙💥 SHORT exit: price closed back inside BB @ {price:.2f}")
                self.position.close()
                self._reset()
                return

            # Trailing stop at middle band
            if is_long and price < middle:
                print(f"🌙🛑 LONG trailing stop (middle band) @ {price:.2f}")
                self.position.close()
                self._reset()
                return
            if not is_long and price > middle:
                print(f"🌙🛑 SHORT trailing stop (middle band) @ {price:.2f}")
                self.position.close()
                self._reset()
                return

            # Hard stop
            if is_long and price <= self.stop_price:
                print(f"🌙🛑 LONG hard stop @ {price:.2f}")
                self.position.close()
                self._reset()
                return
            if not is_long and price >= self.stop_price:
                print(f"🌙🛑 SHORT hard stop @ {price:.2f}")
                self.position.close()
                self._reset()
                return

            # Profit target
            if is_long and price >= self.target_price:
                print(f"🌙🎯 LONG profit target hit @ {price:.2f}")
                self.position.close()
                self._reset()
                return
            if not is_long and price <= self.target_price:
                print(f"🌙🎯 SHORT profit target hit @ {price:.2f}")
                self.position.close()
                self._reset()
                return

            # Time stop
            if bars_held >= self.time_stop:
                print(f"🌙⏰ Time stop hit after {bars_held} bars @ {price:.2f}")
                self.position.close()
                self._reset()
                return
            return

        # 🔍 Detect squeeze (compression)
        in_squeeze = bbw <= bbw_min
        if in_squeeze:
            if not (self.armed_long or self.armed_short):
                print(f"🌙🔒 SQUEEZE detected! BBW={bbw:.5f} <= 30-day min={bbw_min:.5f}")
            self.armed_long = True
            self.armed_short = True

        if not (self.armed_long or self.armed_short):
            return

        # ⚡ Breakout + volume confirmation
        vol_confirmed = vol >= self.vol_mult * vol_avg

        if self.armed_long and price > upper and vol_confirmed:
            self._enter_long(price, upper, lower, middle, atr)
        elif self.armed_short and price < lower and vol_confirmed:
            self._enter_short(price, upper, lower, middle, atr)

    def _enter_long(self, price, upper, lower, middle, atr):
        stop_band = lower
        stop_atr = price - 1.5 * atr
        stop = max(stop_band, stop_atr)
        risk = price - stop
        if risk <= 0:
            return
        size = int(round((self.equity * self.risk_pct) / risk))
        if size <= 0:
            return
        self.initial_band_width = upper - lower
        self.target_price = price + 2 * self.initial_band_width
        self.stop_price = stop
        self.entry_bar = len(self.data)
        self.entry_price = price
        self.buy(size=size)
        self.armed_long = False
        self.armed_short = False
        print(f"🚀🌙 LONG ENTRY @ {price:.2f} | size={size} | stop={stop:.2f} | target={self.target_price:.2f}")

    def _enter_short(self, price, upper, lower, middle, atr):
        stop_band = upper
        stop_atr = price + 1.5 * atr
        stop = min(stop_band, stop_atr)
        risk = stop - price
        if risk <= 0:
            return
        size = int(round((self.equity * self.risk_pct) / risk))
        if size <= 0:
            return
        self.initial_band_width = upper - lower
        self.target_price = price - 2 * self.initial_band_width
        self.stop_price = stop
        self.entry_bar = len(self.data)
        self.entry_price = price
        self.sell(size=size)
        self.armed_long = False
        self.armed_short = False
        print(f"🔻🌙 SHORT ENTRY @ {price:.2f} | size={size} | stop={stop:.2f} | target={self.target_price:.2f}")

    def _reset(self):
        self.armed_long = False
        self.armed_short = False
        self.entry_bar = None
        self.entry_price = None
        self.stop_price = None
        self.target_price = None
        self.initial_band_width = None


bt = Backtest(data, CompressionIgnition, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)