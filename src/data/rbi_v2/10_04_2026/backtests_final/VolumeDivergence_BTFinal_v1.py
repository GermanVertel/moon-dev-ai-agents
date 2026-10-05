import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# 🌙 Moon Dev Data Loading & Cleaning
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'datetime': 'Datetime',
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})
data['Datetime'] = pd.to_datetime(data['Datetime'])
data = data.set_index('Datetime')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
print("🌙✨ Moon Dev data loaded:", data.shape)


class VolumeDivergence(Strategy):
    # Parameters
    vwap_window = 20
    vol_ma_window = 20
    skew_window = 10
    atr_window = 14
    ema_window = 20
    swing_window = 10
    divergence_threshold = 2.0
    volume_spike_mult = 1.5
    risk_pct = 0.01
    rr_ratio = 2.0
    max_bars_hold = 10

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        # 🌙 VWAP (rolling)
        def rolling_vwap(h, l, c, v, n):
            tp = (h + l + c) / 3
            tp_v = pd.Series(tp * v)
            v_s = pd.Series(v)
            num = tp_v.rolling(n).sum().values
            den = v_s.rolling(n).sum().values
            den = np.where(den == 0, np.nan, den)
            return num / den

        self.vwap = self.I(rolling_vwap, high, low, close, volume, self.vwap_window, name='VWAP')

        # 🌙 VWAP std dev bands
        def vwap_std(h, l, c, v, n):
            tp = (h + l + c) / 3
            tp_v = pd.Series(tp * v)
            v_s = pd.Series(v)
            num = tp_v.rolling(n).sum().values
            den = v_s.rolling(n).sum().values
            den = np.where(den == 0, np.nan, den)
            vwap_vals = num / den
            diff = tp - vwap_vals
            return pd.Series(diff).rolling(n).std().values

        self.vwap_std = self.I(vwap_std, high, low, close, volume, self.vwap_window, name='VWAP_STD')

        # 🌙 Volume MA
        self.vol_ma = self.I(talib.SMA, volume, timeperiod=self.vol_ma_window, name='VOL_MA')

        # 🌙 ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_window, name='ATR')

        # 🌙 EMA trend filter
        self.ema = self.I(talib.EMA, close, timeperiod=self.ema_window, name='EMA')

        # 🌙 Volume skew: (up volume - down volume) / total volume over N bars
        def calc_skew(c, v, n):
            c = np.asarray(c, dtype=float)
            v = np.asarray(v, dtype=float)
            up = np.where(c > np.roll(c, 1), v, 0.0)
            down = np.where(c < np.roll(c, 1), v, 0.0)
            up[0] = 0.0
            down[0] = 0.0
            up_s = pd.Series(up).rolling(n).sum().values
            down_s = pd.Series(down).rolling(n).sum().values
            total = up_s + down_s
            total = np.where(total == 0, np.nan, total)
            return (up_s - down_s) / total

        self.skew = self.I(calc_skew, close, volume, self.skew_window, name='SKEW')

        # 🌙 Swing high/low for stops
        self.swing_low = self.I(talib.MIN, low, timeperiod=self.swing_window, name='SWING_LOW')
        self.swing_high = self.I(talib.MAX, high, timeperiod=self.swing_window, name='SWING_HIGH')

        self.entry_bar = None
        self.entry_price = None
        self.stop_price = None
        self.target_price = None

    def next(self):
        # Need at least 2 bars for prev_high/prev_low
        if len(self.data) < 2:
            return

        price = self.data.Close[-1]
        vwap = self.vwap[-1]
        vstd = self.vwap_std[-1]
        vol = self.data.Volume[-1]
        vma = self.vol_ma[-1]
        skew = self.skew[-1]
        atr = self.atr[-1]
        ema = self.ema[-1]
        swing_low = self.swing_low[-1]
        swing_high = self.swing_high[-1]
        prev_high = self.data.High[-2]
        prev_low = self.data.Low[-2]

        if (np.isnan(vwap) or np.isnan(vstd) or np.isnan(vma) or np.isnan(skew)
                or np.isnan(atr) or np.isnan(ema)
                or np.isnan(swing_low) or np.isnan(swing_high)
                or vstd == 0 or price <= 0):
            return

        divergence = (price - vwap) / vstd
        vol_spike = vol > self.volume_spike_mult * vma
        skew_up = skew > 0.1
        skew_down = skew < -0.1

        # 🌙 Manage open position
        if self.position:
            bars_held = len(self.data) - self.entry_bar
            if self.position.is_long:
                if self.data.Low[-1] <= self.stop_price:
                    print(f"🌙💥 LONG STOP @ {self.stop_price:.2f}")
                    self.position.close()
                    self.entry_bar = None
                elif self.data.High[-1] >= self.target_price:
                    print(f"🌙🎯 LONG TARGET @ {self.target_price:.2f}")
                    self.position.close()
                    self.entry_bar = None
                elif bars_held >= self.max_bars_hold:
                    print(f"🌙⏰ LONG TIME EXIT @ {price:.2f}")
                    self.position.close()
                    self.entry_bar = None
            elif self.position.is_short:
                if self.data.High[-1] >= self.stop_price:
                    print(f"🌙💥 SHORT STOP @ {self.stop_price:.2f}")
                    self.position.close()
                    self.entry_bar = None
                elif self.data.Low[-1] <= self.target_price:
                    print(f"🌙🎯 SHORT TARGET @ {self.target_price:.2f}")
                    self.position.close()
                    self.entry_bar = None
                elif bars_held >= self.max_bars_hold:
                    print(f"🌙⏰ SHORT TIME EXIT @ {price:.2f}")
                    self.position.close()
                    self.entry_bar = None
            return

        # 🌙 Long entry
        if (divergence < -self.divergence_threshold and skew_up and vol_spike
                and price > prev_high and price > ema * 0.98):
            stop = min(swing_low, price - 1.5 * atr)
            risk = price - stop
            if risk <= 0:
                return
            target = price + self.rr_ratio * risk
            size = int(round(1000000 / price))
            if size < 1:
                return
            print(f"🌙🚀 LONG ENTRY @ {price:.2f} | div={divergence:.2f} skew={skew:.2f} size={size}")
            self.buy(size=size)
            self.entry_bar = len(self.data)
            self.entry_price = price
            self.stop_price = stop
            self.target_price = target

        # 🌙 Short entry
        elif (divergence > self.divergence_threshold and skew_down and vol_spike
                and price < prev_low and price < ema * 1.02):
            stop = max(swing_high, price + 1.5 * atr)
            risk = stop - price
            if risk <= 0:
                return
            target = price - self.rr_ratio * risk
            size = int(round(1000000 / price))
            if size < 1:
                return
            print(f"🌙🔻 SHORT ENTRY @ {price:.2f} | div={divergence:.2f} skew={skew:.2f} size={size}")
            self.sell(size=size)
            self.entry_bar = len(self.data)
            self.entry_price = price
            self.stop_price = stop
            self.target_price = target


bt = Backtest(data, VolumeDivergence, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)