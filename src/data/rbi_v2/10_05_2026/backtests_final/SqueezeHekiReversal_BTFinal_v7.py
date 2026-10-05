import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

print("🌙✨ SqueezeHeki Reversal Backtest Initializing... 🚀")

DATA_PATH = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"

data = pd.read_csv(DATA_PATH)
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open', 'high': 'High', 'low': 'Low',
    'close': 'Close', 'volume': 'Volume'
})
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
data = data.astype(float)
print(f"🌙 Data loaded: {len(data)} bars ✨")


def heikin_ashi(open_, high, low, close):
    ha_close = (open_ + high + low + close) / 4.0
    ha_open = np.zeros_like(ha_close)
    ha_open[0] = (open_[0] + close[0]) / 2.0
    for i in range(1, len(ha_close)):
        ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2.0
    ha_high = np.maximum.reduce([high, ha_open, ha_close])
    ha_low = np.minimum.reduce([low, ha_open, ha_close])
    return ha_open, ha_high, ha_low, ha_close


def rolling_percentile_rank(arr, lookback):
    arr = np.asarray(arr, dtype=float)
    out = np.full(len(arr), np.nan)
    for i in range(lookback, len(arr)):
        window = arr[i - lookback:i + 1]
        if np.any(np.isnan(window)):
            continue
        out[i] = (np.sum(window <= arr[i]) / len(window)) * 100.0
    return out


class SqueezeHekiReversal(Strategy):
    bb_period = 20
    bb_std = 2.0
    bbw_lookback = 100
    bbw_threshold = 20.0
    atr_period = 14
    vol_period = 20
    vol_mult = 1.5
    ema_period = 200
    atr_stop_mult = 1.5
    atr_tp1_mult = 1.5
    atr_tp2_mult = 3.0
    time_stop_bars = 10
    risk_pct = 0.01

    def init(self):
        print("🌙 Initializing indicators... ✨")
        close = pd.Series(np.asarray(self.data.Close, dtype=float))
        high = pd.Series(np.asarray(self.data.High, dtype=float))
        low = pd.Series(np.asarray(self.data.Low, dtype=float))
        open_ = pd.Series(np.asarray(self.data.Open, dtype=float))
        volume = pd.Series(np.asarray(self.data.Volume, dtype=float))

        self.bb_upper = self.I(
            lambda c: talib.SMA(c, timeperiod=self.bb_period) +
                      self.bb_std * talib.STDDEV(c, timeperiod=self.bb_period, nbdev=1),
            close
        )
        self.bb_lower = self.I(
            lambda c: talib.SMA(c, timeperiod=self.bb_period) -
                      self.bb_std * talib.STDDEV(c, timeperiod=self.bb_period, nbdev=1),
            close
        )

        def bbw(c):
            basis = talib.SMA(c, timeperiod=self.bb_period)
            sd = talib.STDDEV(c, timeperiod=self.bb_period, nbdev=1)
            with np.errstate(divide='ignore', invalid='ignore'):
                return (2 * self.bb_std * sd) / basis

        self.bbw = self.I(bbw, close)
        self.bbw_pct = self.I(rolling_percentile_rank, self.bbw, self.bbw_lookback)

        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)
        self.vol_sma = self.I(talib.SMA, volume, timeperiod=self.vol_period)
        self.ema200 = self.I(talib.EMA, close, timeperiod=self.ema_period)

        ha_o, ha_h, ha_l, ha_c = heikin_ashi(
            np.asarray(self.data.Open, dtype=float),
            np.asarray(self.data.High, dtype=float),
            np.asarray(self.data.Low, dtype=float),
            np.asarray(self.data.Close, dtype=float),
        )
        self.ha_open = self.I(lambda x: x, ha_o)
        self.ha_close = self.I(lambda x: x, ha_c)
        self.ha_high = self.I(lambda x: x, ha_h)
        self.ha_low = self.I(lambda x: x, ha_l)

        def vwap(o, h, l, c, v):
            tp = (h + l + c) / 3.0
            cum_v = np.cumsum(v)
            with np.errstate(divide='ignore', invalid='ignore'):
                return np.cumsum(tp * v) / cum_v

        self.vwap = self.I(vwap, open_, high, low, close, volume)

        self.roll_max_high = self.I(lambda h: pd.Series(h).rolling(10).max().values, high)
        self.roll_min_low = self.I(lambda l: pd.Series(l).rolling(10).min().values, low)

        self.entry_bar = None
        self.entry_price_val = None
        self.stop_price = None
        self.tp1_price = None
        self.tp2_price = None
        self.tp1_hit = False
        self.direction = 0

        print("🌙 Indicators ready! 🚀")

    def next(self):
        if len(self.data) < self.bbw_lookback + 5:
            return

        price = self.data.Close[-1]
        ha_c = self.ha_close[-1]
        ha_o = self.ha_open[-1]
        ha_c_prev = self.ha_close[-2]
        ha_o_prev = self.ha_open[-2]

        bbw_pct = self.bbw_pct[-1]
        atr = self.atr[-1]
        vol = self.data.Volume[-1]
        vol_sma = self.vol_sma[-1]
        vwap = self.vwap[-1]
        ema200 = self.ema200[-1]

        if np.isnan(bbw_pct) or np.isnan(atr) or np.isnan(vwap) or np.isnan(ema200):
            return
        if np.isnan(vol_sma) or vol_sma <= 0:
            return

        squeeze = bbw_pct < self.bbw_threshold
        vol_ok = vol > self.vol_mult * vol_sma

        bull_flip = (ha_c_prev < ha_o_prev) and (ha_c > ha_o)
        bear_flip = (ha_c_prev > ha_o_prev) and (ha_c < ha_o)

        long_breakout = price > self.bb_upper[-1]
        short_breakout = price < self.bb_lower[-1]

        long_signal = squeeze and bull_flip and long_breakout and (price > vwap) and vol_ok and (price > ema200)
        short_signal = squeeze and bear_flip and short_breakout and (price < vwap) and vol_ok and (price < ema200)

        # Manage open position
        if self.position:
            bars_held = len(self.data) - self.entry_bar
            if self.direction == 1:
                if self.data.Close[-1] < self.vwap[-1] and self.data.Close[-2] < self.vwap[-2]:
                    print(f"🌙 VWAP exit long @ {price:.2f} ✨")
                    self.position.close()
                    self.direction = 0
                    return
                hh = self.roll_max_high[-1]
                chandelier = hh - 3 * atr
                if not np.isnan(chandelier):
                    new_stop = max(self.stop_price, chandelier)
                    if new_stop > self.stop_price:
                        self.stop_price = new_stop

                if self.data.Low[-1] <= self.stop_price:
                    print(f"🛑 Long stop hit @ {self.stop_price:.2f} 🌙")
                    self.position.close()
                    self.direction = 0
                    return
                if not self.tp1_hit and self.data.High[-1] >= self.tp1_price:
                    print(f"🎯 TP1 hit long @ {self.tp1_price:.2f} 🚀")
                    self.position.close()
                    self.tp1_hit = True
                    return
                if self.data.High[-1] >= self.tp2_price:
                    print(f"🎯 TP2 hit long @ {self.tp2_price:.2f} 🚀")
                    self.position.close()
                    self.direction = 0
                    return
                if bear_flip and price > self.entry_price_val:
                    print(f"🔄 HA flip exit long @ {price:.2f} 🌙")
                    self.position.close()
                    self.direction = 0
                    return
                if bars_held >= self.time_stop_bars:
                    print(f"⏰ Time stop long @ {price:.2f} ✨")
                    self.position.close()
                    self.direction = 0
                    return

            elif self.direction == -1:
                if self.data.Close[-1] > self.vwap[-1] and self.data.Close[-2] > self.vwap[-2]:
                    print(f"🌙 VWAP exit short @ {price:.2f} ✨")
                    self.position.close()
                    self.direction = 0
                    return
                ll = self.roll_min_low[-1]
                chandelier = ll + 3 * atr
                if not np.isnan(chandelier):
                    new_stop = min(self.stop_price, chandelier)
                    if new_stop < self.stop_price:
                        self.stop_price = new_stop

                if self.data.High[-1] >= self.stop_price:
                    print(f"🛑 Short stop hit @ {self.stop_price:.2f} 🌙")
                    self.position.close()
                    self.direction = 0
                    return
                if not self.tp1_hit and self.data.Low[-1] <= self.tp1_price:
                    print(f"🎯 TP1 hit short @ {self.tp1_price:.2f} 🚀")
                    self.position.close()
                    self.tp1_hit = True
                    return
                if self.data.Low[-1] <= self.tp2_price:
                    print(f"🎯 TP2 hit short @ {self.tp2_price:.2f} 🚀")
                    self.position.close()
                    self.direction = 0
                    return
                if bull_flip and price < self.entry_price_val:
                    print(f"🔄 HA flip exit short @ {price:.2f} 🌙")
                    self.position.close()
                    self.direction = 0
                    return
                if bars_held >= self.time_stop_bars:
                    print(f"⏰ Time stop short @ {price:.2f} ✨")
                    self.position.close()
                    self.direction = 0
                    return
            return

        # Entries
        if long_signal:
            entry = price
            stop = min(self.ha_low[-1], entry - self.atr_stop_mult * atr)
            risk = entry - stop
            if risk <= 0:
                return
            equity = self.equity
            risk_amount = equity * self.risk_pct
            size = int(round(risk_amount / risk))
            size = max(1, min(size, int(equity / entry)))
            if size < 1:
                return
            print(f"🚀🌙 LONG entry @ {entry:.2f} | size={size} | stop={stop:.2f} | BBW%={bbw_pct:.1f}")
            self.buy(size=size)
            self.entry_bar = len(self.data)
            self.entry_price_val = entry
            self.stop_price = stop
            self.tp1_price = entry + self.atr_tp1_mult * atr
            self.tp2_price = entry + self.atr_tp2_mult * atr
            self.tp1_hit = False
            self.direction = 1

        elif short_signal:
            entry = price
            stop = max(self.ha_high[-1], entry + self.atr_stop_mult * atr)
            risk = stop - entry
            if risk <= 0:
                return
            equity = self.equity
            risk_amount = equity * self.risk_pct
            size = int(round(risk_amount / risk))
            size = max(1, min(size, int(equity / entry)))
            if size < 1:
                return
            print(f"🚀🌙 SHORT entry @ {entry:.2f} | size={size} | stop={stop:.2f} | BBW%={bbw_pct:.1f}")
            self.sell(size=size)
            self.entry_bar = len(self.data)
            self.entry_price_val = entry
            self.stop_price = stop
            self.tp1_price = entry - self.atr_tp1_mult * atr
            self.tp2_price = entry - self.atr_tp2_mult * atr
            self.tp1_hit = False
            self.direction = -1


bt = Backtest(
    data,
    SqueezeHekiReversal,
    cash=1_000_000,
    commission=0.0005,
    exclusive_orders=True,
)

print("🌙✨ Running SqueezeHeki Reversal Backtest... 🚀")
stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙 Backtest complete! ✨")