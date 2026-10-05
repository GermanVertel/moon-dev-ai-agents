import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

print("🌙✨ SqueezeIgnition Backtest Initializing... 🚀")

data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"

data = pd.read_csv(data_path)
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
if 'datetime' in data.columns:
    data = data.set_index(pd.to_datetime(data['datetime']))
    data = data.drop(columns=['datetime'])
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
data = data.astype(float)
data = data.dropna()

print(f"🌙 Data loaded: {len(data)} bars ✨")


class SqueezeIgnition(Strategy):
    bb_period = 20
    bb_std = 2.0
    vol_period = 50
    squeeze_lookback = 20
    atr_period = 14
    atr_stop_mult = 1.5
    risk_pct = 0.02
    time_stop_bars = 15
    vol_mult = 1.5

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        def _bbands(c):
            c = np.asarray(c, dtype=np.float64)
            u, m, l = talib.BBANDS(c, timeperiod=self.bb_period,
                                   nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0)
            return u, m, l

        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            _bbands, close
        )

        self.band_width = self.I(
            lambda u, m, l: (u - l) / m, self.bb_upper, self.bb_middle, self.bb_lower
        )

        self.bw_min = self.I(lambda x: talib.MIN(np.asarray(x, dtype=np.float64),
                                                 timeperiod=self.squeeze_lookback),
                             self.band_width)

        self.vol_sma = self.I(lambda v: talib.SMA(np.asarray(v, dtype=np.float64),
                                                  timeperiod=self.vol_period),
                              volume)

        self.atr = self.I(lambda h, l, c: talib.ATR(np.asarray(h, dtype=np.float64),
                                                    np.asarray(l, dtype=np.float64),
                                                    np.asarray(c, dtype=np.float64),
                                                    timeperiod=self.atr_period),
                          high, low, close)

        self.bar_count = 0
        self.entry_bar = None

        print("🌙✨ Indicators initialized successfully 🚀")

    def next(self):
        self.bar_count += 1

        if len(self.data) < max(self.bb_period, self.vol_period, self.squeeze_lookback, self.atr_period) + 2:
            return

        price = self.data.Close[-1]
        upper = self.bb_upper[-1]
        lower = self.bb_lower[-1]
        middle = self.bb_middle[-1]
        bw = self.band_width[-1]
        bw_prev = self.band_width[-2]
        bw_min_prev = self.bw_min[-2]
        vol = self.data.Volume[-1]
        vol_avg = self.vol_sma[-1]
        atr = self.atr[-1]

        if np.isnan(upper) or np.isnan(bw_prev) or np.isnan(bw_min_prev) or np.isnan(vol_avg) or np.isnan(atr):
            return

        squeeze = bw_prev <= bw_min_prev
        vol_spike = vol > (self.vol_mult * vol_avg)

        bar_range = self.data.High[-1] - self.data.Low[-1]
        if bar_range > 0:
            close_pos = (price - self.data.Low[-1]) / bar_range
        else:
            close_pos = 0.5

        if self.position:
            if self.position.is_long:
                if price < middle:
                    print(f"🌙 Exit LONG: price {price:.2f} below middle {middle:.2f} ✨")
                    self.position.close()
                    self.entry_bar = None
                elif bw < bw_prev:
                    print(f"🌙 Exit LONG: band width contracting {bw:.4f} < {bw_prev:.4f} 🚀")
                    self.position.close()
                    self.entry_bar = None
                elif self.entry_bar and (self.bar_count - self.entry_bar) >= self.time_stop_bars:
                    print(f"🌙 Exit LONG: time stop after {self.time_stop_bars} bars ✨")
                    self.position.close()
                    self.entry_bar = None

            elif self.position.is_short:
                if price > middle:
                    print(f"🌙 Exit SHORT: price {price:.2f} above middle {middle:.2f} ✨")
                    self.position.close()
                    self.entry_bar = None
                elif bw < bw_prev:
                    print(f"🌙 Exit SHORT: band width contracting {bw:.4f} < {bw_prev:.4f} 🚀")
                    self.position.close()
                    self.entry_bar = None
                elif self.entry_bar and (self.bar_count - self.entry_bar) >= self.time_stop_bars:
                    print(f"🌙 Exit SHORT: time stop after {self.time_stop_bars} bars ✨")
                    self.position.close()
                    self.entry_bar = None
        else:
            if squeeze and vol_spike:
                if price > upper and close_pos > 0.75:
                    sl = price - (self.atr_stop_mult * atr)
                    print(f"🚀 LONG SIGNAL: price {price:.2f} > upper {upper:.2f} | squeeze={squeeze} vol_spike={vol_spike} | SL={sl:.2f} 🌙")
                    self.buy(size=0.95, sl=sl)
                    self.entry_bar = self.bar_count

                elif price < lower and close_pos < 0.25:
                    sl = price + (self.atr_stop_mult * atr)
                    print(f"🚀 SHORT SIGNAL: price {price:.2f} < lower {lower:.2f} | squeeze={squeeze} vol_spike={vol_spike} | SL={sl:.2f} 🌙")
                    self.sell(size=0.95, sl=sl)
                    self.entry_bar = self.bar_count


bt = Backtest(data, SqueezeIgnition, cash=10_000_000, commission=0.001, exclusive_orders=True)

stats = bt.run()
print(stats)
print(stats._strategy)