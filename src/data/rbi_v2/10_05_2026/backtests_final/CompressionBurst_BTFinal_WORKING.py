import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# 🌙 Moon Dev CompressionBurst Strategy ✨

data_path = '/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv'
data = pd.read_csv(data_path)

data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})

if 'datetime' in data.columns:
    data = data.set_index(pd.to_datetime(data['datetime']))
    data = data.drop(columns=['datetime'])

data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
data['Volume'] = data['Volume'].astype(float)

print("🌙✨ Moon Dev data loaded:", data.shape)
print(data.head())


class CompressionBurst(Strategy):
    bb_period = 20
    bb_std = 2.0
    bbw_lookback = 30
    vol_period = 20
    atr_period = 14
    vol_mult = 1.5
    risk_pct = 0.01
    squeeze_pct = 0.05
    confirmation_window = 10
    time_stop_bars = 10
    false_breakout_bars = 2

    def init(self):
        close = self.data.Close.astype(float)
        high = self.data.High.astype(float)
        low = self.data.Low.astype(float)
        volume = self.data.Volume.astype(float)

        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS, close, timeperiod=self.bb_period,
            nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0
        )
        self.bbw = self.I(
            lambda u, m, l: (u - l) / np.where(m == 0, np.nan, m),
            self.bb_upper, self.bb_middle, self.bb_lower,
            name='BBW'
        )
        self.bbw_min = self.I(talib.MIN, self.bbw, timeperiod=self.bbw_lookback, name='BBW_MIN')
        self.bbw_max = self.I(talib.MAX, self.bbw, timeperiod=self.bbw_lookback, name='BBW_MAX')
        self.vol_sma = self.I(talib.SMA, volume, timeperiod=self.vol_period, name='VOL_SMA')
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period, name='ATR')

        self.squeeze_bar = -1
        self.entry_bar = -1
        self.entry_price = 0.0
        self.stop_price = 0.0
        self.target_price = 0.0
        self.squeeze_level = 0.0
        self.trade_direction = 0

    def next(self):
        i = len(self.data) - 1
        if i < self.bbw_lookback + 2:
            return

        price = self.data.Close[-1]
        upper = self.bb_upper[-1]
        lower = self.bb_lower[-1]
        middle = self.bb_middle[-1]
        bbw_now = self.bbw[-1]
        bbw_min = self.bbw_min[-1]
        bbw_max = self.bbw_max[-1]
        vol = self.data.Volume[-1]
        vol_avg = self.vol_sma[-1]
        atr = self.atr[-1]

        if np.isnan(bbw_now) or np.isnan(bbw_min) or np.isnan(vol_avg) or np.isnan(atr) or atr <= 0:
            return

        # Squeeze detection
        squeeze_threshold = bbw_min + self.squeeze_pct * (bbw_max - bbw_min)
        in_squeeze = bbw_now <= squeeze_threshold

        if in_squeeze and not self.position:
            if self.squeeze_bar < 0:
                self.squeeze_bar = i
                print(f"🌙✨ SQUEEZE DETECTED at bar {i} | BBW={bbw_now:.5f} BBW_MIN={bbw_min:.5f}")

        # Invalidate squeeze if too old
        if self.squeeze_bar > 0 and (i - self.squeeze_bar) > self.confirmation_window:
            self.squeeze_bar = -1

        # Manage open position
        if self.position:
            bars_held = i - self.entry_bar
            entry = self.entry_price
            stop = self.stop_price

            # False breakout filter
            if bars_held <= self.false_breakout_bars:
                if self.trade_direction == 1 and price < middle:
                    print(f"🌙❌ FALSE BREAKOUT LONG EXIT at {price:.2f}")
                    self.position.close()
                    self._reset()
                    return
                if self.trade_direction == -1 and price > middle:
                    print(f"🌙❌ FALSE BREAKOUT SHORT EXIT at {price:.2f}")
                    self.position.close()
                    self._reset()
                    return

            # Middle band exit
            if self.trade_direction == 1 and price < middle:
                print(f"🌙💫 MIDDLE BAND EXIT LONG at {price:.2f}")
                self.position.close()
                self._reset()
                return
            if self.trade_direction == -1 and price > middle:
                print(f"🌙💫 MIDDLE BAND EXIT SHORT at {price:.2f}")
                self.position.close()
                self._reset()
                return

            # Time stop
            if bars_held >= self.time_stop_bars:
                if self.trade_direction == 1 and price < entry + (entry - stop):
                    print(f"🌙⏰ TIME STOP LONG at {price:.2f}")
                    self.position.close()
                    self._reset()
                    return
                if self.trade_direction == -1 and price > entry - (stop - entry):
                    print(f"🌙⏰ TIME STOP SHORT at {price:.2f}")
                    self.position.close()
                    self._reset()
                    return

            # Volatility exit at 2x squeeze BBW
            if self.squeeze_level > 0 and bbw_now >= 2 * self.squeeze_level:
                print(f"🌙🚀 VOLATILITY EXIT at {price:.2f} BBW={bbw_now:.5f}")
                self.position.close()
                self._reset()
                return

            return

        # Entry logic
        if self.squeeze_bar < 0:
            return

        vol_confirm = vol > self.vol_mult * vol_avg

        if price > upper and vol_confirm:
            stop = max(lower, price - 1.5 * atr)
            risk = price - stop
            if risk <= 0:
                return
            size = int(round(1_000_000 / price))
            if size <= 0:
                return
            self.buy(size=size)
            self.entry_bar = i
            self.entry_price = price
            self.stop_price = stop
            self.target_price = price + 2 * risk
            self.squeeze_level = bbw_now
            self.trade_direction = 1
            self.squeeze_bar = -1
            print(f"🌙🚀 LONG ENTRY at {price:.2f} | stop={stop:.2f} size={size}")

        elif price < lower and vol_confirm:
            stop = min(upper, price + 1.5 * atr)
            risk = stop - price
            if risk <= 0:
                return
            size = int(round(1_000_000 / price))
            if size <= 0:
                return
            self.sell(size=size)
            self.entry_bar = i
            self.entry_price = price
            self.stop_price = stop
            self.target_price = price - 2 * risk
            self.squeeze_level = bbw_now
            self.trade_direction = -1
            self.squeeze_bar = -1
            print(f"🌙🚀 SHORT ENTRY at {price:.2f} | stop={stop:.2f} size={size}")

    def _reset(self):
        self.entry_bar = -1
        self.entry_price = 0.0
        self.stop_price = 0.0
        self.target_price = 0.0
        self.squeeze_level = 0.0
        self.trade_direction = 0


bt = Backtest(data, CompressionBurst, cash=1_000_000, commission=0.0002)
stats = bt.run()
print(stats)
print(stats._strategy)