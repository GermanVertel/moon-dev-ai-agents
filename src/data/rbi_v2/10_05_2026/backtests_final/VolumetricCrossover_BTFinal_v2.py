import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# 🌙 Moon Dev's VolumetricCrossover Backtest 🚀

print("🌙 Moon Dev is loading the cosmic data... ✨")

data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper column mapping
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')

# 🌙 Ensure all OHLCV columns are float64 (talib requires double arrays) ✨
for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    data[col] = data[col].astype('float64')

print(f"🌙 Moon Dev loaded {len(data)} rows of stellar data! 🚀")
print(f"✨ Columns: {list(data.columns)}")


class VolumetricCrossover(Strategy):
    ema_fast_period = 50
    ema_slow_period = 200
    vol_sma_period = 20
    vol_mult = 2.0
    atr_period = 14
    atr_mult = 1.5
    risk_pct = 0.02

    def init(self):
        print("🌙 Initializing indicators... ✨")
        # 🌙 Use np.asarray to convert _Array to numpy float64 arrays ✨
        close = np.asarray(self.data.Close, dtype=np.float64)
        high = np.asarray(self.data.High, dtype=np.float64)
        low = np.asarray(self.data.Low, dtype=np.float64)
        volume = np.asarray(self.data.Volume, dtype=np.float64)

        self.ema_fast = self.I(talib.EMA, close, timeperiod=self.ema_fast_period)
        self.ema_slow = self.I(talib.EMA, close, timeperiod=self.ema_slow_period)
        self.vol_sma = self.I(talib.SMA, volume, timeperiod=self.vol_sma_period)
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        self.trailing_stop = None
        self.highest_close = None
        self.lowest_close = None
        self.entry_price = None

    def next(self):
        if len(self.data) < self.ema_slow_period + 2:
            return

        price = self.data.Close[-1]
        ema_f = self.ema_fast[-1]
        ema_s = self.ema_slow[-1]
        ema_f_prev = self.ema_fast[-2]
        ema_s_prev = self.ema_slow[-2]
        vol = self.data.Volume[-1]
        vol_avg = self.vol_sma[-1]
        atr = self.atr[-1]

        if np.isnan(ema_f) or np.isnan(ema_s) or np.isnan(vol_avg) or np.isnan(atr):
            return

        # Manage existing position with trailing stop
        if self.position:
            if self.position.is_long:
                self.highest_close = max(self.highest_close, price)
                new_stop = self.highest_close - self.atr_mult * atr
                if self.trailing_stop is None or new_stop > self.trailing_stop:
                    self.trailing_stop = new_stop
                if price <= self.trailing_stop:
                    print(f"🌙💥 LONG TRAILING STOP HIT at {price:.2f} | stop {self.trailing_stop:.2f} 🚀")
                    self.position.close()
                    self.trailing_stop = None
                    self.highest_close = None
                    return
            elif self.position.is_short:
                self.lowest_close = min(self.lowest_close, price)
                new_stop = self.lowest_close + self.atr_mult * atr
                if self.trailing_stop is None or new_stop < self.trailing_stop:
                    self.trailing_stop = new_stop
                if price >= self.trailing_stop:
                    print(f"🌙💥 SHORT TRAILING STOP HIT at {price:.2f} | stop {self.trailing_stop:.2f} 🚀")
                    self.position.close()
                    self.trailing_stop = None
                    self.lowest_close = None
                    return

        # Detect crossover (numpy/pandas comparison, no backtesting.lib)
        bullish_cross = ema_f_prev <= ema_s_prev and ema_f > ema_s
        bearish_cross = ema_f_prev >= ema_s_prev and ema_f < ema_s

        vol_confirm = vol > self.vol_mult * vol_avg

        # Long entry
        if not self.position and bullish_cross and vol_confirm and price > ema_f and price > ema_s:
            stop_dist = self.atr_mult * atr
            if stop_dist <= 0:
                return
            risk_amount = self.equity * self.risk_pct
            size = int(round(risk_amount / stop_dist))
            if size <= 0:
                return
            print(f"🌙🚀 LONG ENTRY | price {price:.2f} | EMA50 {ema_f:.2f} > EMA200 {ema_s:.2f} | vol {vol:.2f} > {self.vol_mult}×{vol_avg:.2f} | size {size} ✨")
            self.buy(size=size)
            self.entry_price = price
            self.highest_close = price
            self.trailing_stop = price - stop_dist

        # Short entry
        elif not self.position and bearish_cross and vol_confirm and price < ema_f and price < ema_s:
            stop_dist = self.atr_mult * atr
            if stop_dist <= 0:
                return
            risk_amount = self.equity * self.risk_pct
            size = int(round(risk_amount / stop_dist))
            if size <= 0:
                return
            print(f"🌙🔻 SHORT ENTRY | price {price:.2f} | EMA50 {ema_f:.2f} < EMA200 {ema_s:.2f} | vol {vol:.2f} > {self.vol_mult}×{vol_avg:.2f} | size {size} ✨")
            self.sell(size=size)
            self.entry_price = price
            self.lowest_close = price
            self.trailing_stop = price + stop_dist


print("🌙 Moon Dev launching the backtest... 🚀✨")
bt = Backtest(data, VolumetricCrossover, cash=1_000_000, commission=0.001)

stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙 Moon Dev's backtest complete! ✨🚀")