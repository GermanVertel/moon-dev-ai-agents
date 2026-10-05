import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# 🌙 Moon Dev's MomentumVolumeDivergence Backtest 🚀

data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"

print("🌙 Loading Moon Dev data...")
data = pd.read_csv(data_path)

# Clean columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper mapping
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

print(f"✨ Data loaded: {len(data)} bars 🚀")


class MomentumVolumeDivergence(Strategy):
    fast_ema_period = 5
    slow_ema_period = 20
    vol_ma_period = 20
    atr_period = 14
    vol_spike_mult = 1.5
    atr_trail_mult = 2.0
    risk_pct = 0.02
    swing_lookback = 5
    fresh_bars = 3

    def init(self):
        print("🌙 Initializing Moon Dev indicators... ✨")
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        self.fast_ema = self.I(talib.EMA, close, timeperiod=self.fast_ema_period)
        self.slow_ema = self.I(talib.EMA, close, timeperiod=self.slow_ema_period)
        self.vol_ma = self.I(talib.SMA, volume, timeperiod=self.vol_ma_period)
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)
        self.swing_low = self.I(talib.MIN, low, timeperiod=self.swing_lookback)
        self.swing_high = self.I(talib.MAX, high, timeperiod=self.swing_lookback)

        self.trail_stop = None
        self.entry_price = None
        self.entry_dir = 0

    def next(self):
        price = self.data.Close[-1]
        fast = self.fast_ema[-1]
        slow = self.slow_ema[-1]
        vol = self.data.Volume[-1]
        vol_avg = self.vol_ma[-1]
        atr = self.atr[-1]

        if np.isnan(fast) or np.isnan(slow) or np.isnan(vol_avg) or np.isnan(atr):
            return

        divergence = fast - slow

        # Manage open position with trailing stop
        if self.position:
            if self.entry_dir == 1:
                new_stop = self.data.Close[-1] - self.atr_trail_mult * atr
                swing_stop = self.swing_low[-1]
                candidate = max(new_stop, swing_stop)
                if self.trail_stop is None or candidate > self.trail_stop:
                    self.trail_stop = candidate
                    print(f"🌙 Long trail updated: {self.trail_stop:.2f} ✨")
                if self.data.Low[-1] <= self.trail_stop:
                    print(f"🚀 LONG EXIT at {price:.2f} | trail={self.trail_stop:.2f} 🌙")
                    self.position.close()
                    self.trail_stop = None
                    self.entry_dir = 0
                    return
                # Momentum collapse exit
                if divergence < 0:
                    print(f"💥 Momentum collapse exit LONG at {price:.2f} 🌙")
                    self.position.close()
                    self.trail_stop = None
                    self.entry_dir = 0
                    return

            elif self.entry_dir == -1:
                new_stop = self.data.Close[-1] + self.atr_trail_mult * atr
                swing_stop = self.swing_high[-1]
                candidate = min(new_stop, swing_stop)
                if self.trail_stop is None or candidate < self.trail_stop:
                    self.trail_stop = candidate
                    print(f"🌙 Short trail updated: {self.trail_stop:.2f} ✨")
                if self.data.High[-1] >= self.trail_stop:
                    print(f"🚀 SHORT EXIT at {price:.2f} | trail={self.trail_stop:.2f} 🌙")
                    self.position.close()
                    self.trail_stop = None
                    self.entry_dir = 0
                    return
                if divergence > 0:
                    print(f"💥 Momentum collapse exit SHORT at {price:.2f} 🌙")
                    self.position.close()
                    self.trail_stop = None
                    self.entry_dir = 0
                    return
            return

        # Entry logic
        if len(self.data) < self.slow_ema_period + self.fresh_bars + 2:
            return

        prev_div = self.fast_ema[-2] - self.slow_ema[-2]
        div_prev2 = self.fast_ema[-3] - self.slow_ema[-3]

        vol_spike = vol >= self.vol_spike_mult * vol_avg

        # Fresh bullish divergence
        long_cond = (
            divergence > 0 and
            divergence > prev_div and
            prev_div > div_prev2 and
            vol_spike and
            price > fast
        )

        # Fresh bearish divergence
        short_cond = (
            divergence < 0 and
            divergence < prev_div and
            prev_div < div_prev2 and
            vol_spike and
            price < fast
        )

        if long_cond:
            stop = self.swing_low[-1]
            risk = price - stop
            if risk <= 0:
                return
            size = int(round(1_000_000))
            print(f"🌙✨ LONG ENTRY at {price:.2f} | div={divergence:.2f} | vol={vol:.2f} vs avg={vol_avg:.2f} 🚀")
            self.buy(size=size)
            self.entry_dir = 1
            self.entry_price = price
            self.trail_stop = stop

        elif short_cond:
            stop = self.swing_high[-1]
            risk = stop - price
            if risk <= 0:
                return
            size = int(round(1_000_000))
            print(f"🌙✨ SHORT ENTRY at {price:.2f} | div={divergence:.2f} | vol={vol:.2f} vs avg={vol_avg:.2f} 🚀")
            self.sell(size=size)
            self.entry_dir = -1
            self.entry_price = price
            self.trail_stop = stop


print("🌙 Starting Moon Dev Backtest... 🚀")
bt = Backtest(data, MomentumVolumeDivergence, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)