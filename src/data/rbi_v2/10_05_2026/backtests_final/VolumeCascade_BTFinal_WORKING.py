import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# 🌙 Moon Dev's VolumeCascade Strategy ✨

data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"

print("🌙 Moon Dev: Loading cosmic data from the lunar archives... 🚀")
data = pd.read_csv(data_path)

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Map to proper case
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

# 🌙 Moon Dev: Ensure all OHLCV columns are float64 for talib ✨
for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    data[col] = data[col].astype('float64')

print(f"🌙 Moon Dev: Data loaded with {len(data)} lunar cycles ✨")
print(f"🚀 Columns: {list(data.columns)}")


class VolumeCascade(Strategy):
    fast_period = 10
    slow_period = 50
    vol_period = 20
    atr_period = 14
    atr_stop_mult = 2.0
    atr_target_mult = 3.0
    vol_mult = 1.0
    risk_pct = 0.02
    time_stop_bars = 10

    def init(self):
        print("🌙 Moon Dev: Initializing VolumeCascade indicators... ✨")
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        # 🌙 Moon Dev: Wrap talib calls in lambdas so backtesting.py passes numpy arrays correctly ✨
        self.fast_ma = self.I(lambda c: talib.SMA(c, timeperiod=self.fast_period), close, name='SMA_fast')
        self.slow_ma = self.I(lambda c: talib.SMA(c, timeperiod=self.slow_period), close, name='SMA_slow')
        self.vol_ma = self.I(lambda v: talib.SMA(v, timeperiod=self.vol_period), volume, name='SMA_vol')
        self.atr = self.I(lambda h, l, c: talib.ATR(h, l, c, timeperiod=self.atr_period), high, low, close, name='ATR')

        self.atr_high = self.I(lambda a: talib.MAX(a, timeperiod=100), self.atr, name='ATR_high')
        self.atr_low = self.I(lambda a: talib.MIN(a, timeperiod=100), self.atr, name='ATR_low')

        # Track trade state
        self.entry_price = None
        self.entry_bar = None
        self.highest_high = None
        self.lowest_low = None
        self.stop_price = None
        self.target_price = None
        self.trade_dir = None

        print("🚀 Moon Dev: All indicators aligned with the stars! 🌙")

    def next(self):
        price = self.data.Close[-1]
        vol = self.data.Volume[-1]

        # Manage existing position
        if self.position:
            atr_val = self.atr[-1]
            if np.isnan(atr_val) or atr_val <= 0:
                return

            if self.trade_dir == 'long':
                self.highest_high = max(self.highest_high, self.data.High[-1])
                new_stop = self.highest_high - self.atr_stop_mult * atr_val
                if new_stop > self.stop_price:
                    self.stop_price = new_stop
                    print(f"🌙 Moon Dev: Trailing stop raised to {self.stop_price:.2f} ✨")

                if price <= self.stop_price:
                    print(f"🛑 Moon Dev: LONG STOP hit at {price:.2f} (stop={self.stop_price:.2f})")
                    self.position.close()
                    self._reset_state()
                    return

                if price >= self.target_price:
                    print(f"🎯 Moon Dev: LONG TARGET hit at {price:.2f}! Taking profits 🚀")
                    self.position.close()
                    self._reset_state()
                    return

                if self.entry_price and (price - self.entry_price) < atr_val:
                    bars_held = len(self.data) - 1 - self.entry_bar
                    if bars_held >= self.time_stop_bars:
                        print(f"⏰ Moon Dev: TIME STOP — no 1xATR progress in {bars_held} bars. Exiting.")
                        self.position.close()
                        self._reset_state()
                        return

                # Bearish MA crossover (slow crosses above fast)
                if self.slow_ma[-2] < self.fast_ma[-2] and self.slow_ma[-1] > self.fast_ma[-1]:
                    print(f"🔄 Moon Dev: Bearish MA cross — exiting LONG 🌙")
                    self.position.close()
                    self._reset_state()
                    return

            elif self.trade_dir == 'short':
                self.lowest_low = min(self.lowest_low, self.data.Low[-1])
                new_stop = self.lowest_low + self.atr_stop_mult * atr_val
                if new_stop < self.stop_price:
                    self.stop_price = new_stop
                    print(f"🌙 Moon Dev: Trailing stop lowered to {self.stop_price:.2f} ✨")

                if price >= self.stop_price:
                    print(f"🛑 Moon Dev: SHORT STOP hit at {price:.2f} (stop={self.stop_price:.2f})")
                    self.position.close()
                    self._reset_state()
                    return

                if price <= self.target_price:
                    print(f"🎯 Moon Dev: SHORT TARGET hit at {price:.2f}! Taking profits 🚀")
                    self.position.close()
                    self._reset_state()
                    return

                if self.entry_price and (self.entry_price - price) < atr_val:
                    bars_held = len(self.data) - 1 - self.entry_bar
                    if bars_held >= self.time_stop_bars:
                        print(f"⏰ Moon Dev: TIME STOP — no 1xATR progress in {bars_held} bars. Exiting.")
                        self.position.close()
                        self._reset_state()
                        return

                # Bullish MA crossover (fast crosses above slow)
                if self.fast_ma[-2] < self.slow_ma[-2] and self.fast_ma[-1] > self.slow_ma[-1]:
                    print(f"🔄 Moon Dev: Bullish MA cross — exiting SHORT 🌙")
                    self.position.close()
                    self._reset_state()
                    return
            return

        # Entry logic
        if len(self.data) < self.slow_period + 5:
            return

        atr_val = self.atr[-1]
        if np.isnan(atr_val) or atr_val <= 0:
            return

        atr_hi = self.atr_high[-1]
        atr_lo = self.atr_low[-1]
        if np.isnan(atr_hi) or np.isnan(atr_lo) or atr_hi <= atr_lo:
            return

        atr_range = atr_hi - atr_lo
        atr_pct = (atr_val - atr_lo) / atr_range

        # Volatility guard
        if atr_pct > 0.95:
            print(f"⚠️ Moon Dev: ATR in top 5% ({atr_pct:.2%}) — extreme volatility, skipping 🌙")
            return
        if atr_pct < 0.10:
            print(f"⚠️ Moon Dev: ATR in bottom 10% ({atr_pct:.2%}) — dead market, skipping 🌙")
            return

        vol_ok = vol > self.vol_ma[-1] * self.vol_mult

        # Risk-based position sizing
        equity = self.equity
        risk_dollars = equity * self.risk_pct
        stop_dist = self.atr_stop_mult * atr_val
        if stop_dist <= 0:
            return
        raw_size = risk_dollars / stop_dist
        size = max(1, int(round(raw_size / price)))

        # Bullish crossover (fast crosses above slow)
        bull_cross = self.fast_ma[-2] < self.slow_ma[-2] and self.fast_ma[-1] > self.slow_ma[-1]
        # Bearish crossover (slow crosses above fast)
        bear_cross = self.slow_ma[-2] < self.fast_ma[-2] and self.slow_ma[-1] > self.fast_ma[-1]

        if bull_cross and vol_ok:
            print(f"🚀 Moon Dev: BULLISH crossover + volume confirm! Entry LONG @ {price:.2f} | ATR={atr_val:.2f} | size={size}")
            self.buy(size=size)
            self.entry_price = price
            self.entry_bar = len(self.data) - 1
            self.highest_high = self.data.High[-1]
            self.stop_price = price - self.atr_stop_mult * atr_val
            self.target_price = price + self.atr_target_mult * atr_val
            self.trade_dir = 'long'

        elif bear_cross and vol_ok:
            print(f"🚀 Moon Dev: BEARISH crossover + volume confirm! Entry SHORT @ {price:.2f} | ATR={atr_val:.2f} | size={size}")
            self.sell(size=size)
            self.entry_price = price
            self.entry_bar = len(self.data) - 1
            self.lowest_low = self.data.Low[-1]
            self.stop_price = price + self.atr_stop_mult * atr_val
            self.target_price = price - self.atr_target_mult * atr_val
            self.trade_dir = 'short'

    def _reset_state(self):
        self.entry_price = None
        self.entry_bar = None
        self.highest_high = None
        self.lowest_low = None
        self.stop_price = None
        self.target_price = None
        self.trade_dir = None


print("🌙 Moon Dev: Launching VolumeCascade backtest... 🚀✨")
bt = Backtest(data, VolumeCascade, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)