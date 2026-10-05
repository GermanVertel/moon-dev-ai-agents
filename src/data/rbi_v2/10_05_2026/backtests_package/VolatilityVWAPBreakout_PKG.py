import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# 🌙 Moon Dev Backtest - VolatilityVWAPBreakout ✨🚀

print("🌙 Moon Dev: Loading data from the cosmos...")
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
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
print(f"🌙 Moon Dev: Data loaded with {len(data)} bars 🚀")


def rolling_vwap(close, volume, period):
    """Rolling VWAP using talib SMA on price*volume and volume"""
    pv = close * volume
    return talib.SMA(pv, timeperiod=period) / talib.SMA(volume, timeperiod=period)


def upper_band(vwap, atr, mult):
    """Upper VWAP band = VWAP + mult * ATR"""
    return vwap + mult * atr


def lower_band(vwap, atr, mult):
    """Lower VWAP band = VWAP - mult * ATR"""
    return vwap - mult * atr


class VolatilityVWAPBreakout(Strategy):
    vwap_period = 20
    atr_period = 14
    atr_mult = 2.0
    risk_pct = 0.02  # 2% risk per trade
    max_hold_bars = 100  # time-stop

    def init(self):
        print("🌙 Moon Dev: Initializing indicators... ✨")
        # VWAP via talib-wrapped helper (no backtesting.lib)
        self.vwap = self.I(
            rolling_vwap,
            self.data.Close,
            self.data.Volume,
            self.vwap_period,
        )
        # ATR via talib
        self.atr = self.I(
            talib.ATR,
            self.data.High,
            self.data.Low,
            self.data.Close,
            timeperiod=self.atr_period,
        )
        # Bands computed from already-wrapped indicators (no backtesting.lib)
        self.upper = self.I(upper_band, self.vwap, self.atr, self.atr_mult)
        self.lower = self.I(lower_band, self.vwap, self.atr, self.atr_mult)

        # State tracking
        self.entry_band = None
        self.post_high = None
        self.post_low = None
        self.entry_bar = None
        self.direction = 0

    def next(self):
        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]

        # Manage open position
        if self.position:
            bars_held = len(self.data) - self.entry_bar

            if self.direction == 1:  # Long
                if high > self.post_high:
                    self.post_high = high
                exit_level = self.entry_band + 0.5 * (self.post_high - self.entry_band)
                stop_level = self.vwap[-1]

                if price <= exit_level:
                    print(f"🌙 Moon Dev: LONG 50% retrace exit @ {price:.2f} | high={self.post_high:.2f} ✨")
                    self.position.close()
                    self.direction = 0
                elif price <= stop_level:
                    print(f"🌙 Moon Dev: LONG stop @ VWAP {price:.2f} 🛑")
                    self.position.close()
                    self.direction = 0
                elif bars_held >= self.max_hold_bars:
                    print(f"🌙 Moon Dev: LONG time-stop @ {price:.2f} ⏰")
                    self.position.close()
                    self.direction = 0

            elif self.direction == -1:  # Short
                if low < self.post_low:
                    self.post_low = low
                exit_level = self.entry_band - 0.5 * (self.entry_band - self.post_low)
                stop_level = self.vwap[-1]

                if price >= exit_level:
                    print(f"🌙 Moon Dev: SHORT 50% retrace exit @ {price:.2f} | low={self.post_low:.2f} ✨")
                    self.position.close()
                    self.direction = 0
                elif price >= stop_level:
                    print(f"🌙 Moon Dev: SHORT stop @ VWAP {price:.2f} 🛑")
                    self.position.close()
                    self.direction = 0
                elif bars_held >= self.max_hold_bars:
                    print(f"🌙 Moon Dev: SHORT time-stop @ {price:.2f} ⏰")
                    self.position.close()
                    self.direction = 0
            return

        # Entry logic
        if np.isnan(self.upper[-1]) or np.isnan(self.lower[-1]) or np.isnan(self.atr[-1]):
            return

        if price > self.upper[-1]:
            stop_dist = price - self.vwap[-1]
            if stop_dist <= 0:
                return
            size = int(round((self.equity * self.risk_pct) / stop_dist))
            if size <= 0:
                return
            print(f"🌙 Moon Dev: LONG breakout @ {price:.2f} | upper={self.upper[-1]:.2f} | size={size} 🚀")
            self.buy(size=size)
            self.entry_band = self.upper[-1]
            self.post_high = high
            self.entry_bar = len(self.data)
            self.direction = 1

        elif price < self.lower[-1]:
            stop_dist = self.vwap[-1] - price
            if stop_dist <= 0:
                return
            size = int(round((self.equity * self.risk_pct) / stop_dist))
            if size <= 0:
                return
            print(f"🌙 Moon Dev: SHORT breakout @ {price:.2f} | lower={self.lower[-1]:.2f} | size={size} 🚀")
            self.sell(size=size)
            self.entry_band = self.lower[-1]
            self.post_low = low
            self.entry_bar = len(self.data)
            self.direction = -1


print("🌙 Moon Dev: Launching backtest... 🚀✨")
bt = Backtest(data, VolatilityVWAPBreakout, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)