import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Load data
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Rename to proper case for backtesting.py
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

# Set datetime index
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')

print("🌙✨ Moon Dev Data Loaded! Rows:", len(data))
print("🚀 Columns:", list(data.columns))


class VolatilityFilteredCrossover(Strategy):
    ema_fast_period = 50
    ema_slow_period = 200
    atr_period = 14
    atr_sma_period = 20
    atr_stop_mult = 2.0
    risk_pct = 0.02

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low

        self.ema_fast = self.I(talib.EMA, close, timeperiod=self.ema_fast_period)
        self.ema_slow = self.I(talib.EMA, close, timeperiod=self.ema_slow_period)
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)
        self.atr_sma = self.I(talib.SMA, self.atr, timeperiod=self.atr_sma_period)

        self.stop_price = None
        print("🌙 VolatilityFilteredCrossover initialized! ✨")

    def next(self):
        # Need enough bars
        if len(self.data) < self.ema_slow_period + 2:
            return

        price = self.data.Close[-1]
        atr_val = self.atr[-1]
        atr_avg = self.atr_sma[-1]

        ema_fast_now = self.ema_fast[-1]
        ema_slow_now = self.ema_slow[-1]
        ema_fast_prev = self.ema_fast[-2]
        ema_slow_prev = self.ema_slow[-2]

        golden_cross = (ema_fast_prev <= ema_slow_prev) and (ema_fast_now > ema_slow_now)
        low_vol = atr_val < atr_avg
        high_vol = atr_val > atr_avg

        # Manage open position
        if self.position:
            # Hard stop
            if self.stop_price is not None and self.data.Low[-1] <= self.stop_price:
                print(f"🛑 ATR Stop hit @ {self.data.Low[-1]:.2f} | stop={self.stop_price:.2f}")
                self.position.close()
                self.stop_price = None
                return

            # Volatility spike exit
            if high_vol:
                print(f"⚡ Volatility spike exit @ {price:.2f} | ATR={atr_val:.2f} > SMA={atr_avg:.2f}")
                self.position.close()
                self.stop_price = None
                return

            # Trend invalidation
            if ema_fast_now < ema_slow_now:
                print(f"📉 Trend invalidation exit @ {price:.2f}")
                self.position.close()
                self.stop_price = None
                return
            return

        # Entry logic
        if golden_cross and low_vol:
            stop_dist = self.atr_stop_mult * atr_val
            if stop_dist <= 0:
                return
            risk_amount = self.equity * self.risk_pct
            size = int(round(risk_amount / stop_dist))
            if size < 1:
                size = 1
            self.stop_price = price - stop_dist
            print(f"🌙✨ GOLDEN CROSS + LOW VOL ENTRY! Price={price:.2f} ATR={atr_val:.2f} SMA={atr_avg:.2f} Size={size} Stop={self.stop_price:.2f} 🚀")
            self.buy(size=size)


bt = Backtest(data, VolatilityFilteredCrossover, cash=1_000_000, commission=0.002, exclusive_orders=True)
stats = bt.run()
print(stats)
print(stats._strategy)