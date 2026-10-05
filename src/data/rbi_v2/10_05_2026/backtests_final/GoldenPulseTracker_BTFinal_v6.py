import pandas as pd
import numpy as np
import talib
from backtesting import Backtest, Strategy

# Load data
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

print("🌙 GoldenPulse Tracker — Moon Dev Backtest Initializing... ✨🚀")
print(f"📊 Data loaded: {len(data)} bars")
print(f"📅 Range: {data.index[0]} → {data.index[-1]}")


class GoldenPulseTracker(Strategy):
    ema_fast_period = 50
    ema_slow_period = 200
    adx_period = 14
    adx_threshold = 25
    atr_period = 14
    atr_mult = 2.0
    risk_pct = 0.02

    def init(self):
        print("🌙 Initializing indicators... ✨")
        self.ema_fast = self.I(talib.EMA, self.data.Close, timeperiod=self.ema_fast_period)
        self.ema_slow = self.I(talib.EMA, self.data.Close, timeperiod=self.ema_slow_period)
        self.adx = self.I(talib.ADX, self.data.High, self.data.Low, self.data.Close, timeperiod=self.adx_period)
        self.plus_di = self.I(talib.PLUS_DI, self.data.High, self.data.Low, self.data.Close, timeperiod=self.adx_period)
        self.minus_di = self.I(talib.MINUS_DI, self.data.High, self.data.Low, self.data.Close, timeperiod=self.adx_period)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period)
        print("🌙 Indicators ready! 🚀")

    def next(self):
        price = self.data.Close[-1]

        if len(self.data) < self.ema_slow_period + 2:
            return

        # -------- Manage open position (trailing stop) --------
        if self.position:
            atr_val = self.atr[-1]
            if np.isnan(atr_val):
                return
            trail_stop = self.data.Close[-1] - self.atr_mult * atr_val
            current_sl = self.position.sl if self.position.sl else 0
            if trail_stop > current_sl:
                self.position.sl = trail_stop

            # Optional trend exit: death cross
            if self.ema_fast[-1] < self.ema_slow[-1] and self.ema_fast[-2] >= self.ema_slow[-2]:
                print(f"💀 Death cross detected at {price:.2f} — exiting long. 🌙")
                self.position.close()
            return

        # -------- Entry logic --------
        ema_f = self.ema_fast[-1]
        ema_s = self.ema_slow[-1]
        ema_f_prev = self.ema_fast[-2]
        ema_s_prev = self.ema_slow[-2]

        if np.isnan([ema_f, ema_s, ema_f_prev, ema_s_prev, self.adx[-1], self.adx[-2], self.atr[-1]]).any():
            return

        golden_cross = (ema_f_prev <= ema_s_prev) and (ema_f > ema_s)
        adx_ok = (self.adx[-1] > self.adx_threshold) and (self.adx[-1] > self.adx[-2])
        di_ok = self.plus_di[-1] > self.minus_di[-1]

        if golden_cross and adx_ok and di_ok:
            atr_val = self.atr[-1]
            if atr_val <= 0 or np.isnan(atr_val):
                return

            stop_distance = self.atr_mult * atr_val
            risk_amount = self.equity * self.risk_pct

            # Position sizing: use fraction of equity (0 < size < 1)
            # size as fraction of equity allocated to this trade
            size_fraction = risk_amount / self.equity
            # Cap fraction to reasonable max (e.g. 95% of equity)
            size_fraction = min(size_fraction, 0.95)
            if size_fraction <= 0:
                print("⚠️ Invalid size fraction. Skipping. 🌙")
                return

            sl = price - stop_distance
            print(f"🌟 GOLDEN PULSE SIGNAL! ✨ Cross @ {price:.2f} | ADX={self.adx[-1]:.2f} ↑ | +DI={self.plus_di[-1]:.2f} > -DI={self.minus_di[-1]:.2f}")
            print(f"🚀 Entering LONG size={size_fraction:.4f} | SL={sl:.2f} (2×ATR={stop_distance:.2f}) 🌙")
            self.buy(size=size_fraction, sl=sl)


bt = Backtest(data, GoldenPulseTracker, cash=1_000_000, commission=0.002, exclusive_orders=True)
stats = bt.run()
print(stats)
print(stats._strategy)