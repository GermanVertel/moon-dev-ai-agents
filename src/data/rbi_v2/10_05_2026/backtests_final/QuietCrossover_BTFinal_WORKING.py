import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

print("🌙✨ Moon Dev's QuietCrossover Backtest Initializing... 🚀")

data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper case mapping
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

print(f"🌙 Data loaded: {len(data)} bars ✨")
print(f"🚀 Range: {data.index[0]} → {data.index[-1]}")


class QuietCrossover(Strategy):
    fast_ema_period = 50
    slow_ema_period = 200
    adx_period = 14
    adx_entry_threshold = 20
    adx_exit_threshold = 30
    stop_loss_pct = 0.03
    max_bars_hold = 30

    def init(self):
        print("🌙 Initializing indicators... ✨")
        self.ema_fast = self.I(talib.EMA, self.data.Close, timeperiod=self.fast_ema_period)
        self.ema_slow = self.I(talib.EMA, self.data.Close, timeperiod=self.slow_ema_period)
        self.adx = self.I(talib.ADX, self.data.High, self.data.Low, self.data.Close,
                          timeperiod=self.adx_period)
        self.entry_bar = None
        print("🚀 Indicators ready!")

    def next(self):
        price = self.data.Close[-1]

        if len(self.data) < self.slow_ema_period + 2:
            return

        ema_f = self.ema_fast[-1]
        ema_s = self.ema_slow[-1]
        ema_f_prev = self.ema_fast[-2]
        ema_s_prev = self.ema_slow[-2]
        adx_now = self.adx[-1]

        if np.isnan(ema_f) or np.isnan(ema_s) or np.isnan(adx_now):
            return

        # Entry: golden cross + ADX < 20
        if not self.position:
            crossed_up = ema_f_prev <= ema_s_prev and ema_f > ema_s
            if crossed_up and adx_now < self.adx_entry_threshold:
                # Use fraction of equity for position sizing (0 < size < 1)
                print(f"🌙✨ GOLDEN CROSS + QUIET ADX={adx_now:.2f} at {price:.2f} | BUY 🚀")
                self.buy(size=0.95)
                self.entry_bar = len(self.data)
        else:
            # Exit: ADX > 30
            if adx_now > self.adx_exit_threshold:
                print(f"🌙 ADX={adx_now:.2f} > 30, exiting long at {price:.2f} 💰")
                self.position.close()
                self.entry_bar = None
                return

            # Time-based stop
            if self.entry_bar is not None and (len(self.data) - self.entry_bar) >= self.max_bars_hold:
                print(f"⏰ Time stop hit ({self.max_bars_hold} bars), exiting at {price:.2f}")
                self.position.close()
                self.entry_bar = None
                return

            # Hard stop-loss
            if self.trades:
                entry_price = self.trades[-1].entry_price
                if price <= entry_price * (1 - self.stop_loss_pct):
                    print(f"🛑 Stop-loss triggered at {price:.2f} (entry {entry_price:.2f})")
                    self.position.close()
                    self.entry_bar = None


bt = Backtest(data, QuietCrossover, cash=1_000_000, commission=0.001)

print("🌙 Running QuietCrossover backtest... 🚀")
stats = bt.run()
print(stats)
print(stats._strategy)