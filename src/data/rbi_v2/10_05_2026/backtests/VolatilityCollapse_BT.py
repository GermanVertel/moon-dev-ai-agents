import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Load data
data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper column mapping
data.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

# Rename to backtesting.py required format
data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

print("🌙✨ Moon Dev VolatilityCollapse Backtest Initializing... 🚀")
print(f"📊 Data loaded: {len(data)} bars")
print(f"📈 Date range: {data.index[0]} to {data.index[-1]}")


class VolatilityCollapse(Strategy):
    atr_period = 14
    par_ma_period = 20
    ema_period = 20
    atr_stop_mult = 2.0
    risk_pct = 0.01
    time_stop_bars = 30

    def init(self):
        print("🌙 Initializing indicators...")
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close,
                          timeperiod=self.atr_period)
        self.par = self.I(lambda c, a: c / np.where(a == 0, np.nan, a),
                          self.data.Close, self.atr)
        self.par_ma = self.I(talib.SMA, self.par, timeperiod=self.par_ma_period)
        self.ema = self.I(talib.EMA, self.data.Close, timeperiod=self.ema_period)
        print("✨ Indicators ready!")

    def next(self):
        price = self.data.Close[-1]

        if len(self.data) < self.par_ma_period + 2:
            return

        par_now = self.par[-1]
        par_prev = self.par[-2]
        par_ma_now = self.par_ma[-1]
        par_ma_prev = self.par_ma[-2]
        atr_now = self.atr[-1]
        ema_now = self.ema[-1]

        if np.isnan(par_now) or np.isnan(par_ma_now) or np.isnan(atr_now):
            return

        par_roc = par_now - par_prev

        # Entry: short
        if not self.position:
            cross_below_ma = par_prev >= par_ma_prev and par_now < par_ma_now
            if par_roc < 0 and cross_below_ma and price < ema_now:
                stop_price = price + self.atr_stop_mult * atr_now
                risk_per_unit = stop_price - price
                if risk_per_unit > 0:
                    risk_capital = self.equity * self.risk_pct
                    size = int(round(risk_capital / risk_per_unit))
                    if size > 0:
                        print(f"🌙🔻 SHORT ENTRY | Price: {price:.2f} | PAR: {par_now:.2f} | PAR_MA: {par_ma_now:.2f} | ATR: {atr_now:.2f} | Size: {size}")
                        self.sell(size=size, sl=stop_price)
                        self.entry_bar = len(self.data)

        else:
            # Exit conditions
            exit_signal = False
            reason = ""

            if par_roc > 0 and par_now > par_ma_now:
                exit_signal = True
                reason = "PAR reclaim MA + ROC positive"

            if self.position.is_short and price > ema_now:
                exit_signal = True
                reason = "Price above EMA"

            if hasattr(self, 'entry_bar') and (len(self.data) - self.entry_bar) >= self.time_stop_bars:
                exit_signal = True
                reason = "Time stop"

            if exit_signal:
                print(f"🌙✅ EXIT | Reason: {reason} | Price: {price:.2f}")
                self.position.close()


bt = Backtest(data, VolatilityCollapse, cash=1_000_000, commission=0.002)
stats = bt.run()
print(stats)
print(stats._strategy)