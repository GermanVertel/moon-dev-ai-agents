import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy


print("🌙✨🚀 Moon Dev Backtest AI initializing... BandedMomentum strategy loading! 🚀✨🌙")


def load_data(path):
    print("🌙 Loading data from:", path)
    data = pd.read_csv(path)
    data.columns = data.columns.str.strip().str.lower()
    data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
    data = data.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume',
    })
    if 'datetime' in data.columns:
        data['datetime'] = pd.to_datetime(data['datetime'])
        data = data.set_index('datetime')
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    print("🌙 Data loaded successfully! Shape:", data.shape)
    return data


class BandedMomentum(Strategy):
    bb_period = 20
    bb_std = 2.0
    atr_period = 14
    atr_stop_mult = 0.10
    risk_pct = 0.01

    def init(self):
        print("🌙✨ Initializing BandedMomentum indicators...")
        self.ub, self.mb, self.lb = self.I(
            talib.BBANDS,
            self.data.Close,
            timeperiod=self.bb_period,
            nbdevup=self.bb_std,
            nbdevdn=self.bb_std,
            matype=0,
        )
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period)
        print("🚀 Indicators ready! BB(20,2) + ATR(14) loaded.")

    def next(self):
        price = self.data.Close[-1]

        if len(self.data) < self.bb_period + 2:
            return

        ub_prev = self.ub[-2]
        ub_now = self.ub[-1]
        mb_prev = self.mb[-2]
        mb_now = self.mb[-1]
        lb_now = self.lb[-1]
        atr_now = self.atr[-1]

        if np.isnan(ub_prev) or np.isnan(mb_prev) or np.isnan(atr_now):
            return

        cross_up = (ub_prev <= mb_prev) and (ub_now > mb_now)
        above_mb = price > mb_now

        if not self.position:
            if cross_up and above_mb:
                stop_price = price - (self.atr_stop_mult * atr_now)
                risk_per_unit = price - stop_price
                if risk_per_unit <= 0:
                    return
                equity = self.equity
                risk_amount = equity * self.risk_pct
                size = risk_amount / risk_per_unit
                size = int(round(size))
                if size < 1:
                    size = 1
                print(f"🌙🚀 LONG ENTRY! UB crossed above MB | Price={price:.2f} MB={mb_now:.2f} "
                      f"ATR={atr_now:.2f} Stop={stop_price:.2f} Size={size}")
                self.buy(size=size, sl=stop_price)
        else:
            if price <= lb_now:
                print(f"🌙✨ EXIT: Price {price:.2f} touched/closed below LB {lb_now:.2f}. Closing long.")
                self.position.close()


data = load_data("/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv")

bt = Backtest(
    data,
    BandedMomentum,
    cash=1_000_000,
    commission=0.0002,
    exclusive=False,
)

print("🌙✨🚀 Running BandedMomentum backtest... 🚀✨🌙")
stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙✨🚀 Moon Dev backtest complete! 🚀✨🌙")