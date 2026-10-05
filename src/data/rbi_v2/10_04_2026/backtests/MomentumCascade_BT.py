import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

print("🌙✨ Moon Dev MomentumCascade Backtest Initializing... 🚀")

# Load data
data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

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

# Set datetime index
if 'datetime' in data.columns:
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')

print(f"🌙 Data loaded: {len(data)} rows | {data.index[0]} → {data.index[-1]} ✨")


def rmi(series, period=14, lookback=14):
    """Relative Momentum Index"""
    series = pd.Series(series)
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    up_avg = up.ewm(alpha=1/period, adjust=False).mean()
    down_avg = down.ewm(alpha=1/period, adjust=False).mean()
    rmi_val = 100 - (100 / (1 + up_avg / down_avg.replace(0, np.nan)))
    return rmi_val.values


class MomentumCascade(Strategy):
    rmi_period = 14
    rmi_lookback = 52
    r1_threshold = 70
    hard_stop_pct = 0.20
    rmi5_period = 5
    cum_multiplier = 3.0

    def init(self):
        print("🌙 Initializing MomentumCascade indicators... ✨")
        close = pd.Series(self.data.Close)

        # 52W RMI (approximated on available data using long lookback)
        self.rmi52 = self.I(lambda x: rmi(x, self.rmi_period, self.rmi_lookback),
                            self.data.Close, name="RMI52")

        # 5-day RMI
        self.rmi5 = self.I(lambda x: rmi(x, self.rmi5_period, self.rmi5_period),
                           self.data.Close, name="RMI5")

        # Cumulative RMI5 delta
        rmi5_series = pd.Series(self.rmi5)
        rmi5_delta = rmi5_series.diff()
        self.rmi5_cum = self.I(lambda: rmi5_delta.rolling(10).sum().fillna(0).values,
                               name="RMI5_Cum")

        # Baseline RMI gain (rolling mean of abs deltas)
        baseline = rmi5_delta.abs().rolling(20).mean().fillna(1.0)
        self.baseline = self.I(lambda: baseline.values, name="Baseline")

        self.entry_price = None
        print("🚀 Indicators ready! Let's cascade some momentum! 🌙")

    def next(self):
        price = self.data.Close[-1]

        if not self.position:
            # Entry: RMI52 crosses above R1
            if len(self.rmi52) > 2:
                prev = self.rmi52[-2]
                curr = self.rmi52[-1]
                if not np.isnan(prev) and not np.isnan(curr):
                    if prev < self.r1_threshold and curr >= self.r1_threshold:
                        size = int(round(1_000_000 * 0.70 / price))
                        if size > 0:
                            print(f"🌙✨ ENTRY SIGNAL! RMI52 crossed R1={self.r1_threshold} "
                                  f"(prev={prev:.2f} → curr={curr:.2f}) | Price={price:.2f} | Size={size} 🚀")
                            self.buy(size=size)
                            self.entry_price = price
        else:
            # Exit A: Hard stop -20%
            if self.entry_price and price <= self.entry_price * (1 - self.hard_stop_pct):
                print(f"🛑 HARD STOP HIT! Entry={self.entry_price:.2f} → Price={price:.2f} "
                      f"(-{self.hard_stop_pct*100:.0f}%) | Closing position 🌙")
                self.position.close()
                self.entry_price = None
                return

            # Exit B: Cumulative RMI5 gain > 3x baseline
            if len(self.rmi5_cum) > 1 and len(self.baseline) > 1:
                cum = self.rmi5_cum[-1]
                base = self.baseline[-1]
                if not np.isnan(cum) and not np.isnan(base) and base > 0:
                    if cum > self.cum_multiplier * base:
                        print(f"⚡ MOMENTUM EXHAUSTION! CumRMI5={cum:.2f} > "
                              f"{self.cum_multiplier}x Baseline={base:.2f} | Exiting 🌙")
                        self.position.close()
                        self.entry_price = None
                        return


print("🌙 Setting up backtest engine... 🚀")
bt = Backtest(data, MomentumCascade, cash=1_000_000, commission=0.002)

print("✨ Running MomentumCascade backtest... 🌙")
stats = bt.run()
print(stats)
print(stats._strategy)
print("🚀 Moon Dev backtest complete! ✨🌙")