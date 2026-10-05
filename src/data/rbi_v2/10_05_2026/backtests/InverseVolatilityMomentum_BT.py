import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# 🌙 Moon Dev's InverseVolatilityMomentum Strategy ✨
class InverseVolatilityMomentum(Strategy):
    # Strategy parameters
    rsi_period = 5
    rsi_threshold = 30
    atr_period = 14
    atr_lookback = 20
    atr_drop_pct = 0.01  # 1% drop from ATR high
    sma_period = 200
    atr_stop_mult = 2.0
    risk_pct = 0.01
    time_stop_bars = 10
    atr_expand_pct = 0.02  # 2% expansion exit

    def init(self):
        # 🌙 Calculate indicators using talib wrapped in self.I()
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=self.rsi_period, name='RSI_5')
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close,
                          timeperiod=self.atr_period, name='ATR_14')
        self.atr_max = self.I(talib.MAX, self.atr, timeperiod=self.atr_lookback, name='ATR_Max')
        self.sma200 = self.I(talib.SMA, self.data.Close, timeperiod=self.sma_period, name='SMA_200')

        # Track entry state
        self.entry_bar = None
        self.entry_atr = None
        self.entry_price = None

        print("🌙✨ Moon Dev InverseVolatilityMomentum initialized! 🚀")

    def next(self):
        price = self.data.Close[-1]

        # Skip if indicators not ready
        if np.isnan(self.rsi[-1]) or np.isnan(self.atr[-1]) or np.isnan(self.atr_max[-1]) or np.isnan(self.sma200[-1]):
            return

        # 🌙 Exit logic
        if self.position:
            bars_held = len(self.data) - 1 - self.entry_bar if self.entry_bar is not None else 0

            # Profit target: RSI crosses above 50
            if self.rsi[-1] > 50:
                print(f"🌙✨ RSI crossed above 50 ({self.rsi[-1]:.2f}) — closing long! 🚀")
                self.position.close()
                self.entry_bar = None
                return

            # Volatility exit: ATR expanding > 2% from entry ATR
            if self.entry_atr is not None and self.atr[-1] > self.entry_atr * (1 + self.atr_expand_pct):
                print(f"🌙⚠️ ATR expanding ({self.atr[-1]:.2f} vs entry {self.entry_atr:.2f}) — exiting! 🚀")
                self.position.close()
                self.entry_bar = None
                return

            # Time stop
            if bars_held >= self.time_stop_bars:
                print(f"🌙⏰ Time stop hit after {bars_held} bars — exiting! 🚀")
                self.position.close()
                self.entry_bar = None
                return

        # 🌙 Entry logic
        else:
            # Condition 1: RSI(5) < 30 (underbought)
            cond1 = self.rsi[-1] < self.rsi_threshold

            # Condition 2: ATR dropped at least 1% from its rolling max
            cond2 = False
            if self.atr_max[-1] > 0:
                cond2 = self.atr[-1] <= self.atr_max[-1] * (1 - self.atr_drop_pct)

            # Condition 3: Price above 200 SMA (trend filter)
            cond3 = price > self.sma200[-1]

            if cond1 and cond2 and cond3:
                # ATR-based stop loss
                stop_price = price - self.atr_stop_mult * self.atr[-1]
                risk_per_unit = price - stop_price

                if risk_per_unit > 0:
                    # Position sizing based on 1% risk
                    equity = self.equity
                    risk_amount = equity * self.risk_pct
                    position_size = int(round(risk_amount / risk_per_unit))
                    # Cap at reasonable size
                    position_size = max(1, min(position_size, int(equity / price)))

                    print(f"🌙🚀 ENTRY SIGNAL! RSI={self.rsi[-1]:.2f} | ATR={self.atr[-1]:.2f} | "
                          f"ATR_Max={self.atr_max[-1]:.2f} | Price={price:.2f} | Size={position_size}")

                    self.buy(size=position_size, sl=stop_price)
                    self.entry_bar = len(self.data) - 1
                    self.entry_atr = self.atr[-1]
                    self.entry_price = price


# 🌙 Load and clean data
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Map to proper case for backtesting.py
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

print(f"🌙 Data loaded: {len(data)} bars from {data.index[0]} to {data.index[-1]} ✨")

# 🚀 Run backtest
bt = Backtest(data, InverseVolatilityMomentum, cash=1_000_000, commission=0.001)

stats = bt.run()
print(stats)
print(stats._strategy)