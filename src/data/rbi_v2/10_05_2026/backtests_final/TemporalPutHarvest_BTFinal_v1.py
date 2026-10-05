import pandas as pd
import numpy as np
import talib
from backtesting import Backtest, Strategy
from datetime import datetime

# Load data
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Ensure proper column mapping
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
data = data.sort_index()

# Add weekly high and month-end flags
data['week'] = data.index.isocalendar().week
data['year'] = data.index.year
data['month'] = data.index.month
data['dayofweek'] = data.index.dayofweek

# Weekly closing high: highest close within each (year, week)
weekly_high = data.groupby(['year', 'week'])['Close'].transform('max')
data['weekly_close_high'] = weekly_high

# Determine last trading day of each month
data['yearmonth'] = data.index.to_period('M')
last_day_of_month = data.groupby('yearmonth')['Close'].transform(lambda x: x.index.max())
data['is_month_end'] = data.index == last_day_of_month

print("🌙 Moon Dev TemporalPutHarvest backtest initializing... ✨")
print(f"📊 Data loaded: {len(data)} bars")
print(f"📅 Date range: {data.index[0]} to {data.index[-1]}")


class TemporalPutHarvest(Strategy):
    """
    TemporalPutHarvest: Monthly put-selling simulation.
    """
    
    # Risk management parameters
    stop_loss_pct = 0.10
    take_profit_pct = 0.50
    leverage_fraction = 0.30
    
    def init(self):
        # 🌙 Moon Dev: Wrap indicators in self.I() properly
        self.weekly_high = self.I(
            lambda: self.data.df['weekly_close_high'].values,
            name='WeeklyHigh'
        )
        self.is_month_end = self.I(
            lambda: self.data.df['is_month_end'].astype(int).values,
            name='MonthEnd'
        )
        # 🌙 Moon Dev: RSI & ATR indicators
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=14, name='RSI')
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14, name='ATR')
        
    def next(self):
        # Only act on month-end bars
        if len(self.data) < 2:
            return
        
        current_idx = len(self.data) - 1
        is_me = self.data.df['is_month_end'].iloc[current_idx]
        
        if not is_me:
            return
        
        # Check if we already have a position - close it (roll)
        if self.position:
            self.position.close()
            print(f"🔄 Moon Dev ROLL: Closed previous put position at {self.data.Close[-1]:.2f} 🌙")
        
        # Entry: sell put (simulated as long underlying position)
        strike = self.data.df['weekly_close_high'].iloc[current_idx]
        current_price = self.data.Close[-1]
        
        premium_pct = 0.015
        premium = strike * premium_pct
        
        equity = self.equity
        risk_budget = equity * self.leverage_fraction
        contracts = int(round(risk_budget / strike)) if strike > 0 else 0
        
        if contracts < 1:
            contracts = 1
        
        delta_proxy = 0.5
        size = int(round(contracts * delta_proxy))
        
        if size < 1:
            size = 1
        
        stop_price = strike * (1 - self.stop_loss_pct)
        
        print(f"🌙✨ Moon Dev ENTRY: Short Put proxy | Strike(WeeklyHigh)={strike:.2f} | "
              f"Price={current_price:.2f} | Size={size} | Premium≈{premium:.2f} 🚀")
        
        # 🌙 Moon Dev: Entry with stop-loss via sl= parameter
        self.buy(size=size, sl=stop_price)
        
        self.stop_price = stop_price
        self.entry_strike = strike
        self.entry_premium = premium


# Run backtest
bt = Backtest(
    data,
    TemporalPutHarvest,
    cash=1_000_000,
    commission=0.002,
    exclusive_orders=True,
    finalize_trades=True
)

print("🚀 Moon Dev launching TemporalPutHarvest backtest... 🌙")
stats = bt.run()
print(stats)
print(stats._strategy)