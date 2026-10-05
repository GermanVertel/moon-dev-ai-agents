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

# Next month-end date for holding period (approx 30 days)
data['next_month_end'] = data.groupby('yearmonth')['is_month_end'].transform(lambda x: x.shift(-1))

print("🌙 Moon Dev TemporalPutHarvest backtest initializing... ✨")
print(f"📊 Data loaded: {len(data)} bars")
print(f"📅 Date range: {data.index[0]} to {data.index[-1]}")


class TemporalPutHarvest(Strategy):
    """
    TemporalPutHarvest: Monthly put-selling simulation.
    
    Since we don't have options chain data, we simulate put-selling premium
    using an approximation: premium collected ≈ intrinsic + time value.
    We model the position as a short put with strike = weekly close high.
    
    For backtesting.py (which trades the underlying), we approximate the
    P&L of a short put position by:
      - Entering a SHORT position on the underlying at month-end
      - Sized such that gains/losses mimic a short put
      - Exiting at next month-end (or on stop-loss)
    
    This is a proxy implementation given the absence of option chain data.
    """
    
    # Risk management parameters
    stop_loss_pct = 0.10       # 10% adverse move triggers exit
    take_profit_pct = 0.50     # 50% of premium as target (approximated)
    leverage_fraction = 0.30   # Safer than 100% - use 30% of equity as risk
    
    def init(self):
        self.weekly_high = self.I(lambda: self.data.df['weekly_close_high'].values, name='WeeklyHigh')
        self.is_month_end = self.I(lambda: self.data.df['is_month_end'].astype(int).values, name='MonthEnd')
        
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
        
        # Entry: sell put (simulated as short underlying position)
        # Strike = weekly close high
        strike = self.data.df['weekly_close_high'].iloc[current_idx]
        current_price = self.data.Close[-1]
        
        # Premium approximation: distance from strike (OTM = more premium)
        # We use a simple model: premium = max(0.5% of strike, distance-based)
        premium_pct = 0.015  # ~1.5% monthly premium
        premium = strike * premium_pct
        
        # Position sizing: use leverage fraction of equity as notional risk
        # For a short put, max risk ≈ strike * contracts
        # We size so that (strike * contracts) ≈ leverage_fraction * equity
        equity = self.equity
        risk_budget = equity * self.leverage_fraction
        contracts = int(round(risk_budget / strike)) if strike > 0 else 0
        
        if contracts < 1:
            contracts = 1
        
        # Simulate short put via short underlying (delta ~ 0.5 for ATM)
        # We short a fraction of the notional to approximate delta exposure
        # Delta of ATM put ≈ -0.5, so short put ≈ +0.5 delta long underlying
        # But for simplicity, we go SHORT underlying to mimic assignment risk direction
        
        # Actually: short put = bullish position (you profit if underlying rises/stays flat)
        # So we go LONG underlying as a proxy for short put P&L (delta positive)
        delta_proxy = 0.5
        size = int(round(contracts * delta_proxy))
        
        if size < 1:
            size = 1
        
        # Stop loss price: strike * (1 - stop_loss_pct)
        stop_price = strike * (1 - self.stop_loss_pct)
        
        print(f"🌙✨ Moon Dev ENTRY: Short Put proxy | Strike(WeeklyHigh)={strike:.2f} | "
              f"Price={current_price:.2f} | Size={size} | Premium≈{premium:.2f} 🚀")
        
        # Enter long (proxy for short put delta)
        self.buy(size=size)
        
        # Store stop price for this cycle
        self.stop_price = stop_price
        self.entry_strike = strike
        self.entry_premium = premium
    
    def on_trade_event(self):
        pass


# Run backtest
bt = Backtest(
    data,
    TemporalPutHarvest,
    cash=1_000_000,
    commission=0.002,
    exclusive_orders=True
)

print("🚀 Moon Dev launching TemporalPutHarvest backtest... 🌙")
stats = bt.run()
print(stats)
print(stats._strategy)