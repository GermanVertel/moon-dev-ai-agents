import pandas as pd
import numpy as np
import talib
from backtesting import Backtest, Strategy

print("🌙 Moon Dev Backtest AI initializing... ✨")
print("🚀 Loading OversoldContrarian strategy...")

# Load data
data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
print(f"📊 Loading data from: {data_path}")

data = pd.read_csv(data_path)

# Clean column names
data.columns = data.columns.str.strip().str.lower()
print("🧹 Cleaning column names...")

# Drop unnamed columns
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
print("🗑️ Dropped unnamed columns")

# Ensure proper column mapping
data.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')

# Rename to match backtesting requirements
data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

print(f"✅ Data loaded: {len(data)} rows")
print(f"📅 Date range: {data.index[0]} to {data.index[-1]}")
print("🌙 Moon Dev data preparation complete! ✨")


class OversoldContrarian(Strategy):
    """
    OversoldContrarian Strategy 🌙
    
    Entry: RSI drops below 30 within trailing 7-day window
    Exit: Profit target based on new lows OR stop loss
    """
    
    # Strategy parameters
    rsi_period = 14
    rsi_oversold = 30
    rsi_recovery = 50
    lookback_days = 7
    stop_loss_pct = 0.05  # 5% stop loss
    take_profit_pct = 0.10  # 10% take profit
    
    def init(self):
        print("🌙 Initializing OversoldContrarian indicators...")
        
        # RSI indicator using talib
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=self.rsi_period)
        
        # Track RSI oversold condition over lookback period
        self.rsi_oversold_recent = self.I(
            lambda: pd.Series(self.rsi, index=self.data.index).rolling(
                window=self.lookback_days * 96  # 96 bars per day on 15m
            ).min(),
            name='RSI_Min_Lookback'
        )
        
        # Track recent low for new low detection
        self.recent_low = self.I(
            lambda: pd.Series(self.data.Low, index=self.data.index).rolling(
                window=20
            ).min(),
            name='Recent_Low'
        )
        
        # Track entry price reference
        self.entry_price = None
        self.entry_low = None
        
        print("✨ Indicators initialized successfully!")
        print(f"📊 RSI Period: {self.rsi_period}")
        print(f"📉 Oversold Threshold: {self.rsi_oversold}")
        print(f"📈 Recovery Threshold: {self.rsi_recovery}")
    
    def next(self):
        # Skip if not enough data
        if len(self.data) < self.rsi_period + 10:
            return
        
        current_price = self.data.Close[-1]
        current_rsi = self.rsi[-1]
        rsi_min_lookback = self.rsi_oversold_recent[-1]
        
        # Entry Logic
        if not self.position:
            # Check if RSI dropped below 30 within lookback window
            if not np.isnan(rsi_min_lookback) and rsi_min_lookback < self.rsi_oversold:
                # Additional entry confirmation: current RSI should be recovering
                if current_rsi > self.rsi_oversold:
                    # Calculate position size
                    position_size = int(round(1000000 * 0.02))  # 2% of 1M portfolio
                    
                    print(f"🌙✨ ENTRY SIGNAL DETECTED! ✨🌙")
                    print(f"📉 RSI Min (7-day): {rsi_min_lookback:.2f}")
                    print(f"📈 Current RSI: {current_rsi:.2f}")
                    print(f"💰 Entry Price: ${current_price:.2f}")
                    print(f"🎯 Position Size: {position_size}")
                    print(f"🚀 Entering LONG position (contrarian play)...")
                    
                    self.buy(size=position_size)
                    self.entry_price = current_price
                    self.entry_low = current_price
        
        # Exit Logic
        else:
            # Calculate P&L
            pnl_pct = (current_price - self.entry_price) / self.entry_price
            
            # Track new lows
            if current_price < self.entry_low:
                self.entry_low = current_price
            
            # Exit conditions:
            # 1. Take profit reached
            if pnl_pct >= self.take_profit_pct:
                print(f"🌙💰 TAKE PROFIT HIT! 💰🌙")
                print(f"📈 Profit: {pnl_pct*100:.2f}%")
                print(f"💵 Exit Price: ${current_price:.2f}")
                self.position.close()
                self.entry_price = None
                self.entry_low = None
            
            # 2. Stop loss hit
            elif pnl_pct <= -self.stop_loss_pct:
                print(f"🌙🛑 STOP LOSS TRIGGERED! 🛑🌙")
                print(f"📉 Loss: {pnl_pct*100:.2f}%")
                print(f"💵 Exit Price: ${current_price:.2f}")
                self.position.close()
                self.entry_price = None
                self.entry_low = None
            
            # 3. RSI recovery above 50 without new lows
            elif current_rsi > self.rsi_recovery and current_price >= self.entry_low:
                print(f"🌙⚠️ RSI RECOVERY EXIT! ⚠️🌙")
                print(f"📈 RSI recovered to: {current_rsi:.2f}")
                print(f"💵 Exit Price: ${current_price:.2f}")
                self.position.close()
                self.entry_price = None
                self.entry_low = None


# Run backtest
print("\n🌙✨🚀 Starting Moon Dev Backtest... 🚀✨🌙")
print("=" * 60)

bt = Backtest(
    data,
    OversoldContrarian,
    cash=1000000,
    commission=0.002,
    exclusive=False
)

stats = bt.run()

print("\n" + "=" * 60)
print("🌙✨ MOON DEV BACKTEST RESULTS ✨🌙")
print("=" * 60)
print(stats)
print("\n" + "=" * 60)
print("🌙 STRATEGY DETAILS 🌙")
print("=" * 60)
print(stats._strategy)
print("\n🌙✨ Moon Dev Backtest Complete! ✨🌙")