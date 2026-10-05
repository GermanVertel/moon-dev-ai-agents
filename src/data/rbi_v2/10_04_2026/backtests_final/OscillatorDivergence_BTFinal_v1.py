import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# 🌙 Moon Dev's OscillatorDivergence Backtest 🚀
print("🌙✨ Initializing Moon Dev's OscillatorDivergence Strategy ✨🌙")

# Load and clean data
data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
print(f"🚀 Loading data from: {data_path}")
data = pd.read_csv(data_path)

# Clean column names
data.columns = data.columns.str.strip().str.lower()
print(f"🌙 Columns after cleaning: {list(data.columns)}")

# Drop unnamed columns
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
if 'datetime' in data.columns:
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')

print(f"✨ Data shape: {data.shape}")
print(f"🌙 Data head:\n{data.head()}")


class OscillatorDivergence(Strategy):
    """
    🌙 Moon Dev's OscillatorDivergence Strategy 🚀
    
    Mean-reversion SHORT setup exploiting divergence between:
    - RSI(20) vs its 20-period SMA
    - Stochastic(8,20) vs its 20-period SMA
    
    Entry (SHORT): RSI > SMA(RSI) AND Stoch < SMA(Stoch)
    Exit: Stoch > SMA(Stoch) OR RSI < SMA(RSI)
    """
    
    # Strategy parameters
    rsi_period = 20
    rsi_ma_period = 20
    stoch_k_period = 8
    stoch_d_period = 20
    stoch_ma_period = 20
    
    # Risk management
    risk_pct = 0.02  # 2% risk per trade
    atr_period = 14
    atr_multiplier = 2.0
    max_hold_bars = 50  # Time-based stop
    
    def init(self):
        print("🌙✨ Initializing indicators for OscillatorDivergence ✨🌙")
        
        # RSI(20)
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=self.rsi_period, name='RSI(20)')
        
        # SMA of RSI(20)
        self.rsi_ma = self.I(talib.SMA, self.rsi, timeperiod=self.rsi_ma_period, name='SMA(RSI,20)')
        
        # Stochastic %K and %D - wrap talib.STOCH with a helper that returns both arrays
        def stoch_func(high, low, close):
            slowk, slowd = talib.STOCH(
                high, low, close,
                fastk_period=self.stoch_k_period,
                slowk_period=3,
                slowk_matype=0,
                slowd_period=self.stoch_d_period,
                slowd_matype=0
            )
            return slowk
        
        self.stoch_k = self.I(
            stoch_func,
            self.data.High, self.data.Low, self.data.Close,
            name='Stoch K(8,20)'
        )
        
        # SMA of Stochastic %K
        self.stoch_ma = self.I(talib.SMA, self.stoch_k, timeperiod=self.stoch_ma_period, name='SMA(Stoch,20)')
        
        # ATR for volatility-based stops
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period, name='ATR')
        
        # Longer-term trend filter - 200-period MA
        self.trend_ma = self.I(talib.SMA, self.data.Close, timeperiod=200, name='SMA(200)')
        
        # Track entry bar for time-based stop
        self.entry_bar = None
        
        print("🚀 All indicators initialized! 🌙")
    
    def next(self):
        # Skip if not enough data
        if len(self.data) < 200:
            return
        
        # Get current values
        price = self.data.Close[-1]
        rsi_val = self.rsi[-1]
        rsi_ma_val = self.rsi_ma[-1]
        stoch_val = self.stoch_k[-1]
        stoch_ma_val = self.stoch_ma[-1]
        atr_val = self.atr[-1]
        trend_ma_val = self.trend_ma[-1]
        
        # Skip if any indicator is NaN
        if any(np.isnan([rsi_val, rsi_ma_val, stoch_val, stoch_ma_val, atr_val, trend_ma_val])):
            return
        
        # ============================================
        # 🌙 EXIT LOGIC (when in a position)
        # ============================================
        if self.position:
            # Time-based stop
            bars_held = len(self.data) - self.entry_bar if self.entry_bar else 0
            
            # Exit 1: Stochastic crosses back above its MA
            stoch_reversion = stoch_val > stoch_ma_val
            
            # Exit 2: RSI crosses back below its MA
            rsi_reversion = rsi_val < rsi_ma_val
            
            # Time stop
            time_stop = bars_held >= self.max_hold_bars
            
            if stoch_reversion:
                print(f"🌙✨ EXIT SIGNAL: Stoch({stoch_val:.2f}) > Stoch MA({stoch_ma_val:.2f}) - Fast oscillator re-aligned! 🚀")
                self.position.close()
                self.entry_bar = None
            elif rsi_reversion:
                print(f"🌙✨ EXIT SIGNAL: RSI({rsi_val:.2f}) < RSI MA({rsi_ma_val:.2f}) - Slow oscillator confirmed! 🚀")
                self.position.close()
                self.entry_bar = None
            elif time_stop:
                print(f"🌙⏰ TIME STOP: Held {bars_held} bars - divergence did not resolve! 🚀")
                self.position.close()
                self.entry_bar = None
            return
        
        # ============================================
        # 🌙 ENTRY LOGIC (SHORT)
        # ============================================
        
        # Condition A: RSI > its 20-period MA
        condition_a = rsi_val > rsi_ma_val
        
        # Condition B: Stochastic < its 20-period MA
        condition_b = stoch_val < stoch_ma_val
        
        # Trend filter: avoid shorts when price is strongly above 200 MA (strong uptrend)
        trend_filter = price < trend_ma_val * 1.05  # Allow slightly above
        
        if condition_a and condition_b and trend_filter:
            print(f"🌙🚀 SHORT ENTRY SIGNAL DETECTED! 🚀🌙")
            print(f"   📊 RSI({rsi_val:.2f}) > SMA RSI({rsi_ma_val:.2f}) ✓")
            print(f"   📊 Stoch({stoch_val:.2f}) < SMA Stoch({stoch_ma_val:.2f}) ✓")
            print(f"   💰 Price: ${price:.2f} | ATR: {atr_val:.2f}")
            
            # Calculate position size based on risk
            # Risk per trade = risk_pct of equity
            # Stop distance = ATR * multiplier
            equity = self.equity
            risk_amount = equity * self.risk_pct
            stop_distance = atr_val * self.atr_multiplier
            
            if stop_distance > 0:
                # Position size as a fraction of equity (0-1)
                # For backtesting.py, size must be int or fraction 0-1
                # Since we're shorting, size is in units
                position_size = risk_amount / stop_distance
                position_size = int(round(position_size))
                
                # Ensure at least 1 unit
                if position_size < 1:
                    position_size = 1
                
                # Cap at reasonable fraction of equity
                max_size = int(equity / price)
                if position_size > max_size:
                    position_size = max_size
                
                if position_size > 0:
                    # Calculate stop loss and take profit prices
                    stop_price = price + stop_distance
                    tp_price = price - (stop_distance * 1.5)  # 1.5:1 R:R
                    
                    print(f"   🎯 Position Size: {position_size} units")
                    print(f"   🛑 Stop Loss: ${stop_price:.2f} | 🎯 Take Profit: ${tp_price:.2f}")
                    
                    self.sell(size=position_size, sl=stop_price, tp=tp_price)
                    self.entry_bar = len(self.data)
                else:
                    print("   ⚠️ Position size too small, skipping entry")
        else:
            # Debug: show why no entry
            if condition_a and not condition_b:
                pass  # RSI condition met but Stoch not
            elif condition_b and not condition_a:
                pass  # Stoch condition met but RSI not


# 🌙 Run the backtest
print("\n" + "="*60)
print("🌙🚀 Moon Dev's OscillatorDivergence Backtest Starting! 🚀🌙")
print("="*60 + "\n")

bt = Backtest(
    data,
    OscillatorDivergence,
    cash=1_000_000,
    commission=0.002
)

stats = bt.run()
print(stats)
print(stats._strategy)
print("\n🌙✨ Backtest complete! Moon Dev out! 🚀🌙")