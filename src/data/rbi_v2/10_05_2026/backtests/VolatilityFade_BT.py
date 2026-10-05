import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# 🌙 Moon Dev's VolatilityFade Backtest 🌙
# Fade overextended volatility expansions after liquidity gaps

class VolatilityFade(Strategy):
    # Strategy parameters
    bb_period = 20
    bb_std = 2.0
    bbw_lookback = 20
    bbw_threshold = 2.0
    vol_period = 20
    vol_multiplier = 2.0
    atr_period = 14
    retracement_pct = 0.05
    risk_pct = 0.02
    time_stop_bars = 20

    def init(self):
        print("🌙✨ Initializing VolatilityFade Strategy ✨🌙")
        
        # Bollinger Bands
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS,
            self.data.Close,
            timeperiod=self.bb_period,
            nbdevup=self.bb_std,
            nbdevdn=self.bb_std,
            matype=0
        )
        
        # Bollinger Bandwidth
        def compute_bbw(upper, middle, lower):
            return (upper - lower) / middle
        self.bbw = self.I(compute_bbw, self.bb_upper, self.bb_middle, self.bb_lower)
        
        # BBW mean and std
        self.bbw_mean = self.I(talib.SMA, self.bbw, timeperiod=self.bbw_lookback)
        self.bbw_std = self.I(talib.STDDEV, self.bbw, timeperiod=self.bbw_lookback)
        
        # Volume SMA
        self.vol_sma = self.I(talib.SMA, self.data.Volume, timeperiod=self.vol_period)
        
        # ATR
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period)
        
        # Track entry bar for time stop
        self.entry_bar = None
        self.entry_price = None
        self.stop_price = None
        self.target_price = None
        
        print("🌙 Indicators initialized successfully! 🚀")

    def next(self):
        # Skip if not enough data
        if len(self.data) < max(self.bb_period, self.bbw_lookback, self.vol_period, self.atr_period) + 5:
            return
        
        # Check for NaN values
        if (np.isnan(self.bbw[-1]) or np.isnan(self.bbw_mean[-1]) or 
            np.isnan(self.bbw_std[-1]) or np.isnan(self.vol_sma[-1]) or 
            np.isnan(self.atr[-1]) or np.isnan(self.bb_upper[-1])):
            return
        
        price = self.data.Close[-1]
        
        # ========== EXIT LOGIC ==========
        if self.position:
            bars_held = len(self.data) - self.entry_bar if self.entry_bar else 0
            
            # Exit 1: Volatility contraction
            bbw_contraction = self.bbw[-1] < (self.bbw_mean[-1] + 1.0 * self.bbw_std[-1])
            
            # Exit 2: Price retracement (5% in favor of short)
            retracement_hit = price <= self.entry_price * (1 - self.retracement_pct)
            
            # Exit 3: Time stop
            time_stop = bars_held >= self.time_stop_bars
            
            # Exit 4: Stop loss hit
            stop_hit = price >= self.stop_price
            
            if stop_hit:
                print(f"🛑 STOP LOSS HIT at {price:.2f} | Entry: {self.entry_price:.2f} | Stop: {self.stop_price:.2f}")
                self.position.close()
                self.entry_bar = None
            elif retracement_hit:
                print(f"🎯 TARGET HIT! Price retraced 5% to {price:.2f} | Entry: {self.entry_price:.2f}")
                self.position.close()
                self.entry_bar = None
            elif bbw_contraction:
                print(f"📉 VOLATILITY CONTRACTION EXIT at {price:.2f} | BBW: {self.bbw[-1]:.4f} < {self.bbw_mean[-1] + self.bbw_std[-1]:.4f}")
                self.position.close()
                self.entry_bar = None
            elif time_stop:
                print(f"⏰ TIME STOP EXIT at {price:.2f} after {bars_held} bars")
                self.position.close()
                self.entry_bar = None
        
        # ========== ENTRY LOGIC ==========
        if not self.position:
            # Condition A: BBW > BBW_mean + 2 * BBW_std
            bbw_expansion = self.bbw[-1] > (self.bbw_mean[-1] + self.bbw_threshold * self.bbw_std[-1])
            
            # Condition B: Volume >= 2.0 * Volume_SMA
            volume_surge = self.data.Volume[-1] >= self.vol_multiplier * self.vol_sma[-1]
            
            # Condition C: Price in upper half of bands (closing near/above upper band)
            price_upper_half = price >= self.bb_middle[-1]
            
            # Additional: price near upper band for exhaustion confirmation
            near_upper = price >= (self.bb_middle[-1] + 0.5 * (self.bb_upper[-1] - self.bb_middle[-1]))
            
            if bbw_expansion and volume_surge and price_upper_half and near_upper:
                # Calculate stop loss: recent swing high (upper band + 1 ATR buffer)
                recent_high = self.data.High[-1]
                stop_price = max(self.bb_upper[-1], recent_high) + 1.0 * self.atr[-1]
                
                # Risk-based position sizing
                risk_per_unit = stop_price - price
                if risk_per_unit <= 0:
                    return
                
                risk_amount = self.equity * self.risk_pct
                position_size = int(round(risk_amount / risk_per_unit))
                
                # Cap position size to reasonable limits
                max_size = int(self.equity / price)
                position_size = min(position_size, max_size)
                
                if position_size > 0:
                    self.entry_bar = len(self.data)
                    self.entry_price = price
                    self.stop_price = stop_price
                    self.target_price = price * (1 - self.retracement_pct)
                    
                    print(f"🌙✨ VOLATILITY FADE SHORT SIGNAL! ✨🌙")
                    print(f"   📊 Price: {price:.2f}")
                    print(f"   📈 BBW: {self.bbw[-1]:.4f} > {self.bbw_mean[-1] + self.bbw_threshold * self.bbw_std[-1]:.4f} (2σ)")
                    print(f"   📊 Volume: {self.data.Volume[-1]:.2f} >= {self.vol_multiplier * self.vol_sma[-1]:.2f} (2x surge)")
                    print(f"   🎯 Stop: {stop_price:.2f} | Target: {self.target_price:.2f}")
                    print(f"   💰 Size: {position_size} units | Risk: {self.risk_pct*100:.1f}%")
                    
                    self.sell(size=position_size)


# 🌙 Load and prepare data
print("🌙 Loading BTC-USD 15m data...")
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Rename to proper case for backtesting.py
data = data.rename(columns={
    'datetime': 'Date',
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

# Set datetime index
if 'Date' in data.columns:
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.set_index('Date')

print(f"🌙 Data loaded: {len(data)} bars")
print(f"🚀 Running VolatilityFade backtest...")

# Run backtest
bt = Backtest(data, VolatilityFade, cash=1_000_000, commission=0.001)
stats = bt.run()

print("\n" + "="*60)
print("🌙✨ VOLATILITYFADE BACKTEST RESULTS ✨🌙")
print("="*60)
print(stats)
print("\n" + "="*60)
print("🌙 Strategy Details:")
print("="*60)
print(stats._strategy)
print("\n🌙 Moon Dev out! 🚀")