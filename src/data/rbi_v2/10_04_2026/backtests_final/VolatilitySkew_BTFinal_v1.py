import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Load and prepare data
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

print("🌙 Moon Dev VolatilitySkew Backtest - Loading data... ✨")
print(f"📊 Data shape: {data.shape}")
print(f"📅 Date range: {data.index[0]} to {data.index[-1]}")


class VolatilitySkew(Strategy):
    """
    VolatilitySkew Strategy - Adapted for equity backtesting.py
    """
    
    # Strategy parameters
    iv_lookback = 20
    iv_threshold_pct = 70
    profit_target_pct = 0.50
    stop_loss_pct = 0.50
    time_exit_bars = 32
    risk_pct = 0.02
    
    def init(self):
        print("🌙 Initializing VolatilitySkew indicators... ✨")
        
        # Volatility proxy (like IV) - ATR based
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14)
        
        # Realized volatility (rolling std of returns) as IV proxy
        close_series = pd.Series(self.data.Close)
        returns = close_series.pct_change().fillna(0)
        self.realized_vol = self.I(
            lambda s: pd.Series(s).rolling(self.iv_lookback).std().values,
            returns.values,
            name='RealizedVol'
        )
        
        # IV percentile rank
        def iv_rank(vol_series):
            vol_series = pd.Series(vol_series)
            return vol_series.rolling(100).apply(
                lambda x: (x.iloc[-1] > x).mean() * 100 if len(x) > 0 else 50
            ).values
        
        self.iv_rank = self.I(iv_rank, self.realized_vol, name='IV_Rank')
        
        # Trend filter - SMA
        self.sma_fast = self.I(talib.SMA, self.data.Close, timeperiod=10)
        self.sma_slow = self.I(talib.SMA, self.data.Close, timeperiod=50)
        
        # RSI for downside momentum
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=14)
        
        # Track entry info
        self.entry_price_val = None
        self.entry_bar = None
        self.stop_price = None
        self.target_price = None
        
        print("🚀 VolatilitySkew indicators ready!")
    
    def next(self):
        # Skip if not enough data
        if len(self.data) < 100:
            return
        
        # Check for existing position
        if self.position:
            self._manage_position()
            return
        
        # Entry logic
        current_iv_rank = self.iv_rank[-1]
        current_price = self.data.Close[-1]
        current_rsi = self.rsi[-1]
        
        # Skip NaN values
        if (np.isnan(current_iv_rank) or np.isnan(current_rsi) or 
            np.isnan(self.sma_fast[-1]) or np.isnan(self.sma_slow[-1]) or
            np.isnan(self.atr[-1])):
            return
        
        # Signal: High IV + bearish momentum + bearish trend
        high_iv = current_iv_rank > self.iv_threshold_pct
        bearish_momentum = current_rsi < 45
        trend_bearish = self.sma_fast[-1] < self.sma_slow[-1]
        
        if high_iv and bearish_momentum and trend_bearish:
            # Use fractional size (percentage of equity) - valid for backtesting.py
            size = 0.95
            
            atr_val = self.atr[-1]
            if atr_val > 0:
                stop_price = current_price + (atr_val * 2)
                target_price = current_price - (atr_val * 3)
                
                print(f"🌙✨ VolatilitySkew SIGNAL! IV Rank: {current_iv_rank:.1f}% | RSI: {current_rsi:.1f} | Price: {current_price:.2f}")
                print(f"🚀 Entering bearish position (proxy for ATM put) | Size: {size}")
                
                self.sell(size=size)
                self.entry_price_val = current_price
                self.entry_bar = len(self.data)
                self.stop_price = stop_price
                self.target_price = target_price
    
    def _manage_position(self):
        """Manage existing position with exit rules"""
        current_price = self.data.Close[-1]
        bars_held = len(self.data) - self.entry_bar if self.entry_bar else 0
        
        # For short position (proxy for put)
        if self.position.is_short:
            if self.target_price is not None and current_price <= self.target_price:
                print(f"🎯 PROFIT TARGET HIT! Price: {current_price:.2f} | Target: {self.target_price:.2f}")
                self.position.close()
                return
            
            if self.stop_price is not None and current_price >= self.stop_price:
                print(f"🛑 STOP LOSS HIT! Price: {current_price:.2f} | Stop: {self.stop_price:.2f}")
                self.position.close()
                return
            
            if bars_held >= self.time_exit_bars:
                print(f"⏰ TIME EXIT! Held {bars_held} bars | Price: {current_price:.2f}")
                self.position.close()
                return
        
        # For long position (if any)
        elif self.position.is_long:
            if self.entry_price_val:
                pnl_pct = (current_price - self.entry_price_val) / self.entry_price_val
                
                if pnl_pct >= self.profit_target_pct:
                    print(f"🎯 PROFIT TARGET HIT! PnL: {pnl_pct*100:.1f}%")
                    self.position.close()
                    return
                
                if pnl_pct <= -self.stop_loss_pct:
                    print(f"🛑 STOP LOSS HIT! PnL: {pnl_pct*100:.1f}%")
                    self.position.close()
                    return
                
                if bars_held >= self.time_exit_bars:
                    print(f"⏰ TIME EXIT! Held {bars_held} bars")
                    self.position.close()
                    return


# Run backtest
print("\n🌙✨🚀 Starting VolatilitySkew Backtest... 🚀✨🌙\n")

bt = Backtest(
    data,
    VolatilitySkew,
    cash=10000000,
    commission=0.002,
    exclusive_orders=True
)

stats = bt.run()
print("\n" + "="*60)
print("🌙 MOON DEV VOLATILITYSKEW BACKTEST RESULTS 🌙")
print("="*60)
print(stats)
print("\n" + "="*60)
print("📊 STRATEGY DETAILS 📊")
print("="*60)
print(stats._strategy)
print("\n🌙 Backtest complete! ✨🚀")