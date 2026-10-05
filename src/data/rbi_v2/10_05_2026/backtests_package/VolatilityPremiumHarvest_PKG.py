import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# 🌙 Moon Dev's VolatilityPremiumHarvest Strategy 🚀
# Data path
DATA_PATH = '/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv'

# Load and clean data
print("🌙 Loading data from Moon Dev's vault...")
data = pd.read_csv(DATA_PATH)

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Map columns to backtesting.py requirements
data = data.rename(columns={
    'datetime': 'Date',
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

# Set datetime index
data['Date'] = pd.to_datetime(data['Date'])
data = data.set_index('Date')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

print(f"✨ Data loaded: {len(data)} bars from {data.index[0]} to {data.index[-1]}")


class VolatilityPremiumHarvest(Strategy):
    """
    🌙 VolatilityPremiumHarvest Strategy
    
    Since we don't have actual VIX futures data in this dataset, we proxy the 
    volatility premium using realized volatility measures on the underlying.
    
    The strategy:
    - Computes short-term realized volatility (proxy for VIX spot)
    - Computes longer-term realized volatility (proxy for VIX futures / forward expectations)
    - The "premium" = short-term RV - long-term RV baseline
    - Entry when premium exceeds mean + 1 std (elevated vol expectations → mean reversion)
    - Uses ATR-based stops and profit targets
    """
    
    # Strategy parameters
    vol_short_period = 14      # Short-term volatility window (proxy for VIX spot)
    vol_long_period = 60       # Long-term volatility window (proxy for forward expectation)
    premium_lookback = 100     # Rolling window for premium mean/std (proxy for 1-year)
    premium_std_mult = 1.0     # 1 standard deviation threshold
    atr_period = 14
    risk_pct = 0.02            # 2% risk per trade
    reward_ratio = 0.60        # Close at 60% of premium (profit target)
    stop_atr_mult = 1.5        # Stop loss at 1.5x ATR
    
    def init(self):
        print("🌙 Initializing Moon Dev's VolatilityPremiumHarvest indicators...")
        
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        
        # ATR for volatility & risk management
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)
        
        # Short-term realized volatility (proxy for spot VIX)
        # Using standard deviation of returns
        self.short_vol = self.I(
            lambda x: pd.Series(x).pct_change().rolling(self.vol_short_period).std() * np.sqrt(252),
            close
        )
        
        # Long-term realized volatility (proxy for forward vol expectation)
        self.long_vol = self.I(
            lambda x: pd.Series(x).pct_change().rolling(self.vol_long_period).std() * np.sqrt(252),
            close
        )
        
        # Volatility premium: short-term vol vs long-term baseline
        self.vol_premium = self.I(
            lambda s, l: s - l,
            self.short_vol, self.long_vol
        )
        
        # Rolling mean and std of the premium
        self.premium_mean = self.I(
            lambda x: pd.Series(x).rolling(self.premium_lookback).mean(),
            self.vol_premium
        )
        
        self.premium_std = self.I(
            lambda x: pd.Series(x).rolling(self.premium_lookback).std(),
            self.vol_premium
        )
        
        # Upper threshold (mean + 1 std)
        self.premium_upper = self.I(
            lambda m, s: m + self.premium_std_mult * s,
            self.premium_mean, self.premium_std
        )
        
        # Term structure slope proxy: long_vol - short_vol (contango when positive)
        self.term_slope = self.I(
            lambda l, s: l - s,
            self.long_vol, self.short_vol
        )
        
        # SMA for trend context
        self.sma50 = self.I(talib.SMA, close, timeperiod=50)
        
        print("✨ Indicators initialized! Ready to harvest volatility premium 🌙")
    
    def next(self):
        price = self.data.Close[-1]
        
        # Get indicator values
        if len(self.premium_upper) < 2:
            return
        
        premium = self.vol_premium[-1] if not np.isnan(self.vol_premium[-1]) else 0
        upper = self.premium_upper[-1] if not np.isnan(self.premium_upper[-1]) else 0
        mean = self.premium_mean[-1] if not np.isnan(self.premium_mean[-1]) else 0
        slope = self.term_slope[-1] if not np.isnan(self.term_slope[-1]) else 0
        atr = self.atr[-1] if not np.isnan(self.atr[-1]) else 0
        
        if atr == 0 or np.isnan(atr):
            return
        
        # ENTRY CONDITIONS
        # 1. Premium >= mean + 1 std (elevated volatility expectation)
        # 2. Contango regime (slope positive → long-term vol > short-term vol)
        # 3. VIX-like regime filter: short_vol between reasonable bounds
        entry_signal = (
            premium >= upper and
            slope > 0 and
            not np.isnan(upper) and
            upper > 0
        )
        
        # EXIT CONDITIONS
        # 1. Premium reverts to mean or below (volatility mean reversion)
        exit_signal = premium <= mean
        
        if not self.position:
            if entry_signal:
                # 🌙 Moon Dev position sizing: risk-based
                # Risk amount = 2% of equity
                risk_amount = self.equity * self.risk_pct
                stop_distance = atr * self.stop_atr_mult
                
                if stop_distance > 0:
                    position_size = risk_amount / stop_distance
                    position_size = int(round(position_size))
                    
                    if position_size > 0:
                        # Cap size to equity / price
                        max_size = int(self.equity / price)
                        position_size = min(position_size, max_size)
                        
                        if position_size > 0:
                            stop_price = price - stop_distance
                            take_profit = price + stop_distance * self.reward_ratio / (1 - self.reward_ratio) if self.reward_ratio < 1 else price + stop_distance
                            
                            print(f"🌙✨ ENTRY SIGNAL DETECTED! ✨🌙")
                            print(f"   Premium: {premium:.4f} >= Upper: {upper:.4f}")
                            print(f"   Term Slope (contango): {slope:.4f}")
                            print(f"   Price: {price:.2f}, ATR: {atr:.2f}")
                            print(f"   Position Size: {position_size} units")
                            print(f"   Stop: {stop_price:.2f}, Target: {take_profit:.2f}")
                            print(f"   🚀 Harvesting volatility premium!")
                            
                            self.buy(size=position_size)
        else:
            # Manage open position
            entry_price = self.trades[-1].entry_price
            stop_price = entry_price - atr * self.stop_atr_mult
            target_price = entry_price + atr * self.stop_atr_mult * 1.5
            
            # Exit on premium mean reversion
            if exit_signal:
                print(f"🌙 EXIT: Premium reverted to mean ({premium:.4f} <= {mean:.4f})")
                print(f"   Closing position at {price:.2f} 🎯")
                self.position.close()
            
            # Stop loss
            elif price <= stop_price:
                print(f"🌙 STOP LOSS triggered at {price:.2f} (stop: {stop_price:.2f}) 🛑")
                self.position.close()
            
            # Take profit
            elif price >= target_price:
                print(f"🌙 TAKE PROFIT hit at {price:.2f} (target: {target_price:.2f}) 💰")
                self.position.close()


# 🌙 Run the backtest
print("=" * 60)
print("🌙 Moon Dev's VolatilityPremiumHarvest Backtest 🚀")
print("=" * 60)

bt = Backtest(
    data,
    VolatilityPremiumHarvest,
    cash=1_000_000,
    commission=0.002,
    exclusive=False
)

stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙✨ Backtest complete! Moon Dev out! 🚀")