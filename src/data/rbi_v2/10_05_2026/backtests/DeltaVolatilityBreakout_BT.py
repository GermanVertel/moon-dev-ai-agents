import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

print("🌙✨🚀 Moon Dev Backtest AI Initializing...")
print("🌙 Strategy: DeltaVolatility Breakout (Synthetic Proxy)")

# Load data
data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Rename to proper case for backtesting.py
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

# Ensure datetime index
if 'datetime' in data.columns:
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')

print(f"🌙 Data loaded: {len(data)} bars")
print(f"🌙 Columns: {list(data.columns)}")


class DeltaVolatilityBreakout(Strategy):
    """
    DeltaVolatility Breakout Strategy
    
    Since options chain data is not available in the CSV, we simulate the 
    "Delta-to-Volatility Ratio" (DV ratio) selection by using:
    - Delta proxy = price momentum direction (RSI-based)
    - Volatility proxy = ATR / Price (realized vol)
    - DV ratio = |momentum| / volatility
    
    The 3-legged BTS spread is proxied as a scaled directional position
    with defined risk (stop loss) and profit target (asymmetric R:R).
    """
    
    # Horizontal support detection
    swing_lookback = 20
    support_tolerance = 0.005  # 0.5% band
    min_touches = 2
    
    # Breakout confirmation
    atr_period = 14
    atr_buffer_mult = 0.3
    vol_ma_period = 20
    vol_mult = 1.5
    
    # Momentum
    rsi_period = 14
    rsi_threshold = 50
    
    # Risk management
    risk_pct = 0.015  # 1.5% of equity
    reward_ratio = 2.0  # 2:1 reward to risk (proxy for 50-75% spread profit)
    
    def init(self):
        print("🌙 Initializing indicators...")
        
        # ATR for breakout buffer and stop calculation
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close,
                          timeperiod=self.atr_period)
        
        # Volume MA
        self.vol_ma = self.I(talib.SMA, self.data.Volume, timeperiod=self.vol_ma_period)
        
        # RSI momentum
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=self.rsi_period)
        
        # Swing lows for horizontal support detection
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=self.swing_lookback)
        
        # Realized volatility proxy (for DV ratio)
        self.volatility = self.I(lambda h, l, c: (talib.ATR(h, l, c, timeperiod=14) / c) * 100,
                                 self.data.High, self.data.Low, self.data.Close)
        
        # EMA trend filter
        self.ema_fast = self.I(talib.EMA, self.data.Close, timeperiod=10)
        self.ema_slow = self.I(talib.EMA, self.data.Close, timeperiod=30)
        
        # Track state
        self.support_level = None
        self.entry_price = None
        self.stop_price = None
        self.target_price = None
        
        print("🌙✨ Indicators ready!")
    
    def detect_horizontal_support(self, idx):
        """Detect horizontal support by checking if recent lows cluster in a tight band."""
        if idx < self.swing_lookback + 5:
            return None
        
        lookback = 40
        start = max(0, idx - lookback)
        recent_lows = self.data.Low[start:idx]
        
        if len(recent_lows) < 20:
            return None
        
        # Find local swing lows
        swing_lows = []
        lows_arr = np.array(recent_lows)
        for i in range(2, len(lows_arr) - 2):
            if (lows_arr[i] < lows_arr[i-1] and lows_arr[i] < lows_arr[i-2] and
                lows_arr[i] < lows_arr[i+1] and lows_arr[i] < lows_arr[i+2]):
                swing_lows.append(lows_arr[i])
        
        if len(swing_lows) < self.min_touches:
            return None
        
        # Cluster swing lows within tolerance band
        swing_lows = sorted(swing_lows)
        best_level = None
        best_count = 0
        
        for level in swing_lows:
            touches = sum(1 for s in swing_lows 
                         if abs(s - level) / level <= self.support_tolerance)
            if touches > best_count:
                best_count = touches
                best_level = level
        
        if best_count >= self.min_touches:
            return best_level
        return None
    
    def next(self):
        price = self.data.Close[-1]
        idx = len(self.data) - 1
        
        # If in position, check exits
        if self.position:
            # Stop loss hit
            if self.data.Low[-1] <= self.stop_price:
                print(f"🌙💥 STOP LOSS hit at {self.stop_price:.2f} | Price: {price:.2f}")
                self.position.close()
                return
            
            # Profit target hit
            if self.data.High[-1] >= self.target_price:
                print(f"🌙🎯 PROFIT TARGET hit at {self.target_price:.2f} | Price: {price:.2f}")
                self.position.close()
                return
            
            # Breakout failure - close below support
            if self.support_level and price < self.support_level * (1 - self.support_tolerance):
                print(f"🌙⚠️ Breakout FAILURE - close below support {self.support_level:.2f}")
                self.position.close()
                return
            
            return
        
        # Not in position - look for entry
        if len(self.data) < 60:
            return
        
        # Detect horizontal support
        support = self.detect_horizontal_support(idx)
        if support is None:
            return
        
        self.support_level = support
        
        # Breakout trigger: close above support + ATR buffer
        atr_val = self.atr[-1]
        if np.isnan(atr_val) or atr_val <= 0:
            return
        
        breakout_trigger = support + (self.atr_buffer_mult * atr_val)
        
        # Check breakout conditions
        breakout_close = price > breakout_trigger
        
        # Volume confirmation
        vol_confirm = (not np.isnan(self.vol_ma[-1]) and 
                       self.vol_ma[-1] > 0 and 
                       self.data.Volume[-1] > self.vol_mult * self.vol_ma[-1])
        
        # Momentum confirmation: RSI > 50 and rising
        rsi_ok = (not np.isnan(self.rsi[-1]) and 
                  self.rsi[-1] > self.rsi_threshold and
                  len(self.rsi) > 2 and self.rsi[-1] > self.rsi[-2])
        
        # Trend filter: fast EMA > slow EMA
        trend_ok = (not np.isnan(self.ema_fast[-1]) and 
                    not np.isnan(self.ema_slow[-1]) and
                    self.ema_fast[-1] > self.ema_slow[-1])
        
        # DV Ratio proxy: momentum strength / volatility
        # Higher = better directional exposure per unit vol
        vol_proxy = self.volatility[-1] if not np.isnan(self.volatility[-1]) else 1.0
        momentum_proxy = abs(self.rsi[-1] - 50) if not np.isnan(self.rsi[-1]) else 0
        dv_ratio = momentum_proxy / max(vol_proxy, 0.01)
        
        # Require decent DV ratio (instrument selection quality filter)
        dv_ok = dv_ratio > 0.5
        
        if breakout_close and vol_confirm and rsi_ok and trend_ok and dv_ok:
            # Calculate position size based on risk
            stop_distance = 1.5 * atr_val
            self.stop_price = price - stop_distance
            self.target_price = price + (stop_distance * self.reward_ratio)
            
            # Risk-based sizing
            equity = self.equity
            risk_amount = equity * self.risk_pct
            position_size = int(round(risk_amount / stop_distance)) if stop_distance > 0 else 0
            
            # Cap at reasonable size
            max_size = int(equity / price)
            position_size = min(position_size, max_size)
            
            if position_size > 0:
                print(f"🌙🚀 ENTRY SIGNAL | Price: {price:.2f} | Support: {support:.2f}")
                print(f"🌙📊 DV Ratio: {dv_ratio:.3f} | RSI: {self.rsi[-1]:.1f} | Vol Mult: {self.data.Volume[-1]/self.vol_ma[-1]:.2f}x")
                print(f"🌙🛑 Stop: {self.stop_price:.2f} | 🎯 Target: {self.target_price:.2f}")
                print(f"🌙💰 Size: {position_size} units")
                
                self.buy(size=position_size)
                self.entry_price = price


# Run backtest
print("\n🌙✨🚀 Starting Backtest...")
bt = Backtest(data, DeltaVolatilityBreakout, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙✨🚀 Backtest Complete!")