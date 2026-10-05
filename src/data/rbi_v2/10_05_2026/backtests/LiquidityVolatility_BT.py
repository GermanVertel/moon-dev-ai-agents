import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# 🌙 Moon Dev's LiquidityVolatility Strategy ✨
# Since we don't have real order book heatmap data, we approximate
# liquidation clusters using swing highs/lows + volume profile
# Volatility spike confirmation uses 1m ATR vs baseline ATR

print("🌙 Moon Dev initializing LiquidityVolatility backtest... ✨")

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

# Ensure datetime index
if 'datetime' in data.columns:
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')

data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

print(f"🌙 Data loaded: {len(data)} candles ✨")
print(f"🚀 Date range: {data.index[0]} -> {data.index[-1]}")


class LiquidityVolatility(Strategy):
    """
    🌙 LiquidityVolatility Strategy ✨
    
    Approximates liquidation clusters using swing highs/lows and volume.
    Uses ATR volatility spike vs baseline as confirmation trigger.
    
    Entry:
      - Long: price breaks above recent swing high (short liquidation cluster) 
              + volatility spike + volume confirmation
      - Short: price breaks below recent swing low (long liquidation cluster)
              + volatility spike + volume confirmation
    
    Exit:
      - TP1: 1.5% move
      - TP2: 3.0% move
      - Hard SL: 0.7% adverse move
      - Time stop: 20 bars
      - Volatility decay exit
    """
    
    # Cluster detection params
    cluster_lookback = 20       # swing high/low window
    proximity_pct = 0.005       # 0.5% proximity to cluster
    
    # Volatility spike params
    atr_period = 14
    atr_baseline_period = 50
    vol_spike_mult = 2.0        # current ATR > 2x baseline
    
    # Volume confirmation
    vol_ma_period = 20
    vol_mult = 1.5
    
    # Risk management
    risk_pct = 0.005            # 0.5% risk per trade
    sl_pct = 0.007              # 0.7% hard stop
    tp1_pct = 0.015             # 1.5% TP1
    tp2_pct = 0.030             # 3.0% TP2
    time_stop_bars = 20
    cooldown_bars = 4           # ~15 min cooldown on 15m candles
    
    def init(self):
        print("🌙 Initializing indicators... ✨")
        
        # ATR for volatility spike
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close,
                          timeperiod=self.atr_period, name="ATR")
        
        # Baseline ATR (smoothed)
        self.atr_baseline = self.I(talib.SMA, self.atr, timeperiod=self.atr_baseline_period,
                                   name="ATR_baseline")
        
        # Volume MA
        self.vol_ma = self.I(talib.SMA, self.data.Volume, timeperiod=self.vol_ma_period,
                             name="Vol_MA")
        
        # Swing highs/lows = liquidation cluster proxies
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=self.cluster_lookback,
                                 name="SwingHigh")
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=self.cluster_lookback,
                                name="SwingLow")
        
        # Track trade state
        self.entry_price = None
        self.tp1_hit = False
        self.entry_bar = None
        self.last_loss_bar = -9999
        
        print("🚀 Indicators ready!")
    
    def next(self):
        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        
        # Skip if indicators not ready
        if np.isnan(self.atr[-1]) or np.isnan(self.atr_baseline[-1]) or np.isnan(self.vol_ma[-1]):
            return
        if self.atr_baseline[-1] <= 0 or self.vol_ma[-1] <= 0:
            return
        
        # === Volatility spike detection ===
        vol_spike = self.atr[-1] > (self.vol_spike_mult * self.atr_baseline[-1])
        vol_decay = self.atr[-1] < (1.2 * self.atr_baseline[-1])
        
        # === Volume confirmation ===
        volume_conf = self.data.Volume[-1] > (self.vol_mult * self.vol_ma[-1])
        
        # === Cluster levels (previous bar to avoid lookahead) ===
        cluster_high = self.swing_high[-2] if len(self.swing_high) > 1 else np.nan
        cluster_low = self.swing_low[-2] if len(self.swing_low) > 1 else np.nan
        
        if np.isnan(cluster_high) or np.isnan(cluster_low):
            return
        
        # === Cooldown check ===
        in_cooldown = (len(self.data) - self.last_loss_bar) < self.cooldown_bars
        
        # === Manage open position ===
        if self.position:
            bars_held = len(self.data) - self.entry_bar if self.entry_bar else 0
            
            if self.position.is_long:
                # Volatility decay exit
                if vol_decay:
                    print(f"🌙 Volatility decay exit (LONG) @ {price:.2f} ✨")
                    self.position.close()
                    self._reset_trade()
                    return
                
                # Time stop
                if bars_held >= self.time_stop_bars:
                    print(f"⏰ Time stop exit (LONG) @ {price:.2f}")
                    self.position.close()
                    self._reset_trade()
                    return
                
                # TP1: close 50%
                if not self.tp1_hit and price >= self.entry_price * (1 + self.tp1_pct):
                    print(f"🎯 TP1 hit (LONG) @ {price:.2f} - closing 50%")
                    self.position.close(0.5)
                    self.tp1_hit = True
                
                # TP2: close remaining
                if self.tp1_hit and price >= self.entry_price * (1 + self.tp2_pct):
                    print(f"🎯 TP2 hit (LONG) @ {price:.2f} - closing rest 🚀")
                    self.position.close()
                    self._reset_trade()
                    return
                
                # Hard stop
                if price <= self.entry_price * (1 - self.sl_pct):
                    print(f"🛑 Hard SL hit (LONG) @ {price:.2f}")
                    self.position.close()
                    self.last_loss_bar = len(self.data)
                    self._reset_trade()
                    return
            
            elif self.position.is_short:
                if vol_decay:
                    print(f"🌙 Volatility decay exit (SHORT) @ {price:.2f} ✨")
                    self.position.close()
                    self._reset_trade()
                    return
                
                if bars_held >= self.time_stop_bars:
                    print(f"⏰ Time stop exit (SHORT) @ {price:.2f}")
                    self.position.close()
                    self._reset_trade()
                    return
                
                if not self.tp1_hit and price <= self.entry_price * (1 - self.tp1_pct):
                    print(f"🎯 TP1 hit (SHORT) @ {price:.2f} - closing 50%")
                    self.position.close(0.5)
                    self.tp1_hit = True
                
                if self.tp1_hit and price <= self.entry_price * (1 - self.tp2_pct):
                    print(f"🎯 TP2 hit (SHORT) @ {price:.2f} - closing rest 🚀")
                    self.position.close()
                    self._reset_trade()
                    return
                
                if price >= self.entry_price * (1 + self.sl_pct):
                    print(f"🛑 Hard SL hit (SHORT) @ {price:.2f}")
                    self.position.close()
                    self.last_loss_bar = len(self.data)
                    self._reset_trade()
                    return
            
            return  # don't open new while in position
        
        # === Entry logic ===
        if in_cooldown:
            return
        
        if not vol_spike or not volume_conf:
            return
        
        # LONG: price breaks above cluster_high (short liquidation cluster)
        if high > cluster_high and price > cluster_high:
            # proximity check: previous close was within proximity of cluster
            prev_close = self.data.Close[-2]
            if abs(prev_close - cluster_high) / cluster_high <= self.proximity_pct * 3:
                sl_price = price * (1 - self.sl_pct)
                risk_per_unit = price - sl_price
                
                if risk_per_unit > 0:
                    risk_amount = self.equity * self.risk_pct
                    size = int(round(risk_amount / risk_per_unit))
                    size = max(1, min(size, 1000000))
                    
                    print(f"🚀 LONG entry @ {price:.2f} | cluster_high={cluster_high:.2f} | "
                          f"ATR_spike={self.atr[-1]/self.atr_baseline[-1]:.2f}x | size={size} 🌙")
                    self.buy(size=size, sl=sl_price)
                    self.entry_price = price
                    self.entry_bar = len(self.data)
                    self.tp1_hit = False
        
        # SHORT: price breaks below cluster_low (long liquidation cluster)
        elif low < cluster_low and price < cluster_low:
            prev_close = self.data.Close[-2]
            if abs(prev_close - cluster_low) / cluster_low <= self.proximity_pct * 3:
                sl_price = price * (1 + self.sl_pct)
                risk_per_unit = sl_price - price
                
                if risk_per_unit > 0:
                    risk_amount = self.equity * self.risk_pct
                    size = int(round(risk_amount / risk_per_unit))
                    size = max(1, min(size, 1000000))
                    
                    print(f"🔻 SHORT entry @ {price:.2f} | cluster_low={cluster_low:.2f} | "
                          f"ATR_spike={self.atr[-1]/self.atr_baseline[-1]:.2f}x | size={size} 🌙")
                    self.sell(size=size, sl=sl_price)
                    self.entry_price = price
                    self.entry_bar = len(self.data)
                    self.tp1_hit = False
    
    def _reset_trade(self):
        self.entry_price = None
        self.entry_bar = None
        self.tp1_hit = False


print("🌙 Running backtest... ✨🚀")
bt = Backtest(data, LiquidityVolatility, cash=1_000_000, commission=0.001)

stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙✨ Backtest complete! 🚀")