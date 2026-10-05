import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# ============================================================
# 🌙 MOON DEV BACKTEST ENGINE - VOLATILITY VERTEX STRATEGY 🚀
# ============================================================

print("🌙 Moon Dev: Loading BTC data for VolatilityVertex adaptation...")
data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

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
if 'datetime' in data.columns:
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')

print(f"✨ Moon Dev: Data loaded with {len(data)} bars 🚀")
print(f"🌙 Moon Dev: Date range: {data.index[0]} to {data.index[-1]}")

# ============================================================
# 🎯 VOLATILITY VERTEX STRATEGY CLASS
# ============================================================

class VolatilityVertex(Strategy):
    """
    🌙 VolatilityVertex Strategy
    
    Since we only have BTC data (no VIX futures), we adapt the concept:
    - Use BTC's own realized volatility term structure as a proxy for VIX futures
    - Detect "V" and inverted "V" shapes in volatility slope
    - Contrarian entries on shape inflections
    """
    
    # Strategy Parameters
    atr_period = 14
    vol_short_period = 5
    vol_mid_period = 10
    vol_long_period = 20
    inflection_threshold = 0.03  # 3% change in slope
    risk_pct = 0.01  # 1% account risk per trade
    atr_tp_mult = 1.75  # Take profit ATR multiple
    time_stop_bars = 10  # 10 trading days (using bars)
    
    def init(self):
        print("🌙 Moon Dev: Initializing VolatilityVertex indicators...")
        
        # 🎯 ATR on underlying
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 
                          timeperiod=self.atr_period)
        
        # 🎯 Realized volatility proxies (rolling std of returns) as VIX term structure proxy
        close = pd.Series(self.data.Close)
        returns = np.log(close / close.shift(1)).fillna(0)
        
        def rolling_vol(arr, period):
            s = pd.Series(arr)
            return s.rolling(period).std().fillna(method='bfill').values
        
        # Three "term structure" volatility measures (short, mid, long)
        self.vol_front = self.I(rolling_vol, returns.values, self.vol_short_period)
        self.vol_second = self.I(rolling_vol, returns.values, self.vol_mid_period)
        self.vol_third = self.I(rolling_vol, returns.values, self.vol_long_period)
        
        # 🎯 Term structure slope (front - third month)
        self.slope = self.I(lambda a, b: a - b, self.vol_front, self.vol_third)
        
        # 🎯 Rolling volume average for liquidity filter
        self.vol_avg = self.I(talib.SMA, self.data.Volume, timeperiod=20)
        
        # 🎯 RSI on volatility for divergence
        self.rsi_vol = self.I(talib.RSI, self.vol_front, timeperiod=14)
        
        # 🎯 Track slope history for shape detection
        self.slope_history = []
        
        print("✨ Moon Dev: All indicators initialized! 🚀")
    
    def _detect_v_shape(self, slopes):
        """Detect 'V' shape: down-up-down (local minimum in middle)"""
        if len(slopes) < 3:
            return False, 0
        s1, s2, s3 = slopes[-3], slopes[-2], slopes[-1]
        # V shape: negative, then positive, then negative slope change
        d1 = s2 - s1  # should be positive (going up)
        d2 = s3 - s2  # should be negative (going down)
        if d1 > 0 and d2 < 0:
            magnitude = abs(d1) + abs(d2)
            if magnitude > self.inflection_threshold:
                return True, magnitude
        return False, 0
    
    def _detect_inv_v_shape(self, slopes):
        """Detect inverted 'V' shape: up-down-up (local maximum in middle)"""
        if len(slopes) < 3:
            return False, 0
        s1, s2, s3 = slopes[-3], slopes[-2], slopes[-1]
        d1 = s2 - s1  # should be negative (going down)
        d2 = s3 - s2  # should be positive (going up)
        if d1 < 0 and d2 > 0:
            magnitude = abs(d1) + abs(d2)
            if magnitude > self.inflection_threshold:
                return True, magnitude
        return False, 0
    
    def next(self):
        # Need enough bars
        if len(self.data) < 25:
            return
        
        # Track slope
        current_slope = self.slope[-1]
        self.slope_history.append(current_slope)
        if len(self.slope_history) > 50:
            self.slope_history.pop(0)
        
        if len(self.slope_history) < 3:
            return
        
        price = self.data.Close[-1]
        atr = self.atr[-1]
        
        if np.isnan(atr) or atr == 0:
            return
        
        # 🌙 Moon Dev: Check for V shape (Long signal)
        v_shape, v_mag = self._detect_v_shape(self.slope_history)
        inv_v_shape, inv_v_mag = self._detect_inv_v_shape(self.slope_history)
        
        # Contango check: front < third (proxy: short vol < long vol)
        contango = self.vol_front[-1] < self.vol_third[-1]
        backwardation = self.vol_front[-1] > self.vol_third[-1]
        
        # Volume filter
        volume_ok = self.data.Volume[-1] > 0.5 * self.vol_avg[-1] if not np.isnan(self.vol_avg[-1]) else True
        
        # Price stabilization for long (higher low)
        price_stable = self.data.Close[-1] > self.data.Close[-2] if len(self.data) > 2 else False
        price_breakdown = self.data.Close[-1] < self.data.Close[-2] if len(self.data) > 2 else False
        
        # ============================================================
        # 🎯 ENTRY LOGIC
        # ============================================================
        
        if not self.position:
            # LONG ENTRY: V shape + contango + price stabilization + volume
            if v_shape and contango and price_stable and volume_ok:
                # Position sizing based on ATR risk
                risk_amount = self.equity * self.risk_pct
                stop_distance = 1.5 * atr
                position_size = risk_amount / stop_distance
                position_size = int(round(position_size))
                
                if position_size > 0:
                    sl = price - stop_distance
                    tp = price + (self.atr_tp_mult * atr)
                    
                    print(f"🚀 🌙 Moon Dev LONG SIGNAL! V-Shape detected (mag={v_mag:.4f})")
                    print(f"   💰 Entry: {price:.2f} | SL: {sl:.2f} | TP: {tp:.2f} | Size: {position_size}")
                    
                    self.buy(size=position_size, sl=sl, tp=tp)
            
            # SHORT ENTRY: Inverted V shape + backwardation + breakdown + volume
            elif inv_v_shape and backwardation and price_breakdown and volume_ok:
                risk_amount = self.equity * self.risk_pct
                stop_distance = 1.5 * atr
                position_size = risk_amount / stop_distance
                position_size = int(round(position_size))
                
                if position_size > 0:
                    sl = price + stop_distance
                    tp = price - (self.atr_tp_mult * atr)
                    
                    print(f"🔻 🌙 Moon Dev SHORT SIGNAL! Inverted V-Shape detected (mag={inv_v_mag:.4f})")
                    print(f"   💰 Entry: {price:.2f} | SL: {sl:.2f} | TP: {tp:.2f} | Size: {position_size}")
                    
                    self.sell(size=position_size, sl=sl, tp=tp)
        
        # ============================================================
        # 🎯 EXIT LOGIC (Time Stop & VIX spike protection)
        # ============================================================
        
        else:
            # Time stop: close after N bars
            bars_in_trade = len(self.data) - self.trades[-1].entry_bar if self.trades else 0
            
            if bars_in_trade >= self.time_stop_bars:
                print(f"⏰ 🌙 Moon Dev TIME STOP hit after {bars_in_trade} bars")
                self.position.close()
                return
            
            # Hard stop: volatility spike > 40%
            if len(self.vol_front) > 1:
                vol_change = (self.vol_front[-1] - self.vol_front[-2]) / (self.vol_front[-2] + 1e-10)
                if vol_change > 0.40:
                    print(f"🚨 🌙 Moon Dev VOL SPIKE! {vol_change*100:.1f}% - Closing all positions!")
                    self.position.close()
                    return
            
            # Mean reversion exit: VIX reverts to pre-signal level
            if self.position.is_long and self.vol_front[-1] < self.vol_third[-1] * 0.9:
                print(f"✅ 🌙 Moon Dev TP: Volatility reverted to mean")
                self.position.close()
            elif self.position.is_short and self.vol_front[-1] > self.vol_third[-1] * 1.1:
                print(f"✅ 🌙 Moon Dev TP: Volatility reverted to mean")
                self.position.close()


# ============================================================
# 🚀 RUN BACKTEST
# ============================================================

print("🌙 Moon Dev: Starting VolatilityVertex backtest... 🚀")
bt = Backtest(
    data,
    VolatilityVertex,
    cash=1_000_000,
    commission=0.002,
    exclusive=False
)

stats = bt.run()
print(stats)
print(stats._strategy)