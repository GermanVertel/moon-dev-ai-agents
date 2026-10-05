import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Load data
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Ensure proper column mapping
data.columns = [col.capitalize() for col in data.columns]

# Set datetime index
if 'Datetime' in data.columns:
    data['Datetime'] = pd.to_datetime(data['Datetime'])
    data.set_index('Datetime', inplace=True)

print("🌙 Moon Dev Data Loaded! ✨")
print(f"📊 Data shape: {data.shape}")
print(f"📈 Columns: {list(data.columns)}")


class ConvergentVolatility(Strategy):
    # Strategy parameters
    fast_ema_period = 10
    slow_ema_period = 50
    rsi_period = 14
    atr_period = 14
    vol_ma_period = 20
    bb_period = 20
    bb_std = 2.0
    squeeze_lookback = 50
    squeeze_threshold = 0.5
    vol_spike_mult = 1.5
    rsi_long_threshold = 55
    rsi_short_threshold = 45
    risk_pct = 0.02
    
    def init(self):
        print("🚀 Moon Dev ConvergentVolatility Strategy Initializing... 🌙")
        
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume
        
        # Moving averages
        self.fast_ema = self.I(talib.EMA, close, timeperiod=self.fast_ema_period)
        self.slow_ema = self.I(talib.EMA, close, timeperiod=self.slow_ema_period)
        
        # RSI
        self.rsi = self.I(talib.RSI, close, timeperiod=self.rsi_period)
        
        # ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)
        
        # Volume MA
        self.vol_ma = self.I(talib.SMA, volume, timeperiod=self.vol_ma_period)
        
        # Bollinger Bands for squeeze confirmation
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS, close, timeperiod=self.bb_period,
            nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0
        )
        
        # MA spread (absolute distance between fast and slow EMA)
        self.ma_spread = self.I(lambda f, s: np.abs(f - s), self.fast_ema, self.slow_ema)
        
        # Average MA spread over lookback period
        self.avg_spread = self.I(talib.SMA, self.ma_spread, timeperiod=self.squeeze_lookback)
        
        # Consolidation range high/low (rolling)
        self.range_high = self.I(talib.MAX, high, timeperiod=self.squeeze_lookback)
        self.range_low = self.I(talib.MIN, low, timeperiod=self.squeeze_lookback)
        
        # Track trade state
        self.entry_price = None
        self.stop_price = None
        self.entry_bar = None
        self.trail_active = False
        
        print("✨ Indicators ready! Let's ride the volatility wave! 🌊")
    
    def _is_squeeze(self, idx):
        """Detect volatility squeeze: MA spread compressed below threshold."""
        if idx < self.squeeze_lookback:
            return False
        avg = self.avg_spread[idx]
        spread = self.ma_spread[idx]
        if avg is None or spread is None or np.isnan(avg) or np.isnan(spread):
            return False
        if avg <= 0:
            return False
        return spread < (self.squeeze_threshold * avg)
    
    def _bullish_divergence(self, idx):
        """Check for bullish RSI divergence: price lower low, RSI higher low."""
        if idx < 20:
            return False
        close = self.data.Close
        rsi = self.rsi
        # Compare recent 10-bar window vs prior 10-bar window
        recent_low = np.min(close[idx-9:idx+1])
        prior_low = np.min(close[idx-19:idx-9])
        recent_rsi_low = np.min(rsi[idx-9:idx+1])
        prior_rsi_low = np.min(rsi[idx-19:idx-9])
        return (recent_low <= prior_low) and (recent_rsi_low > prior_rsi_low)
    
    def _bearish_divergence(self, idx):
        """Check for bearish RSI divergence: price higher high, RSI lower high."""
        if idx < 20:
            return False
        close = self.data.Close
        rsi = self.rsi
        recent_high = np.max(close[idx-9:idx+1])
        prior_high = np.max(close[idx-19:idx-9])
        recent_rsi_high = np.max(rsi[idx-9:idx+1])
        prior_rsi_high = np.max(rsi[idx-19:idx-9])
        return (recent_high >= prior_high) and (recent_rsi_high < prior_rsi_high)
    
    def _converging(self, idx):
        """Check if MA spread is narrowing for 2+ consecutive bars."""
        if idx < 3:
            return False
        s0 = self.ma_spread[idx]
        s1 = self.ma_spread[idx-1]
        s2 = self.ma_spread[idx-2]
        if any(np.isnan(x) for x in [s0, s1, s2]):
            return False
        return (s0 < s1) and (s1 < s2)
    
    def _rsi_neutral(self, idx):
        """RSI back in neutral zone 45-55."""
        r = self.rsi[idx]
        if r is None or np.isnan(r):
            return False
        return 45 <= r <= 55
    
    def next(self):
        idx = len(self.data) - 1
        if idx < self.squeeze_lookback + 5:
            return
        
        price = self.data.Close[-1]
        rsi_val = self.rsi[-1]
        
        # === MANAGE OPEN POSITION ===
        if self.position:
            # Time stop: 15 bars without 1x risk profit
            if self.entry_bar is not None and (idx - self.entry_bar) >= 15:
                if self.entry_price and self.stop_price:
                    risk = abs(self.entry_price - self.stop_price)
                    if self.position.is_long:
                        if price < (self.entry_price + risk):
                            print(f"⏰ Moon Dev Time Stop (Long) at {price:.2f} 🌙")
                            self.position.close()
                            self._reset_trade()
                            return
                    else:
                        if price > (self.entry_price - risk):
                            print(f"⏰ Moon Dev Time Stop (Short) at {price:.2f} 🌙")
                            self.position.close()
                            self._reset_trade()
                            return
            
            # Trailing stop activation
            if self.entry_price and self.stop_price and not self.trail_active:
                risk = abs(self.entry_price - self.stop_price)
                if self.position.is_long and price >= self.entry_price + 1.5 * risk:
                    self.trail_active = True
                    print(f"🎯 Moon Dev Trailing Stop Activated (Long) 🚀")
                elif self.position.is_short and price <= self.entry_price - 1.5 * risk:
                    self.trail_active = True
                    print(f"🎯 Moon Dev Trailing Stop Activated (Short) 🚀")
            
            # Update trailing stop using fast EMA
            if self.trail_active:
                if self.position.is_long:
                    new_stop = self.fast_ema[-1]
                    if new_stop > self.stop_price:
                        self.stop_price = new_stop
                else:
                    new_stop = self.fast_ema[-1]
                    if new_stop < self.stop_price:
                        self.stop_price = new_stop
            
            # Stop loss hit
            if self.position.is_long and price <= self.stop_price:
                print(f"🛑 Moon Dev Stop Loss (Long) at {price:.2f} 💔")
                self.position.close()
                self._reset_trade()
                return
            if self.position.is_short and price >= self.stop_price:
                print(f"🛑 Moon Dev Stop Loss (Short) at {price:.2f} 💔")
                self.position.close()
                self._reset_trade()
                return
            
            # === CONVERGENCE EXIT ===
            if self._converging(idx) and self._rsi_neutral(idx):
                if self.position.is_long:
                    print(f"🌙 Moon Dev Convergence Exit (Long) at {price:.2f} ✨")
                else:
                    print(f"🌙 Moon Dev Convergence Exit (Short) at {price:.2f} ✨")
                self.position.close()
                self._reset_trade()
                return
            
            return
        
        # === ENTRY LOGIC ===
        if not self._is_squeeze(idx):
            return
        
        vol_avg = self.vol_ma[-1]
        if vol_avg is None or np.isnan(vol_avg) or vol_avg <= 0:
            return
        vol_spike = self.data.Volume[-1] > (self.vol_spike_mult * vol_avg)
        
        if not vol_spike:
            return
        
        range_high = self.range_high[-2]  # use prior bar's range to avoid lookahead
        range_low = self.range_low[-2]
        atr_val = self.atr[-1]
        
        if atr_val is None or np.isnan(atr_val):
            return
        
        # Long entry
        if (price > range_high and rsi_val > self.rsi_long_threshold 
                and self._bullish_divergence(idx)):
            stop = price - atr_val  # 1x ATR beyond breakout candle
            risk = price - stop
            if risk <= 0:
                return
            size = int(round(1_000_000 / price))
            if size < 1:
                return
            print(f"🚀 Moon Dev LONG Entry! Price: {price:.2f} | RSI: {rsi_val:.2f} | Vol Spike! 📈")
            self.buy(size=size)
            self.entry_price = price
            self.stop_price = stop
            self.entry_bar = idx
            self.trail_active = False
        
        # Short entry
        elif (price < range_low and rsi_val < self.rsi_short_threshold 
                and self._bearish_divergence(idx)):
            stop = price + atr_val
            risk = stop - price
            if risk <= 0:
                return
            size = int(round(1_000_000 / price))
            if size < 1:
                return
            print(f"🚀 Moon Dev SHORT Entry! Price: {price:.2f} | RSI: {rsi_val:.2f} | Vol Spike! 📉")
            self.sell(size=size)
            self.entry_price = price
            self.stop_price = stop
            self.entry_bar = idx
            self.trail_active = False
    
    def _reset_trade(self):
        self.entry_price = None
        self.stop_price = None
        self.entry_bar = None
        self.trail_active = False


print("🌙✨ Starting Moon Dev Backtest... 🚀")
bt = Backtest(data, ConvergentVolatility, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)