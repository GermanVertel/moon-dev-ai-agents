from backtesting import Backtest, Strategy
import talib
import pandas as pd
import numpy as np

# Load and clean data
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

print("🌙 Moon Dev DevianceSqueeze Backtest Loading... ✨")
print(f"📊 Data shape: {data.shape}")
print(f"📈 Columns: {list(data.columns)}")


def detect_bearish_divergence(price_high, rsi_high, lookback=10):
    """Detect bearish divergence: price higher high, RSI lower high"""
    if len(price_high) < lookback or len(rsi_high) < lookback:
        return False
    
    # Recent half vs older half
    mid = lookback // 2
    recent_price_max = np.max(price_high[-mid:])
    older_price_max = np.max(price_high[:mid])
    recent_rsi_max = np.max(rsi_high[-mid:])
    older_rsi_max = np.max(rsi_high[:mid])
    
    return (recent_price_max > older_price_max) and (recent_rsi_max < older_rsi_max)


def detect_bullish_divergence(price_low, rsi_low, lookback=10):
    """Detect bullish divergence: price lower low, RSI higher low"""
    if len(price_low) < lookback or len(rsi_low) < lookback:
        return False
    
    mid = lookback // 2
    recent_price_min = np.min(price_low[-mid:])
    older_price_min = np.min(price_low[:mid])
    recent_rsi_min = np.min(rsi_low[-mid:])
    older_rsi_min = np.min(rsi_low[:mid])
    
    return (recent_price_min < older_price_min) and (recent_rsi_min > older_rsi_min)


class DevianceSqueeze(Strategy):
    bb_period = 20
    bb_std = 2.0
    rsi_period = 14
    atr_period = 14
    lookback = 10
    risk_pct = 0.02
    atr_mult = 1.5
    time_stop_bars = 12
    
    def init(self):
        print("🌙 Initializing DevianceSqueeze indicators... ✨")
        
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        
        # Bollinger Bands
        self.bb_mid = self.I(talib.SMA, close, timeperiod=self.bb_period, name="BB_Mid")
        self.bb_std_dev = self.I(talib.STDDEV, close, timeperiod=self.bb_period, nbdev=1, name="BB_StdDev")
        
        # RSI
        self.rsi = self.I(talib.RSI, close, timeperiod=self.rsi_period, name="RSI")
        
        # ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period, name="ATR")
        
        # Swing high/low
        self.swing_high = self.I(talib.MAX, high, timeperiod=10, name="SwingHigh")
        self.swing_low = self.I(talib.MIN, low, timeperiod=10, name="SwingLow")
        
        # Track entry bar for time stop
        self.entry_bar = None
        self.entry_price = None
        self.stop_price = None
        self.tp_price = None
        self.trade_direction = None
        
        print("🚀 Indicators ready! Let's squeeze some profits! 💰")
    
    def next(self):
        if len(self.data) < self.bb_period + self.lookback + 5:
            return
        
        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        open_price = self.data.Open[-1]
        
        # Compute bands manually
        mid = self.bb_mid[-1]
        std = self.bb_std_dev[-1]
        if np.isnan(mid) or np.isnan(std):
            return
        
        upper_band = mid + self.bb_std * std
        lower_band = mid - self.bb_std * std
        
        rsi_val = self.rsi[-1]
        atr_val = self.atr[-1]
        
        if np.isnan(rsi_val) or np.isnan(atr_val):
            return
        
        # Time stop check
        if self.position and self.entry_bar is not None:
            bars_held = len(self.data) - self.entry_bar
            if bars_held >= self.time_stop_bars:
                print(f"⏰ Moon Dev Time Stop hit ({bars_held} bars) — exiting! 🌙")
                self.position.close()
                self.entry_bar = None
                return
        
        # Exit logic — profit target at middle band
        if self.position:
            if self.trade_direction == 'short':
                if price <= mid:
                    print(f"🎯 Short TP hit at mid band {mid:.2f} — Moon Dev profit! 💰")
                    self.position.close()
                    self.entry_bar = None
                    return
                if self.stop_price and high >= self.stop_price:
                    print(f"🛑 Short SL hit at {self.stop_price:.2f} — Moon Dev cuts losses! 🌙")
                    self.position.close()
                    self.entry_bar = None
                    return
            elif self.trade_direction == 'long':
                if price >= mid:
                    print(f"🎯 Long TP hit at mid band {mid:.2f} — Moon Dev profit! 💰")
                    self.position.close()
                    self.entry_bar = None
                    return
                if self.stop_price and low <= self.stop_price:
                    print(f"🛑 Long SL hit at {self.stop_price:.2f} — Moon Dev cuts losses! 🌙")
                    self.position.close()
                    self.entry_bar = None
                    return
            return
        
        # Entry: need price beyond band
        broke_upper = price > upper_band
        broke_lower = price < lower_band
        
        if not (broke_upper or broke_lower):
            return
        
        # Divergence detection using recent windows
        price_highs = self.data.High[-self.lookback:]
        price_lows = self.data.Low[-self.lookback:]
        rsi_window = self.rsi[-self.lookback:]
        
        if np.any(np.isnan(rsi_window)):
            return
        
        bearish_div = detect_bearish_divergence(price_highs, rsi_window, self.lookback)
        bullish_div = detect_bullish_divergence(price_lows, rsi_window, self.lookback)
        
        # Confirmation candle
        prev_low = self.data.Low[-2]
        prev_high = self.data.High[-2]
        bearish_confirm = (price < open_price) or (price < prev_low) or (rsi_val < 70 and self.rsi[-2] >= 70)
        bullish_confirm = (price > open_price) or (price > prev_high) or (rsi_val > 30 and self.rsi[-2] <= 30)
        
        # SHORT setup: upper band breakout + bearish divergence + confirmation
        if broke_upper and bearish_div and bearish_confirm and rsi_val > 60:
            stop = max(self.swing_high[-1], price + self.atr_mult * atr_val)
            risk = stop - price
            if risk <= 0:
                return
            size = int(round(1000000 / price))
            if size < 1:
                size = 1
            print(f"🌙 SHORT SIGNAL! Price {price:.2f} > Upper {upper_band:.2f} | RSI {rsi_val:.1f} | Bearish Divergence! 🐻")
            print(f"🚀 Entry: {price:.2f} | Stop: {stop:.2f} | Size: {size}")
            self.sell(size=size)
            self.entry_bar = len(self.data)
            self.entry_price = price
            self.stop_price = stop
            self.trade_direction = 'short'
            return
        
        # LONG setup: lower band breakout + bullish divergence + confirmation
        if broke_lower and bullish_div and bullish_confirm and rsi_val < 40:
            stop = min(self.swing_low[-1], price - self.atr_mult * atr_val)
            risk = price - stop
            if risk <= 0:
                return
            size = int(round(1000000 / price))
            if size < 1:
                size = 1
            print(f"🌙 LONG SIGNAL! Price {price:.2f} < Lower {lower_band:.2f} | RSI {rsi_val:.1f} | Bullish Divergence! 🐂")
            print(f"🚀 Entry: {price:.2f} | Stop: {stop:.2f} | Size: {size}")
            self.buy(size=size)
            self.entry_bar = len(self.data)
            self.entry_price = price
            self.stop_price = stop
            self.trade_direction = 'long'
            return


print("🌙✨ Starting Moon Dev DevianceSqueeze Backtest! 🚀")
bt = Backtest(data, DevianceSqueeze, cash=1000000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)