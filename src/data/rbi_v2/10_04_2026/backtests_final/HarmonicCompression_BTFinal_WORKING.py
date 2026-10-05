import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# 🌙 Moon Dev's HarmonicCompression Backtest 🚀

class HarmonicCompression(Strategy):
    # Strategy parameters
    ema_fast_period = 50
    ema_slow_period = 200
    atr_period = 14
    atr_lookback = 15
    atr_compression_ratio = 0.85
    pivot_lookback = 20
    fib_levels = [0.382, 0.5, 0.618]
    fib_tolerance = 0.005  # 0.5% tolerance around fib levels
    risk_per_trade = 0.01
    stop_atr_mult = 1.5
    expansion_mult = 1.3
    max_compression_bars = 30

    def init(self):
        print("🌙 Moon Dev: Initializing HarmonicCompression strategy...")
        
        self.ema_fast = self.I(talib.EMA, self.data.Close, timeperiod=self.ema_fast_period)
        self.ema_slow = self.I(talib.EMA, self.data.Close, timeperiod=self.ema_slow_period)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period)
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=14)
        
        # Pivot high/low using rolling max/min
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=self.pivot_lookback)
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=self.pivot_lookback)
        
        self.entry_atr = None
        self.compression_start = None
        self.entry_price = None
        self.stop_price = None
        self.tp_price = None
        print("✨ Moon Dev: Indicators loaded successfully!")

    def is_bullish_reversal(self, idx):
        """Check for bullish reversal candle patterns"""
        o = self.data.Open[idx]
        h = self.data.High[idx]
        l = self.data.Low[idx]
        c = self.data.Close[idx]
        
        body = abs(c - o)
        rng = h - l
        if rng == 0:
            return False
        
        lower_wick = min(o, c) - l
        upper_wick = h - max(o, c)
        
        # Hammer: small body, long lower wick, small upper wick
        is_hammer = (body <= rng * 0.35) and (lower_wick >= rng * 0.5) and (upper_wick <= rng * 0.2)
        
        # Bullish engulfing
        is_engulfing = False
        if idx > 0:
            prev_o = self.data.Open[idx-1]
            prev_c = self.data.Close[idx-1]
            is_engulfing = (prev_c < prev_o) and (c > o) and (c > prev_o) and (o < prev_c)
        
        # Pin bar
        is_pin = (lower_wick >= rng * 0.6) and (body <= rng * 0.3)
        
        return is_hammer or is_engulfing or is_pin

    def next(self):
        # Need enough history
        if len(self.data) < self.ema_slow_period + 5:
            return
        
        idx = len(self.data) - 1
        price = self.data.Close[-1]
        
        # Handle open position
        if self.position:
            # Primary exit: ATR expansion
            if self.entry_atr is not None:
                if self.atr[-1] > self.entry_atr * self.expansion_mult:
                    print(f"🚀 Moon Dev: ATR expansion exit! ATR={self.atr[-1]:.2f} > Entry ATR*{self.expansion_mult}={self.entry_atr*self.expansion_mult:.2f}")
                    self.position.close()
                    self.entry_atr = None
                    return
            
            # Stop loss / take profit handled by bracket orders
            return
        
        # Check compression duration
        atr_prev = self.atr[-self.atr_lookback]
        compression = self.atr[-1] < atr_prev * self.atr_compression_ratio
        
        if compression:
            if self.compression_start is None:
                self.compression_start = idx
        else:
            self.compression_start = None
        
        # Skip if compression lasted too long
        if self.compression_start is not None and (idx - self.compression_start) > self.max_compression_bars:
            return
        
        # Trend filter
        uptrend = (price > self.ema_slow[-1]) and (self.ema_fast[-1] > self.ema_slow[-1])
        if not uptrend:
            return
        
        # Volatility compression
        if not compression:
            return
        
        # Fibonacci levels from recent swing
        sh = self.swing_high[-1]
        sl = self.swing_low[-1]
        if sh <= sl:
            return
        
        swing_range = sh - sl
        in_fib_zone = False
        touched_level = None
        
        for fib in self.fib_levels:
            fib_price = sh - swing_range * fib
            if abs(price - fib_price) / fib_price <= self.fib_tolerance:
                in_fib_zone = True
                touched_level = fib
                break
        
        if not in_fib_zone:
            return
        
        # Candlestick confirmation
        if not self.is_bullish_reversal(idx):
            return
        
        # RSI filter (avoid overbought)
        if self.rsi[-1] > 75:
            return
        
        # Calculate stop loss
        stop_price = price - self.stop_atr_mult * self.atr[-1]
        stop_distance = price - stop_price
        if stop_distance <= 0:
            return
        
        # Position sizing: risk 1% of equity -> use fraction of equity for backtesting compat
        risk_amount = self.equity * self.risk_per_trade
        position_size_units = risk_amount / stop_distance
        position_size_units = int(round(position_size_units))
        
        if position_size_units <= 0:
            return
        
        # Convert to fraction of equity for percentage-based sizing
        position_size_frac = (position_size_units * price) / self.equity
        if position_size_frac <= 0 or position_size_frac >= 1:
            # Cap at 0.95 to avoid over-leverage issues
            position_size_frac = min(max(position_size_frac, 0.001), 0.95)
        
        # Take profit at swing high
        tp_price = sh
        
        print(f"🌙 Moon Dev: ENTRY SIGNAL! Fib {touched_level*100:.1f}% touch at {price:.2f}")
        print(f"   📈 Trend: EMA50={self.ema_fast[-1]:.2f} > EMA200={self.ema_slow[-1]:.2f}")
        print(f"   📉 ATR Compression: {self.atr[-1]:.2f} < {atr_prev*self.atr_compression_ratio:.2f}")
        print(f"   🎯 Entry={price:.2f} | SL={stop_price:.2f} | TP={tp_price:.2f} | SizeFrac={position_size_frac:.4f}")
        
        self.entry_atr = self.atr[-1]
        self.entry_price = price
        self.stop_price = stop_price
        
        self.buy(size=position_size_frac, sl=stop_price, tp=tp_price)


# Load and prepare data
print("🌙 Moon Dev: Loading BTC-USD 15m data...")
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Rename to proper case
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

print(f"✨ Moon Dev: Data loaded! Shape={data.shape}")
print(f"🚀 Moon Dev: Running backtest with 1,000,000 initial equity...")

bt = Backtest(
    data,
    HarmonicCompression,
    cash=1_000_000,
    commission=0.001
)

stats = bt.run()
print(stats)
print(stats._strategy)