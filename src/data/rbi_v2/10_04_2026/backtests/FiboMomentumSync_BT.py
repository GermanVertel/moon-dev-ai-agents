import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# 🌙 Moon Dev's FiboMomentumSync Backtest 🌙
print("🚀 Initializing FiboMomentumSync Strategy...")

# Load data
data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper column mapping
data.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')
data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

print(f"✨ Data loaded: {len(data)} bars from {data.index[0]} to {data.index[-1]}")


class FiboMomentumSync(Strategy):
    # Strategy parameters
    ema_fast = 50
    ema_slow = 200
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    atr_period = 14
    swing_lookback = 50
    risk_pct = 0.02
    time_stop_bars = 20

    def init(self):
        print("🌙 Initializing indicators...")
        
        # Trend EMAs
        self.ema50 = self.I(talib.EMA, self.data.Close, timeperiod=self.ema_fast)
        self.ema200 = self.I(talib.EMA, self.data.Close, timeperiod=self.ema_slow)
        
        # MACD
        self.macd, self.macd_signal_line, self.macd_hist = self.I(
            talib.MACD, self.data.Close,
            fastperiod=self.macd_fast,
            slowperiod=self.macd_slow,
            signalperiod=self.macd_signal
        )
        
        # ATR for stops
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close,
                          timeperiod=self.atr_period)
        
        # Swing high/low
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=self.swing_lookback)
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=self.swing_lookback)
        
        # Track entry info
        self.entry_bar = 0
        self.entry_price = 0
        self.stop_price = 0
        self.tp1_price = 0
        self.tp2_price = 0
        self.tp1_hit = False
        
        print("✨ All indicators initialized!")

    def next(self):
        # Skip if not enough data
        if len(self.data) < self.ema_slow + 10:
            return
        
        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        
        ema50 = self.ema50[-1]
        ema200 = self.ema200[-1]
        macd = self.macd[-1]
        macd_sig = self.macd_signal_line[-1]
        macd_hist = self.macd_hist[-1]
        macd_hist_prev = self.macd_hist[-2]
        atr = self.atr[-1]
        
        swing_high = self.swing_high[-1]
        swing_low = self.swing_low[-1]
        
        # ============ MANAGE OPEN POSITION ============
        if self.position:
            bars_in_trade = len(self.data) - self.entry_bar
            
            # Time stop
            if bars_in_trade >= self.time_stop_bars and not self.tp1_hit:
                print(f"⏰ Time stop hit after {bars_in_trade} bars - closing at {price:.2f}")
                self.position.close()
                return
            
            if self.position.is_long:
                # Check TP1
                if not self.tp1_hit and high >= self.tp1_price:
                    self.tp1_hit = True
                    print(f"🎯 TP1 hit (long) at {self.tp1_price:.2f} - moving SL to breakeven")
                    self.stop_price = self.entry_price
                
                # Check TP2
                if self.tp1_hit and high >= self.tp2_price:
                    print(f"🚀 TP2 hit (long) at {self.tp2_price:.2f} - closing position!")
                    self.position.close()
                    return
                
                # Trailing stop with ATR after TP1
                if self.tp1_hit:
                    trail_stop = price - 2 * atr
                    if trail_stop > self.stop_price:
                        self.stop_price = trail_stop
                
                # Check stop loss
                if low <= self.stop_price:
                    print(f"🛑 Stop loss hit (long) at {self.stop_price:.2f}")
                    self.position.close()
                    return
                
                # Momentum exit: MACD crosses back against trade
                if not self.tp1_hit and macd < macd_sig and self.macd[-2] >= self.macd_signal_line[-2]:
                    print(f"📉 MACD bearish cross - momentum exit (long)")
                    self.position.close()
                    return
            
            elif self.position.is_short:
                # Check TP1
                if not self.tp1_hit and low <= self.tp1_price:
                    self.tp1_hit = True
                    print(f"🎯 TP1 hit (short) at {self.tp1_price:.2f} - moving SL to breakeven")
                    self.stop_price = self.entry_price
                
                # Check TP2
                if self.tp1_hit and low <= self.tp2_price:
                    print(f"🚀 TP2 hit (short) at {self.tp2_price:.2f} - closing position!")
                    self.position.close()
                    return
                
                # Trailing stop with ATR after TP1
                if self.tp1_hit:
                    trail_stop = price + 2 * atr
                    if trail_stop < self.stop_price:
                        self.stop_price = trail_stop
                
                # Check stop loss
                if high >= self.stop_price:
                    print(f"🛑 Stop loss hit (short) at {self.stop_price:.2f}")
                    self.position.close()
                    return
                
                # Momentum exit
                if not self.tp1_hit and macd > macd_sig and self.macd[-2] <= self.macd_signal_line[-2]:
                    print(f"📈 MACD bullish cross - momentum exit (short)")
                    self.position.close()
                    return
            return
        
        # ============ ENTRY LOGIC ============
        
        # Trend identification
        uptrend = price > ema200 and ema50 > ema200
        downtrend = price < ema200 and ema50 < ema200
        
        if not uptrend and not downtrend:
            return
        
        # Calculate Fibonacci levels
        swing_range = swing_high - swing_low
        if swing_range <= 0:
            return
        
        if uptrend:
            # Fib retracement levels for long (from swing low to swing high)
            fib_382 = swing_high - 0.382 * swing_range
            fib_500 = swing_high - 0.500 * swing_range
            fib_618 = swing_high - 0.618 * swing_range
            fib_786 = swing_high - 0.786 * swing_range
            
            # Price in golden zone (38.2% - 61.8%)
            in_fib_zone = fib_618 <= price <= fib_382
            
            if not in_fib_zone:
                return
            
            # MACD confirmation
            macd_bull_cross = macd > macd_sig and self.macd[-2] <= self.macd_signal_line[-2]
            macd_hist_flip = macd_hist > 0 and macd_hist_prev <= 0
            macd_bullish_div = (low < self.data.Low[-2] and macd > self.macd[-2])
            
            macd_confirm = macd_bull_cross or macd_hist_flip or macd_bullish_div
            
            if not macd_confirm:
                return
            
            # Candlestick confirmation: close above prior candle's high
            candle_confirm = price > self.data.High[-2]
            
            if not candle_confirm:
                return
            
            # Calculate stops and targets
            stop_price = fib_786 - 0.5 * atr
            risk = price - stop_price
            
            if risk <= 0:
                return
            
            tp1_price = swing_high
            tp2_price = swing_high + 0.272 * swing_range  # 127.2% extension
            
            # Risk-reward check (min 1:2 for TP1)
            reward1 = tp1_price - price
            if reward1 / risk < 2.0:
                print(f"⚠️ R:R too low ({reward1/risk:.2f}) - skipping long")
                return
            
            # Position sizing: risk 2% of equity
            equity = self.equity
            risk_amount = equity * self.risk_pct
            position_size = risk_amount / risk
            position_size = int(round(position_size))
            
            if position_size <= 0:
                return
            
            print(f"🌙 LONG ENTRY | Price: {price:.2f} | Fib Zone: {fib_618:.2f}-{fib_382:.2f}")
            print(f"   SL: {stop_price:.2f} | TP1: {tp1_price:.2f} | TP2: {tp2_price:.2f}")
            print(f"   Size: {position_size} | R:R = {reward1/risk:.2f}")
            
            self.buy(size=position_size)
            self.entry_bar = len(self.data)
            self.entry_price = price
            self.stop_price = stop_price
            self.tp1_price = tp1_price
            self.tp2_price = tp2_price
            self.tp1_hit = False
        
        elif downtrend:
            # Fib retracement levels for short (from swing high to swing low)
            fib_382 = swing_low + 0.382 * swing_range
            fib_500 = swing_low + 0.500 * swing_range
            fib_618 = swing_low + 0.618 * swing_range
            fib_786 = swing_low + 0.786 * swing_range
            
            # Price in golden zone
            in_fib_zone = fib_382 <= price <= fib_618
            
            if not in_fib_zone:
                return
            
            # MACD confirmation
            macd_bear_cross = macd < macd_sig and self.macd[-2] >= self.macd_signal_line[-2]
            macd_hist_flip = macd_hist < 0 and macd_hist_prev >= 0
            macd_bearish_div = (high > self.data.High[-2] and macd < self.macd[-2])
            
            macd_confirm = macd_bear_cross or macd_hist_flip or macd_bearish_div
            
            if not macd_confirm:
                return
            
            # Candlestick confirmation: close below prior candle's low
            candle_confirm = price < self.data.Low[-2]
            
            if not candle_confirm:
                return
            
            # Calculate stops and targets
            stop_price = fib_786 + 0.5 * atr
            risk = stop_price - price
            
            if risk <= 0:
                return
            
            tp1_price = swing_low
            tp2_price = swing_low - 0.272 * swing_range  # 127.2% extension
            
            # Risk-reward check
            reward1 = price - tp1_price
            if reward1 / risk < 2.0:
                print(f"⚠️ R:R too low ({reward1/risk:.2f}) - skipping short")
                return
            
            # Position sizing
            equity = self.equity
            risk_amount = equity * self.risk_pct
            position_size = risk_amount / risk
            position_size = int(round(position_size))
            
            if position_size <= 0:
                return
            
            print(f"🌙 SHORT ENTRY | Price: {price:.2f} | Fib Zone: {fib_382:.2f}-{fib_618:.2f}")
            print(f"   SL: {stop_price:.2f} | TP1: {tp1_price:.2f} | TP2: {tp2_price:.2f}")
            print(f"   Size: {position_size} | R:R = {reward1/risk:.2f}")
            
            self.sell(size=position_size)
            self.entry_bar = len(self.data)
            self.entry_price = price
            self.stop_price = stop_price
            self.tp1_price = tp1_price
            self.tp2_price = tp2_price
            self.tp1_hit = False


print("🚀 Running backtest...")
bt = Backtest(data, FiboMomentumSync, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)