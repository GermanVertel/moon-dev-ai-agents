import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

print("🌙 Moon Dev's VolatilityBracket Backtest Initializing... ✨")

# Load data
data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper column mapping
data = data.rename(columns={
    'datetime': 'Date',
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

data['Date'] = pd.to_datetime(data['Date'])
data = data.set_index('Date')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)

print(f"🚀 Data loaded: {len(data)} bars from {data.index[0]} to {data.index[-1]}")


class VolatilityBracket(Strategy):
    atr_period = 14
    atr_ma_period = 50
    bb_period = 20
    bb_std = 2
    rsi_period = 14
    vol_ma_period = 20
    swing_window = 5
    
    risk_pct = 0.01
    max_atr_mult = 2.0
    circuit_breaker_mult = 2.0
    
    def init(self):
        print("🌙 Initializing indicators... ✨")
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume
        
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period, name='ATR')
        # Wrap ATR in a passthrough so SMA can consume it properly
        self.atr_series = self.I(lambda x: x, self.atr, name='ATR_Series')
        self.atr_ma = self.I(talib.SMA, self.atr_series, timeperiod=self.atr_ma_period, name='ATR_MA')
        self.rsi = self.I(talib.RSI, close, timeperiod=self.rsi_period, name='RSI')
        self.vol_ma = self.I(talib.SMA, volume, timeperiod=self.vol_ma_period, name='Vol_MA')
        
        self.bb_upper = self.I(
            lambda c: talib.BBANDS(c, timeperiod=self.bb_period, nbdevup=self.bb_std, nbdevdn=self.bb_std)[0],
            close, name='BB_Upper'
        )
        self.bb_mid = self.I(
            lambda c: talib.BBANDS(c, timeperiod=self.bb_period, nbdevup=self.bb_std, nbdevdn=self.bb_std)[1],
            close, name='BB_Mid'
        )
        self.bb_lower = self.I(
            lambda c: talib.BBANDS(c, timeperiod=self.bb_period, nbdevup=self.bb_std, nbdevdn=self.bb_std)[2],
            close, name='BB_Lower'
        )
        
        self.swing_high = self.I(talib.MAX, high, timeperiod=self.swing_window * 2 + 1, name='SwingHigh')
        self.swing_low = self.I(talib.MIN, low, timeperiod=self.swing_window * 2 + 1, name='SwingLow')
        
        self.trade_bars = 0
        self.entry_price = 0
        self.stop_price = 0
        self.tp_price = 0
        self.trailing_active = False
        self.trail_stop = 0
        self.mode = None
        print("✅ Indicators ready! 🚀")
    
    def next(self):
        if len(self.data) < max(self.atr_ma_period, self.swing_window * 2 + 1, self.vol_ma_period) + 5:
            return
        
        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        atr = self.atr[-1]
        atr_prev = self.atr[-2]
        atr_ma = self.atr_ma[-1]
        rsi = self.rsi[-1]
        vol = self.data.Volume[-1]
        vol_ma = self.vol_ma[-1]
        swing_high = self.swing_high[-1]
        swing_low = self.swing_low[-1]
        
        if np.isnan(atr) or np.isnan(atr_ma) or atr <= 0:
            return
        
        # Manage open trade
        if self.position:
            self.trade_bars += 1
            
            # Time stop
            if self.trade_bars >= 10:
                unrealized = (price - self.entry_price) if self.position.is_long else (self.entry_price - price)
                if unrealized < 0.5 * atr:
                    print(f"⏰ Time stop hit at {price:.2f} 🌙")
                    self.position.close()
                    return
            
            # Trailing stop activation
            if not self.trailing_active:
                if self.position.is_long and (price - self.entry_price) >= 1.0 * atr:
                    self.trailing_active = True
                    print(f"🎯 Trailing stop activated (LONG) at {price:.2f} ✨")
                elif self.position.is_short and (self.entry_price - price) >= 1.0 * atr:
                    self.trailing_active = True
                    print(f"🎯 Trailing stop activated (SHORT) at {price:.2f} ✨")
            
            if self.trailing_active:
                if self.position.is_long:
                    new_trail = price - 1.5 * atr
                    if new_trail > self.trail_stop:
                        self.trail_stop = new_trail
                    if price <= self.trail_stop:
                        print(f"🛑 Trailing stop exit LONG at {price:.2f} 🌙")
                        self.position.close()
                        return
                else:
                    new_trail = price + 1.5 * atr
                    if self.trail_stop == 0 or new_trail < self.trail_stop:
                        self.trail_stop = new_trail
                    if price >= self.trail_stop:
                        print(f"🛑 Trailing stop exit SHORT at {price:.2f} 🌙")
                        self.position.close()
                        return
            
            # Fixed SL/TP
            if self.position.is_long:
                if low <= self.stop_price:
                    print(f"❌ Stop loss LONG at {self.stop_price:.2f} 🌙")
                    self.position.close()
                    return
                if high >= self.tp_price:
                    print(f"💰 Take profit LONG at {self.tp_price:.2f} ✨")
                    self.position.close()
                    return
            else:
                if high >= self.stop_price:
                    print(f"❌ Stop loss SHORT at {self.stop_price:.2f} 🌙")
                    self.position.close()
                    return
                if low <= self.tp_price:
                    print(f"💰 Take profit SHORT at {self.tp_price:.2f} ✨")
                    self.position.close()
                    return
            return
        
        # Regime detection
        low_vol_regime = atr < atr_ma
        high_vol_regime = atr > atr_ma
        
        # Circuit breaker
        circuit_breaker = atr > self.circuit_breaker_mult * atr_ma
        size_mult = 0.5 if circuit_breaker else 1.0
        stop_mult = 1.5 if circuit_breaker else 1.0
        
        # Position sizing
        risk_amount = self.equity * self.risk_pct * size_mult
        
        # Support/resistance zones
        resistance_zone = swing_high
        support_zone = swing_low
        
        vol_confirm = vol > vol_ma
        
        # BREAKOUT MODE (low volatility)
        if low_vol_regime and not self.position:
            # Breakout above resistance
            if price > resistance_zone + 0.3 * atr and vol_confirm and atr > atr_prev:
                stop_dist = 0.5 * atr * stop_mult
                stop_price = price - stop_dist
                tp_price = price + 2.0 * atr
                size = int(round(risk_amount / stop_dist))
                if size > 0:
                    print(f"🚀 BREAKOUT LONG at {price:.2f} | ATR={atr:.2f} | Size={size} 🌙")
                    self.buy(size=size)
                    self.entry_price = price
                    self.stop_price = stop_price
                    self.tp_price = tp_price
                    self.trade_bars = 0
                    self.trailing_active = False
                    self.trail_stop = 0
                    self.mode = 'breakout'
                    return
            
            # Breakdown below support
            if price < support_zone - 0.3 * atr and vol_confirm and atr > atr_prev:
                stop_dist = 0.5 * atr * stop_mult
                stop_price = price + stop_dist
                tp_price = price - 2.0 * atr
                size = int(round(risk_amount / stop_dist))
                if size > 0:
                    print(f"🔻 BREAKOUT SHORT at {price:.2f} | ATR={atr:.2f} | Size={size} 🌙")
                    self.sell(size=size)
                    self.entry_price = price
                    self.stop_price = stop_price
                    self.tp_price = tp_price
                    self.trade_bars = 0
                    self.trailing_active = False
                    self.trail_stop = 0
                    self.mode = 'breakout'
                    return
        
        # REVERSAL MODE (high volatility)
        if high_vol_regime and not self.position:
            # Support rejection (bullish)
            touched_support = low <= support_zone + 0.25 * atr and low >= support_zone - 0.5 * atr
            closed_inside = price > support_zone
            bullish_candle = price > self.data.Open[-1]
            
            if touched_support and closed_inside and bullish_candle and rsi < 40:
                stop_dist = 1.0 * atr * stop_mult
                stop_price = support_zone - stop_dist
                tp_price = price + 1.5 * atr
                size = int(round(risk_amount / (price - stop_price)))
                if size > 0:
                    print(f"💚 REVERSAL LONG (support rejection) at {price:.2f} | ATR={atr:.2f} | Size={size} 🌙")
                    self.buy(size=size)
                    self.entry_price = price
                    self.stop_price = stop_price
                    self.tp_price = tp_price
                    self.trade_bars = 0
                    self.trailing_active = False
                    self.trail_stop = 0
                    self.mode = 'reversal'
                    return
            
            # Resistance rejection (bearish)
            touched_resistance = high >= resistance_zone - 0.25 * atr and high <= resistance_zone + 0.5 * atr
            closed_inside_r = price < resistance_zone
            bearish_candle = price < self.data.Open[-1]
            
            if touched_resistance and closed_inside_r and bearish_candle and rsi > 60:
                stop_dist = 1.0 * atr * stop_mult
                stop_price = resistance_zone + stop_dist
                tp_price = price - 1.5 * atr
                size = int(round(risk_amount / (stop_price - price)))
                if size > 0:
                    print(f"❤️ REVERSAL SHORT (resistance rejection) at {price:.2f} | ATR={atr:.2f} | Size={size} 🌙")
                    self.sell(size=size)
                    self.entry_price = price
                    self.stop_price = stop_price
                    self.tp_price = tp_price
                    self.trade_bars = 0
                    self.trailing_active = False
                    self.trail_stop = 0
                    self.mode = 'reversal'
                    return


print("🌙 Setting up backtest... ✨")
bt = Backtest(data, VolatilityBracket, cash=1000000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)