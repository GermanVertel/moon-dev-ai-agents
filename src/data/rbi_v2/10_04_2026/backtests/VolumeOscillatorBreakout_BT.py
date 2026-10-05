import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# 🌙 Moon Dev Backtest AI - Volume Oscillator Breakout Strategy
print("🌙✨ Moon Dev Backtest AI initializing... 🚀")
print("🌙 Strategy: Volume Oscillator Breakout")
print("🌙 Data path: /Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv")

# Load data
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

# Set datetime index
if 'datetime' in data.columns:
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')

print(f"🌙 Data loaded: {len(data)} bars ✨")
print(f"🌙 Date range: {data.index[0]} to {data.index[-1]}")


class VolumeOscillatorBreakout(Strategy):
    """
    🌙 Volume Oscillator Breakout Strategy
    Combines volume spikes + RSI oversold + breakout confirmation
    """
    # Strategy parameters
    vol_ma_period = 20
    rsi_period = 14
    atr_period = 14
    ema_period = 200
    breakout_period = 5
    vol_multiplier = 2.0
    rsi_oversold = 30
    rsi_overbought = 70
    risk_pct = 0.01  # 1% risk per trade
    rr_ratio = 2.0   # 2:1 risk-reward
    time_exit_bars = 10
    signal_window = 3  # breakout within 3 bars of signal

    def init(self):
        print("🌙✨ Initializing indicators... 🚀")
        
        # Volume moving average
        self.vol_ma = self.I(talib.SMA, self.data.Volume, timeperiod=self.vol_ma_period)
        
        # RSI
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=self.rsi_period)
        
        # ATR
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period)
        
        # EMA trend filter
        self.ema = self.I(talib.EMA, self.data.Close, timeperiod=self.ema_period)
        
        # 5-period high/low for breakout
        self.high_5 = self.I(talib.MAX, self.data.High, timeperiod=self.breakout_period)
        self.low_5 = self.I(talib.MIN, self.data.Low, timeperiod=self.breakout_period)
        
        # Signal tracking state
        self.long_signal_bar = None
        self.short_signal_bar = None
        self.long_signal_high = None
        self.short_signal_low = None
        self.entry_bar = None
        self.stop_price = None
        self.target_price = None
        
        print("🌙✨ Indicators initialized! ✨")

    def next(self):
        # Skip if not enough data
        if len(self.data) < self.ema_period + 5:
            return

        current_price = self.data.Close[-1]
        current_high = self.data.High[-1]
        current_low = self.data.Low[-1]
        current_vol = self.data.Volume[-1]
        current_rsi = self.rsi[-1]
        current_atr = self.atr[-1]
        current_vol_ma = self.vol_ma[-1]
        current_ema = self.ema[-1]
        
        # Handle open position - check exits
        if self.position:
            # Time exit
            if self.entry_bar is not None:
                bars_held = len(self.data) - self.entry_bar
                if bars_held >= self.time_exit_bars:
                    print(f"⏰ Moon Dev Time Exit after {bars_held} bars 🌙")
                    self.position.close()
                    self._reset_position_state()
                    return
            
            # Stop loss / Target exit (backtesting.py handles via SL/TP orders)
            # Check for trailing stop using 10-period EMA
            if self.position.is_long:
                # Trailing stop: exit if close below EMA10
                ema10 = talib.EMA(self.data.Close, timeperiod=10)[-1]
                if not np.isnan(ema10) and current_price < ema10:
                    print(f"🌙 Trailing stop hit (below EMA10) - Exit long ✨")
                    self.position.close()
                    self._reset_position_state()
                    return
            elif self.position.is_short:
                ema10 = talib.EMA(self.data.Close, timeperiod=10)[-1]
                if not np.isnan(ema10) and current_price > ema10:
                    print(f"🌙 Trailing stop hit (above EMA10) - Exit short ✨")
                    self.position.close()
                    self._reset_position_state()
                    return
        
        # Detect volume oversold signal
        vol_spike = current_vol > (self.vol_multiplier * current_vol_ma) if not np.isnan(current_vol_ma) else False
        
        # Long signal: volume spike + RSI oversold
        if vol_spike and current_rsi < self.rsi_oversold and not self.position:
            self.long_signal_bar = len(self.data)
            self.long_signal_high = current_high
            print(f"🌙🔔 LONG SIGNAL detected! Vol spike: {current_vol:.2f} > {self.vol_multiplier}x MA({current_vol_ma:.2f}) | RSI: {current_rsi:.2f} 📉")
        
        # Short signal: volume spike + RSI overbought
        if vol_spike and current_rsi > self.rsi_overbought and not self.position:
            self.short_signal_bar = len(self.data)
            self.short_signal_low = current_low
            print(f"🌙🔔 SHORT SIGNAL detected! Vol spike: {current_vol:.2f} > {self.vol_multiplier}x MA({current_vol_ma:.2f}) | RSI: {current_rsi:.2f} 📈")
        
        # Check for long breakout confirmation
        if self.long_signal_bar is not None and not self.position:
            bars_since_signal = len(self.data) - self.long_signal_bar
            if bars_since_signal <= self.signal_window and bars_since_signal > 0:
                # Breakout above signal bar high
                if current_price > self.long_signal_high:
                    # Trend filter: price above 200 EMA
                    if not np.isnan(current_ema) and current_price > current_ema:
                        print(f"🚀🌙 LONG BREAKOUT confirmed! Price {current_price:.2f} > Signal High {self.long_signal_high:.2f} | EMA200: {current_ema:.2f}")
                        
                        # Calculate position size based on risk
                        if not np.isnan(current_atr) and current_atr > 0:
                            stop_price = min(self.long_signal_high - current_atr, current_low - current_atr * 0.5)
                            risk_per_unit = current_price - stop_price
                            if risk_per_unit > 0:
                                risk_amount = self.equity * self.risk_pct
                                position_size = int(round(risk_amount / risk_per_unit))
                                position_size = max(1, min(position_size, 1000000))
                                
                                target_price = current_price + (risk_per_unit * self.rr_ratio)
                                
                                self.buy(size=position_size, sl=stop_price, tp=target_price)
                                self.entry_bar = len(self.data)
                                self.stop_price = stop_price
                                self.target_price = target_price
                                print(f"🌙✨ ENTRY LONG: size={position_size} | Entry: {current_price:.2f} | SL: {stop_price:.2f} | TP: {target_price:.2f} 🚀")
                    else:
                        print(f"🌙 ⚠️ Long breakout but price below EMA200 - skipping (downtrend filter)")
                    self.long_signal_bar = None
                    self.long_signal_high = None
            elif bars_since_signal > self.signal_window:
                self.long_signal_bar = None
                self.long_signal_high = None
        
        # Check for short breakout confirmation
        if self.short_signal_bar is not None and not self.position:
            bars_since_signal = len(self.data) - self.short_signal_bar
            if bars_since_signal <= self.signal_window and bars_since_signal > 0:
                # Breakdown below signal bar low
                if current_price < self.short_signal_low:
                    print(f"🚀🌙 SHORT BREAKDOWN confirmed! Price {current_price:.2f} < Signal Low {self.short_signal_low:.2f}")
                    
                    if not np.isnan(current_atr) and current_atr > 0:
                        stop_price = max(self.short_signal_low + current_atr, current_high + current_atr * 0.5)
                        risk_per_unit = stop_price - current_price
                        if risk_per_unit > 0:
                            risk_amount = self.equity * self.risk_pct
                            position_size = int(round(risk_amount / risk_per_unit))
                            position_size = max(1, min(position_size, 1000000))
                            
                            target_price = current_price - (risk_per_unit * self.rr_ratio)
                            
                            self.sell(size=position_size, sl=stop_price, tp=target_price)
                            self.entry_bar = len(self.data)
                            self.stop_price = stop_price
                            self.target_price = target_price
                            print(f"🌙✨ ENTRY SHORT: size={position_size} | Entry: {current_price:.2f} | SL: {stop_price:.2f} | TP: {target_price:.2f} 🚀")
                    self.short_signal_bar = None
                    self.short_signal_low = None
            elif bars_since_signal > self.signal_window:
                self.short_signal_bar = None
                self.short_signal_low = None

    def _reset_position_state(self):
        self.entry_bar = None
        self.stop_price = None
        self.target_price = None


print("🌙✨ Setting up backtest... 🚀")
bt = Backtest(
    data,
    VolumeOscillatorBreakout,
    cash=1_000_000,
    commission=0.002,
    exclusive_orders=True
)

print("🌙✨ Running backtest... 🚀")
stats = bt.run()
print("🌙✨ Backtest complete! ✨")
print(stats)
print(stats._strategy)