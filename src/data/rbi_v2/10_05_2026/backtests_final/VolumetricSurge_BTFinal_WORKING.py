import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# 🌙 Moon Dev VolumetricSurge Strategy ✨
print("🌙 Moon Dev is loading the VolumetricSurge strategy... 🚀")

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

data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')

print(f"🌙 Moon Dev loaded {len(data)} bars of data ✨")
print(f"🚀 Data columns: {list(data.columns)}")


class VolumetricSurge(Strategy):
    # Strategy parameters
    volume_sma_period = 20
    volume_multiplier = 1.5
    donchian_period = 20
    atr_period = 14
    atr_tp_mult = 2.0
    atr_sl_mult = 1.0
    rsi_period = 14
    adx_period = 14
    adx_threshold = 20
    max_bars_in_trade = 20
    risk_per_trade = 0.02

    def init(self):
        print("🌙 Moon Dev initializing indicators... ✨")
        
        # 🌙 Fix: convert Volume to float64 for talib compatibility
        volume_arr = np.asarray(self.data.Volume, dtype=np.float64)
        high_arr = np.asarray(self.data.High, dtype=np.float64)
        low_arr = np.asarray(self.data.Low, dtype=np.float64)
        close_arr = np.asarray(self.data.Close, dtype=np.float64)
        
        # Volume baseline
        self.volume_sma = self.I(talib.SMA, volume_arr, timeperiod=self.volume_sma_period)
        
        # Donchian channel (highest high / lowest low)
        self.donchian_high = self.I(talib.MAX, high_arr, timeperiod=self.donchian_period)
        self.donchian_low = self.I(talib.MIN, low_arr, timeperiod=self.donchian_period)
        
        # ATR for stops/targets
        self.atr = self.I(talib.ATR, high_arr, low_arr, close_arr, timeperiod=self.atr_period)
        
        # RSI momentum
        self.rsi = self.I(talib.RSI, close_arr, timeperiod=self.rsi_period)
        
        # ADX trend strength
        self.adx = self.I(talib.ADX, high_arr, low_arr, close_arr, timeperiod=self.adx_period)
        
        # Track entry info
        self.entry_bar = None
        self.entry_price = None
        self.tp_price = None
        self.sl_price = None
        
        print("🚀 Moon Dev indicators ready! 🌙")

    def next(self):
        # Skip if not enough data
        if len(self.data) < max(self.volume_sma_period, self.donchian_period, self.atr_period, self.adx_period) + 2:
            return
        
        current_price = self.data.Close[-1]
        current_volume = self.data.Volume[-1]
        vol_baseline = self.volume_sma[-1]
        atr_val = self.atr[-1]
        adx_val = self.adx[-1]
        rsi_val = self.rsi[-1]
        
        # Volume surge check
        volume_surge = current_volume > (vol_baseline * self.volume_multiplier)
        
        # Trend strength filter
        trending = adx_val > self.adx_threshold
        
        # Manage open position
        if self.position:
            bars_held = len(self.data) - self.entry_bar
            
            if self.position.is_long:
                # Trailing stop / TP / SL / time exit
                if self.data.High[-1] >= self.tp_price:
                    print(f"🌙✨ TP HIT LONG @ {self.tp_price:.2f} | Moon Dev profits! 🚀")
                    self.position.close()
                    self._reset_trade()
                    return
                if self.data.Low[-1] <= self.sl_price:
                    print(f"🌙💥 SL HIT LONG @ {self.sl_price:.2f} | Moon Dev cut losses ✂️")
                    self.position.close()
                    self._reset_trade()
                    return
                if bars_held >= self.max_bars_in_trade:
                    print(f"🌙⏰ TIME EXIT LONG after {bars_held} bars")
                    self.position.close()
                    self._reset_trade()
                    return
                    
            elif self.position.is_short:
                if self.data.Low[-1] <= self.tp_price:
                    print(f"🌙✨ TP HIT SHORT @ {self.tp_price:.2f} | Moon Dev profits! 🚀")
                    self.position.close()
                    self._reset_trade()
                    return
                if self.data.High[-1] >= self.sl_price:
                    print(f"🌙💥 SL HIT SHORT @ {self.sl_price:.2f} | Moon Dev cut losses ✂️")
                    self.position.close()
                    self._reset_trade()
                    return
                if bars_held >= self.max_bars_in_trade:
                    print(f"🌙⏰ TIME EXIT SHORT after {bars_held} bars")
                    self.position.close()
                    self._reset_trade()
                    return
            return
        
        # Entry logic - only when flat
        if not volume_surge or not trending:
            return
        
        # Long breakout: close > highest high of last N periods (excluding current)
        prev_donchian_high = self.donchian_high[-2]
        prev_donchian_low = self.donchian_low[-2]
        
        long_breakout = current_price > prev_donchian_high and rsi_val > 50
        short_breakout = current_price < prev_donchian_low and rsi_val < 50
        
        if long_breakout:
            # Position sizing: risk 2% of equity
            equity = self.equity
            risk_amount = equity * self.risk_per_trade
            stop_distance = atr_val * self.atr_sl_mult
            if stop_distance <= 0:
                return
            position_size = int(round(risk_amount / stop_distance))
            if position_size <= 0:
                position_size = 1
            
            self.sl_price = current_price - stop_distance
            self.tp_price = current_price + (atr_val * self.atr_tp_mult)
            self.entry_bar = len(self.data)
            self.entry_price = current_price
            
            print(f"🌙🚀 LONG ENTRY @ {current_price:.2f} | Vol Surge: {current_volume:.2f} > {vol_baseline * self.volume_multiplier:.2f} | RSI: {rsi_val:.2f} | ADX: {adx_val:.2f} | Size: {position_size}")
            self.buy(size=position_size)
            
        elif short_breakout:
            equity = self.equity
            risk_amount = equity * self.risk_per_trade
            stop_distance = atr_val * self.atr_sl_mult
            if stop_distance <= 0:
                return
            position_size = int(round(risk_amount / stop_distance))
            if position_size <= 0:
                position_size = 1
            
            self.sl_price = current_price + stop_distance
            self.tp_price = current_price - (atr_val * self.atr_tp_mult)
            self.entry_bar = len(self.data)
            self.entry_price = current_price
            
            print(f"🌙🔻 SHORT ENTRY @ {current_price:.2f} | Vol Surge: {current_volume:.2f} > {vol_baseline * self.volume_multiplier:.2f} | RSI: {rsi_val:.2f} | ADX: {adx_val:.2f} | Size: {position_size}")
            self.sell(size=position_size)

    def _reset_trade(self):
        self.entry_bar = None
        self.entry_price = None
        self.tp_price = None
        self.sl_price = None


print("🌙 Moon Dev is running the backtest... 🚀✨")
bt = Backtest(
    data,
    VolumetricSurge,
    cash=1000000,
    commission=0.001,
    exclusive_orders=True
)

stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙✨ Moon Dev backtest complete! 🚀")