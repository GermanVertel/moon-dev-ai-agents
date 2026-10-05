import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy
import warnings
warnings.filterwarnings('ignore')

# Load data
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')

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
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

print("🌙 Moon Dev VolumetricVolatility Backtest Initializing... ✨")
print(f"📊 Data loaded: {len(data)} bars")
print(f"🚀 Starting backtest...\n")


class VolumetricVolatility(Strategy):
    # Strategy parameters
    volume_ma_period = 20
    atr_period = 14
    vol_lookback = 30
    vol_expansion_threshold = 1.5
    risk_pct = 0.02
    time_stop_bars = 15
    consolidation_period = 20
    
    def init(self):
        # Volume indicators
        self.volume_ma = self.I(talib.SMA, self.data.Volume, timeperiod=self.volume_ma_period)
        
        # ATR volatility indicator
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period)
        self.atr_ma = self.I(talib.SMA, self.atr, timeperiod=self.volume_ma_period)
        
        # Rolling min/max of ATR (volatility reference range)
        self.atr_min = self.I(talib.MIN, self.atr, timeperiod=self.vol_lookback)
        self.atr_max = self.I(talib.MAX, self.atr, timeperiod=self.vol_lookback)
        
        # Consolidation high/low
        self.consol_high = self.I(talib.MAX, self.data.High, timeperiod=self.consolidation_period)
        self.consol_low = self.I(talib.MIN, self.data.Low, timeperiod=self.consolidation_period)
        
        # Bollinger Band Width for additional volatility confirmation
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS, self.data.Close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
        )
        self.bb_width = self.I(lambda u, l, m: (u - l) / m, self.bb_upper, self.bb_lower, self.bb_middle)
        self.bb_width_ma = self.I(talib.SMA, self.bb_width, timeperiod=20)
        
        # Trade tracking
        self.entry_bar = None
        self.entry_price = None
        self.stop_price = None
        self.target_price = None
        self.trade_direction = None
        
        print("🌙 Indicators initialized successfully! ✨")
        print(f"   📈 Volume MA period: {self.volume_ma_period}")
        print(f"   📊 ATR period: {self.atr_period}")
        print(f"   🔍 Volatility lookback: {self.vol_lookback}")
        print(f"   ⚡ Volatility expansion threshold: {self.vol_expansion_threshold}x\n")
    
    def next(self):
        # Skip if not enough data
        if len(self.data) < max(self.vol_lookback, self.consolidation_period, 30) + 5:
            return
        
        current_price = self.data.Close[-1]
        current_volume = self.data.Volume[-1]
        current_atr = self.atr[-1]
        current_atr_ma = self.atr_ma[-1]
        
        # Skip if indicators are nan
        if np.isnan(current_atr) or np.isnan(current_atr_ma) or np.isnan(self.volume_ma[-1]):
            return
        
        # Manage existing position
        if self.position:
            self._manage_position()
            return
        
        # === VOLUME PATTERN DETECTION ===
        # Check for rising volume (accumulation/distribution)
        vol_rising = current_volume > self.volume_ma[-1]
        vol_expanding = (len(self.data.Volume) > 3 and 
                        self.data.Volume[-1] > self.data.Volume[-2] > self.data.Volume[-3])
        
        # Volume divergence: price relatively flat but volume rising
        price_range = self.data.High[-1] - self.data.Low[-1]
        avg_price_range = np.mean([self.data.High[i] - self.data.Low[i] for i in range(-5, 0)])
        price_flat = price_range < avg_price_range * 1.2
        
        # Volume climax followed by contraction
        vol_climax = (len(self.data.Volume) > 5 and 
                     self.data.Volume[-2] > self.volume_ma[-2] * 2 and
                     current_volume < self.data.Volume[-2] * 0.7)
        
        volume_pattern_forming = (vol_rising or vol_expanding) and not vol_climax
        
        # === VOLATILITY EXPANSION DETECTION ===
        # ATR expansion above its rolling mean
        atr_expansion = current_atr > current_atr_ma * self.vol_expansion_threshold
        
        # BB width expansion
        bb_expansion = False
        if not np.isnan(self.bb_width[-1]) and not np.isnan(self.bb_width_ma[-1]):
            bb_expansion = self.bb_width[-1] > self.bb_width_ma[-1] * 1.2
        
        volatility_expanding = atr_expansion or bb_expansion
        
        # Avoid volatility climax (extreme highs)
        volatility_climax = False
        if not np.isnan(self.atr_max[-1]) and self.atr_max[-1] > 0:
            volatility_climax = current_atr > self.atr_max[-1] * 0.95
        
        volatility_valid = volatility_expanding and not volatility_climax
        
        # === DUAL CONFIRMATION ===
        if not (volume_pattern_forming and volatility_valid):
            return
        
        # === ENTRY LOGIC ===
        # Long entry: price breaks above consolidation high with bullish close
        bullish_close = self.data.Close[-1] > self.data.Open[-1]
        bearish_close = self.data.Close[-1] < self.data.Open[-1]
        
        # Volume direction bias
        up_volume = bullish_close and vol_rising
        down_volume = bearish_close and vol_rising
        
        # Breakout detection
        breakout_up = current_price > self.consol_high[-2] if len(self.data) > 2 else False
        breakout_down = current_price < self.consol_low[-2] if len(self.data) > 2 else False
        
        # Position sizing: risk-based with ATR
        if current_atr > 0:
            risk_amount = self.equity * self.risk_pct
            # Stop distance based on ATR
            stop_distance = current_atr * 2
            position_size = int(round(risk_amount / stop_distance))
            # Cap position size
            position_size = min(position_size, int(self.equity / current_price))
            position_size = max(position_size, 1)
        else:
            position_size = 1
        
        # LONG ENTRY
        if up_volume and breakout_up:
            # Stop below prior volatility minimum (ATR min level)
            stop_price = current_price - (current_atr * 2)
            # Target: distance proportional to prior volatility range
            vol_range = self.atr_max[-1] - self.atr_min[-1] if not np.isnan(self.atr_max[-1]) else current_atr * 3
            target_price = current_price + max(vol_range * 1.5, current_atr * 3)
            
            print(f"🌙🚀 LONG SIGNAL! Price: {current_price:.2f} | Vol Ratio: {current_volume/self.volume_ma[-1]:.2f}x | ATR Ratio: {current_atr/current_atr_ma:.2f}x")
            print(f"   💰 Size: {position_size} | Stop: {stop_price:.2f} | Target: {target_price:.2f}")
            
            self.buy(size=position_size)
            self.entry_bar = len(self.data)
            self.entry_price = current_price
            self.stop_price = stop_price
            self.target_price = target_price
            self.trade_direction = 'long'
        
        # SHORT ENTRY
        elif down_volume and breakout_down:
            stop_price = current_price + (current_atr * 2)
            vol_range = self.atr_max[-1] - self.atr_min[-1] if not np.isnan(self.atr_max[-1]) else current_atr * 3
            target_price = current_price - max(vol_range * 1.5, current_atr * 3)
            
            print(f"🌙🔻 SHORT SIGNAL! Price: {current_price:.2f} | Vol Ratio: {current_volume/self.volume_ma[-1]:.2f}x | ATR Ratio: {current_atr/current_atr_ma:.2f}x")
            print(f"   💰 Size: {position_size} | Stop: {stop_price:.2f} | Target: {target_price:.2f}")
            
            self.sell(size=position_size)
            self.entry_bar = len(self.data)
            self.entry_price = current_price
            self.stop_price = stop_price
            self.target_price = target_price
            self.trade_direction = 'short'
    
    def _manage_position(self):
        current_price = self.data.Close[-1]
        current_atr = self.atr[-1]
        bars_in_trade = len(self.data) - self.entry_bar if self.entry_bar else 0
        
        if np.isnan(current_atr):
            return
        
        # Trailing stop based on volatility
        if self.trade_direction == 'long':
            # Trail stop up as volatility expands
            new_stop = current_price - (current_atr * 2)
            if new_stop > self.stop_price:
                self.stop_price = new_stop
                print(f"🌙📈 Trailing stop updated to {self.stop_price:.2f}")
            
            # Check stop loss
            if current_price <= self.stop_price:
                print(f"🌙🛑 LONG STOP HIT at {current_price:.2f} (stop: {self.stop_price:.2f})")
                self.position.close()
                self._reset_trade()
                return
            
            # Check target
            if current_price >= self.target_price:
                print(f"🌙🎯 LONG TARGET HIT at {current_price:.2f} (target: {self.target_price:.2f})")
                self.position.close()
                self._reset_trade()
                return
            
            # Volatility contraction exit: ATR contracts back toward prior min
            if not np.isnan(self.atr_min[-1]) and current_atr < self.atr_min[-1] * 1.1:
                print(f"🌙📉 Volatility contraction exit LONG at {current_price:.2f}")
                self.position.close()
                self._reset_trade()
                return
        
        elif self.trade_direction == 'short':
            new_stop = current_price + (current_atr * 2)
            if new_stop < self.stop_price:
                self.stop_price = new_stop
                print(f"🌙📉 Trailing stop updated to {self.stop_price:.2f}")
            
            if current_price >= self.stop_price:
                print(f"🌙🛑 SHORT STOP HIT at {current_price:.2f} (stop: {self.stop_price:.2f})")
                self.position.close()
                self._reset_trade()
                return
            
            if current_price <= self.target_price:
                print(f"🌙🎯 SHORT TARGET HIT at {current_price:.2f} (target: {self.target_price:.2f})")
                self.position.close()
                self._reset_trade()
                return
            
            if not np.isnan(self.atr_min[-1]) and current_atr < self.atr_min[-1] * 1.1:
                print(f"🌙📈 Volatility contraction exit SHORT at {current_price:.2f}")
                self.position.close()
                self._reset_trade()
                return
        
        # Time stop
        if bars_in_trade >= self.time_stop_bars:
            print(f"🌙⏰ TIME STOP ({bars_in_trade} bars) at {current_price:.2f}")
            self.position.close()
            self._reset_trade()
            return
    
    def _reset_trade(self):
        self.entry_bar = None
        self.entry_price = None
        self.stop_price = None
        self.target_price = None
        self.trade_direction = None


# Run backtest
print("🌙✨🚀 Starting Moon Dev VolumetricVolatility Backtest 🚀✨🌙\n")

bt = Backtest(
    data,
    VolumetricVolatility,
    cash=1_000_000,
    commission=0.002,
    exclusive_orders=True
)

stats = bt.run()
print("\n" + "="*80)
print("🌙 MOON DEV VOLUMETRIC VOLATILITY - FULL BACKTEST STATS 🌙")
print("="*80)
print(stats)
print("\n" + "="*80)
print("🌙 STRATEGY DETAILS 🌙")
print("="*80)
print(stats._strategy)
print("="*80)
print("🌙✨ Backtest complete! Moon Dev out! ✨🌙")