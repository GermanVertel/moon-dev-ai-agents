import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Moon Dev's Volatility Crossover Strategy 🌙
print("🌙 Moon Dev Backtest AI initializing...")
print("🚀 Loading data from the cosmos...")

data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper column mapping
data.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

# Rename to backtesting.py format
data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

print(f"✨ Data loaded: {len(data)} bars")
print(f"🌙 Date range: {data.index[0]} to {data.index[-1]}")


class VolatilityCrossover(Strategy):
    """
    Moon Dev's Volatility Crossover Strategy 🌙
    - Detects volatility regime shifts using ATR and BBW
    - Trades mean reversion on spread Z-score extremes
    - Uses price action reversal confirmation
    """
    
    # Strategy parameters
    atr_period = 14
    bb_period = 20
    bb_std = 2.0
    zscore_period = 50
    bbw_lookback = 100
    bbw_high_pct = 90
    bbw_low_pct = 50
    z_entry = 2.0
    z_exit = 0.5
    z_stop = 3.5
    time_stop_bars = 15
    risk_pct = 0.005  # 0.5% of equity
    position_size = 0.5  # fraction of equity (0 < size < 1)
    
    def init(self):
        print("🌙 Initializing Moon Dev indicators...")
        
        # ATR (14-period)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period)
        
        # Bollinger Bands
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS, self.data.Close, timeperiod=self.bb_period,
            nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0
        )
        
        # Bollinger Band Width
        def bbw_calc(upper, middle, lower):
            result = np.full(len(upper), np.nan)
            mask = middle != 0
            result[mask] = (upper[mask] - lower[mask]) / middle[mask]
            return result
        
        self.bbw = self.I(bbw_calc, self.bb_upper, self.bb_middle, self.bb_lower)
        
        # BBW percentile thresholds (rolling)
        def rolling_pct(arr, window, pct):
            s = pd.Series(arr)
            return s.rolling(window).quantile(pct / 100.0).values
        
        self.bbw_high = self.I(rolling_pct, self.bbw, self.bbw_lookback, self.bbw_high_pct)
        self.bbw_low = self.I(rolling_pct, self.bbw, self.bbw_lookback, self.bbw_low_pct)
        
        # Spread as Z-score of price deviation from rolling mean
        def rolling_zscore(arr, window):
            s = pd.Series(arr)
            mean = s.rolling(window).mean()
            std = s.rolling(window).std()
            result = (s - mean) / std
            return result.values
        
        self.spread_z = self.I(rolling_zscore, self.data.Close, self.zscore_period)
        
        # Volume average
        self.vol_ma = self.I(talib.SMA, self.data.Volume, timeperiod=20)
        
        # Candlestick patterns
        self.hammer = self.I(talib.CDLHAMMER, self.data.Open, self.data.High, self.data.Low, self.data.Close)
        self.shooting_star = self.I(talib.CDLSHOOTINGSTAR, self.data.Open, self.data.High, self.data.Low, self.data.Close)
        self.bull_engulf = self.I(talib.CDLENGULFING, self.data.Open, self.data.High, self.data.Low, self.data.Close)
        
        # Track entry bar for time stop
        self.entry_bar = None
        
        print("✨ Moon Dev indicators ready!")
    
    def next(self):
        price = self.data.Close[-1]
        
        # Skip if indicators not ready
        if (np.isnan(self.spread_z[-1]) or np.isnan(self.bbw[-1]) or 
            np.isnan(self.bbw_high[-1]) or np.isnan(self.bbw_low[-1]) or 
            np.isnan(self.atr[-1]) or np.isnan(self.vol_ma[-1])):
            return
        
        z = self.spread_z[-1]
        bbw = self.bbw[-1]
        bbw_hi = self.bbw_high[-1]
        bbw_lo = self.bbw_low[-1]
        
        high_vol_regime = bbw > bbw_hi
        low_vol_regime = bbw < bbw_lo
        
        # Volume filter
        vol_ok = self.data.Volume[-1] > self.vol_ma[-1]
        
        # Candlestick signals
        bull_reversal = (self.hammer[-1] > 0) or (self.bull_engulf[-1] > 0)
        bear_reversal = (self.shooting_star[-1] < 0) or (self.bull_engulf[-1] < 0)
        
        # === EXIT LOGIC ===
        if self.position:
            bars_held = len(self.data) - self.entry_bar if self.entry_bar else 0
            
            # Profit target: Z reverts to ±0.5
            if self.position.is_long and z >= -self.z_exit:
                print(f"🌙 Moon Dev EXIT LONG | Z reverted to {z:.2f} | Price: {price:.2f}")
                self.position.close()
                self.entry_bar = None
                return
            if self.position.is_short and z <= self.z_exit:
                print(f"🌙 Moon Dev EXIT SHORT | Z reverted to {z:.2f} | Price: {price:.2f}")
                self.position.close()
                self.entry_bar = None
                return
            
            # Stop loss: Z moves further to extreme
            if self.position.is_long and z <= -self.z_stop:
                print(f"🚨 Moon Dev STOP LONG | Z breakdown {z:.2f} | Price: {price:.2f}")
                self.position.close()
                self.entry_bar = None
                return
            if self.position.is_short and z >= self.z_stop:
                print(f"🚨 Moon Dev STOP SHORT | Z breakdown {z:.2f} | Price: {price:.2f}")
                self.position.close()
                self.entry_bar = None
                return
            
            # Time stop: 15 bars
            if bars_held >= self.time_stop_bars:
                print(f"⏰ Moon Dev TIME STOP | Held {bars_held} bars | Price: {price:.2f}")
                self.position.close()
                self.entry_bar = None
                return
            
            # Volatility exit: BBW contracts below 50th percentile
            if low_vol_regime:
                print(f"📉 Moon Dev VOL EXIT | BBW contracted | Price: {price:.2f}")
                self.position.close()
                self.entry_bar = None
                return
        
        # === ENTRY LOGIC ===
        if not self.position:
            # Long spread entry: Z < -2, high vol regime, bullish reversal, volume ok
            if z < -self.z_entry and high_vol_regime and bull_reversal and vol_ok:
                print(f"🚀 Moon Dev LONG ENTRY | Z={z:.2f} | BBW={bbw:.4f} | Price: {price:.2f}")
                self.buy(size=self.position_size)
                self.entry_bar = len(self.data)
            
            # Short spread entry: Z > +2, high vol regime, bearish reversal, volume ok
            elif z > self.z_entry and high_vol_regime and bear_reversal and vol_ok:
                print(f"🚀 Moon Dev SHORT ENTRY | Z={z:.2f} | BBW={bbw:.4f} | Price: {price:.2f}")
                self.sell(size=self.position_size)
                self.entry_bar = len(self.data)


print("🌙 Moon Dev launching backtest...")
bt = Backtest(data, VolatilityCrossover, cash=1_000_000, commission=0.0005)

stats = bt.run()
print(stats)
print(stats._strategy)
print("✨ Moon Dev backtest complete! 🚀")