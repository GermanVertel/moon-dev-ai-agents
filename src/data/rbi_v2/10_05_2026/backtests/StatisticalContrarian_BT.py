import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# 🌙 Moon Dev's StatisticalContrarian Strategy 🚀

class StatisticalContrarian(Strategy):
    # Strategy Parameters
    donchian_period = 20
    bb_period = 20
    bb_std = 2.0
    zscore_period = 100
    atr_period = 14
    adx_period = 14
    vol_ma_period = 20
    atr_stop_long = 2.0
    atr_stop_short = 1.5
    risk_per_trade = 0.01  # 1% of equity
    max_positions = 3
    max_same_direction = 2
    time_exit_bars = 10
    
    def init(self):
        print("🌙✨ Moon Dev StatisticalContrarian Strategy Initializing... 🚀")
        
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume
        
        # 📊 Donchian Channels (20-period)
        self.donchian_high = self.I(talib.MAX, high, timeperiod=self.donchian_period)
        self.donchian_low = self.I(talib.MIN, low, timeperiod=self.donchian_period)
        print("🌙 Donchian Channels calculated ✨")
        
        # 📈 Bollinger Bands (20-period, 2 std dev)
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS, close, timeperiod=self.bb_period, nbdevup=self.bb_std, nbdevdn=self.bb_std
        )
        print("🌙 Bollinger Bands calculated ✨")
        
        # 📉 ATR (14-period) for volatility
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)
        print("🌙 ATR calculated ✨")
        
        # 📊 ADX (14-period) for regime filter
        self.adx = self.I(talib.ADX, high, low, close, timeperiod=self.adx_period)
        print("🌙 ADX calculated ✨")
        
        # 📊 Volume MA (20-period)
        self.vol_ma = self.I(talib.SMA, volume, timeperiod=self.vol_ma_period)
        print("🌙 Volume MA calculated ✨")
        
        # 🧮 Z-score of returns (rolling 100-period)
        returns = np.zeros(len(close))
        returns[1:] = (close[1:] - close[:-1]) / close[:-1]
        self.returns = self.I(lambda x: x, returns, name='Returns')
        
        # Rolling mean and std of returns for z-score
        self.ret_mean = self.I(talib.SMA, returns, timeperiod=self.zscore_period)
        self.ret_std = self.I(talib.STDDEV, returns, timeperiod=self.zscore_period)
        print("🌙 Z-score components calculated ✨")
        
        # Track entry info
        self.entry_price = None
        self.entry_bar = None
        self.entry_type = None  # 'long' or 'short'
        self.stop_price = None
        self.target_price = None
        
        # For MA of close (20-period)
        self.sma20 = self.I(talib.SMA, close, timeperiod=20)
        
        print("🚀 Moon Dev StatisticalContrarian Ready! 🌙✨")
    
    def zscore(self):
        """Calculate current return z-score"""
        if self.ret_std[-1] == 0 or np.isnan(self.ret_std[-1]):
            return 0
        return (self.returns[-1] - self.ret_mean[-1]) / self.ret_std[-1]
    
    def atr_percentile(self):
        """Calculate ATR percentile rank over lookback"""
        lookback = min(100, len(self.atr) - 1)
        if lookback < 10:
            return 50
        atr_vals = self.atr[-lookback:]
        if np.all(np.isnan(atr_vals)):
            return 50
        current = self.atr[-1]
        if np.isnan(current):
            return 50
        rank = np.sum(atr_vals < current) / len(atr_vals) * 100
        return rank
    
    def count_positions(self):
        """Count current open positions"""
        long_count = sum(1 for t in self.trades if t.is_long)
        short_count = sum(1 for t in self.trades if t.is_short)
        return long_count, short_count
    
    def next(self):
        # Need enough data
        if len(self.data) < self.zscore_period + 5:
            return
        
        price = self.data.Close[-1]
        z = self.zscore()
        atr_pct = self.atr_percentile()
        adx_val = self.adx[-1]
        
        if np.isnan(z) or np.isnan(adx_val) or np.isnan(self.atr[-1]):
            return
        
        long_count, short_count = self.count_positions()
        total_positions = long_count + short_count
        
        # 🚪 EXIT LOGIC for Long positions
        for trade in self.trades:
            if trade.is_long:
                # Trailing stop at 2x ATR
                trailing_stop = self.data.Close[-1] - (self.atr_stop_long * self.atr[-1])
                if price < trailing_stop:
                    print(f"🌙 EXIT LONG: Trailing stop hit at {price:.2f} ✨")
                    trade.close()
                    continue
                # Exit if z-score > +2.0 (exhaustion)
                if z > 2.0:
                    print(f"🌙 EXIT LONG: Z-score exhaustion ({z:.2f}) 🚀")
                    trade.close()
                    continue
                # Exit if price closes below 20-period MA
                if not np.isnan(self.sma20[-1]) and price < self.sma20[-1]:
                    print(f"🌙 EXIT LONG: Price below SMA20 at {price:.2f} ✨")
                    trade.close()
                    continue
            
            elif trade.is_short:
                # Profit target at 20-period MA
                if not np.isnan(self.sma20[-1]) and price <= self.sma20[-1]:
                    print(f"🌙 EXIT SHORT: Target SMA20 hit at {price:.2f} 🚀")
                    trade.close()
                    continue
                # Stop loss at 1.5x ATR above entry
                if self.stop_price and price >= self.stop_price:
                    print(f"🌙 EXIT SHORT: Stop loss hit at {price:.2f} ✨")
                    trade.close()
                    continue
        
        # 🚀 ENTRY LOGIC
        if total_positions >= self.max_positions:
            return
        
        # LONG ENTRY (Momentum) - in-distribution breakout
        if long_count < self.max_same_direction:
            # Price breaks above upper Donchian
            breakout = price > self.donchian_high[-2]
            # In-distribution z-score
            in_dist = -1.5 <= z <= 1.5
            # ATR percentile above 30%
            vol_ok = atr_pct > 30
            # Volume confirmation
            vol_confirm = self.data.Volume[-1] > self.vol_ma[-1] if not np.isnan(self.vol_ma[-1]) else True
            
            if breakout and in_dist and vol_ok and vol_confirm:
                # Position sizing: risk 1% of equity per trade
                risk_amount = self.equity * self.risk_per_trade
                stop_distance = self.atr_stop_long * self.atr[-1]
                if stop_distance > 0:
                    position_size = int(round(risk_amount / stop_distance))
                    if position_size > 0:
                        print(f"🌙🚀 LONG ENTRY (Momentum): Price={price:.2f}, Z={z:.2f}, ATR%={atr_pct:.1f}, Size={position_size}")
                        self.buy(size=position_size)
        
        # SHORT ENTRY (Contrarian) - out-of-distribution mean reversion
        if short_count < self.max_same_direction:
            # Price above upper Bollinger Band
            above_bb = price > self.bb_upper[-1] if not np.isnan(self.bb_upper[-1]) else False
            # Out-of-distribution z-score > +2.0
            out_dist = z > 2.0
            # ADX < 25 (non-trending)
            not_trending = adx_val < 25
            
            if above_bb and out_dist and not_trending:
                # Position sizing
                risk_amount = self.equity * self.risk_per_trade
                stop_distance = self.atr_stop_short * self.atr[-1]
                if stop_distance > 0:
                    position_size = int(round(risk_amount / stop_distance))
                    if position_size > 0:
                        print(f"🌙🚀 SHORT ENTRY (Contrarian): Price={price:.2f}, Z={z:.2f}, ADX={adx_val:.1f}, Size={position_size}")
                        self.sell(size=position_size)
                        self.stop_price = price + stop_distance


# 🌙 Load and prepare data
data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
print("🌙 Loading data from Moon Dev data vault... 🚀")
data = pd.read_csv(data_path)

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper column mapping
data.columns = [col.capitalize() for col in data.columns]
if 'Datetime' in data.columns:
    data['Datetime'] = pd.to_datetime(data['Datetime'])
    data = data.set_index('Datetime')

print(f"🌙 Data loaded: {len(data)} bars ✨")
print(f"🌙 Columns: {list(data.columns)} 🚀")

# 🚀 Run Backtest
bt = Backtest(data, StatisticalContrarian, cash=1000000, commission=0.002)
stats = bt.run()
print(stats)
print(stats._strategy)