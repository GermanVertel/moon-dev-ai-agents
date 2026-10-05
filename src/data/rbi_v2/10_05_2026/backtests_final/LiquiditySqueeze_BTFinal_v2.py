import pandas as pd
import numpy as np
import talib
from backtesting import Backtest, Strategy
import warnings
warnings.filterwarnings('ignore')

# 🌙 Moon Dev's LiquiditySqueeze Backtest 🚀

def load_data(path):
    print("🌙 Loading data from Moon Dev's data vault...")
    data = pd.read_csv(path)
    data.columns = data.columns.str.strip().str.lower()
    data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
    
    # Ensure proper column mapping
    data = data.rename(columns={
        'datetime': 'Datetime',
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    })
    
    if 'Datetime' in data.columns:
        data['Datetime'] = pd.to_datetime(data['Datetime'])
        data = data.set_index('Datetime')
    
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    # 🌙 Force all price/volume columns to float64 (talib requires double)
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        data[col] = data[col].astype(np.float64)
    data = data.dropna()
    print(f"✨ Data loaded: {len(data)} bars from {data.index[0]} to {data.index[-1]}")
    return data


class LiquiditySqueeze(Strategy):
    """
    🌙 LiquiditySqueeze Strategy 🚀
    
    Since we don't have order book depth data in OHLCV, we proxy liquidity zones
    using swing highs/lows (where stop losses cluster). ATR contraction/expansion
    regime detection drives entries and exits.
    """
    
    # ATR regime parameters
    atr_period = 14
    atr_avg_period = 20
    contraction_mult = 0.7
    expansion_mult = 1.3
    
    # Volume confirmation
    volume_period = 20
    volume_mult = 1.5
    
    # Liquidity zone detection (proxy using swing highs/lows)
    swing_lookback = 20
    zone_proximity_pct = 0.005  # 0.5% proximity to liquidity zone
    
    # Risk management
    risk_pct = 0.02  # 2% risk per trade
    max_bars_in_trade = 20  # exit at breakeven if no expansion
    trail_atr_mult = 1.5
    
    def init(self):
        print("🌙 Initializing LiquiditySqueeze indicators...")
        
        # ATR for regime detection
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period)
        self.atr_avg = self.I(talib.SMA, self.atr, timeperiod=self.atr_avg_period)
        
        # Volume average — cast to float64 array to satisfy talib
        vol_arr = np.asarray(self.data.Volume, dtype=np.float64)
        self.vol_avg = self.I(talib.SMA, vol_arr, timeperiod=self.volume_period)
        
        # Swing highs/lows as liquidity zone proxies
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=self.swing_lookback)
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=self.swing_lookback)
        
        # For state tracking
        self.entry_bar = None
        self.entry_price = None
        self.stop_price = None
        self.trail_stop = None
        
        print("✨ Indicators ready: ATR, ATR_avg, Volume_avg, Swing High/Low")
    
    def next(self):
        # Skip if not enough data
        if len(self.data) < max(self.atr_avg_period, self.swing_lookback) + 5:
            return
        
        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        
        atr = self.atr[-1]
        atr_avg = self.atr_avg[-1]
        vol = self.data.Volume[-1]
        vol_avg = self.vol_avg[-1]
        
        if np.isnan(atr) or np.isnan(atr_avg) or atr_avg == 0:
            return
        if np.isnan(vol_avg) or vol_avg == 0:
            return
        
        # ATR regime classification
        atr_ratio = atr / atr_avg
        is_contraction = atr_ratio < self.contraction_mult
        is_expansion = atr_ratio > self.expansion_mult
        
        # Liquidity zone references (swing levels)
        liq_high = self.swing_high[-1]  # resistance / short liquidation cluster
        liq_low = self.swing_low[-1]    # support / long liquidation cluster
        
        # If in a position, manage exits
        if self.position:
            self._manage_position(price, high, low, atr, atr_avg, is_expansion)
            return
        
        # === ENTRY LOGIC ===
        
        # Long entry: contraction + breakout above liquidity zone + volume confirmation
        if is_contraction:
            # Check proximity to a liquidity zone BELOW current price (short liq cluster)
            # Proxy: price is near swing low (support zone with clustered longs)
            near_zone_below = abs(price - liq_low) / price < self.zone_proximity_pct * 2
            
            # Breakout: close above zone upper edge (swing low + small buffer)
            zone_upper = liq_low * (1 + self.zone_proximity_pct)
            breakout_up = self.data.Close[-1] > zone_upper and self.data.Close[-2] <= zone_upper
            
            # Volume confirmation
            vol_confirm = vol > self.volume_mult * vol_avg
            
            if near_zone_below and breakout_up and vol_confirm:
                stop = liq_low * (1 - self.zone_proximity_pct)  # just below zone lower edge
                risk_per_unit = price - stop
                
                if risk_per_unit > 0:
                    # Position sizing: risk_pct of equity / risk per unit
                    equity = self.equity
                    risk_amount = equity * self.risk_pct
                    position_size = int(round(risk_amount / risk_per_unit))
                    
                    if position_size > 0:
                        print(f"🚀🌙 LONG SIGNAL | Price={price:.2f} | Zone Low={liq_low:.2f} | "
                              f"Stop={stop:.2f} | ATR_ratio={atr_ratio:.2f} | Vol_ratio={vol/vol_avg:.2f} | "
                              f"Size={position_size}")
                        self.buy(size=position_size)
                        self.entry_bar = len(self.data)
                        self.entry_price = price
                        self.stop_price = stop
                        self.trail_stop = stop
                        return
        
        # Short entry: contraction + breakdown below liquidity zone + volume confirmation
        if is_contraction:
            # Proxy: price is near swing high (resistance zone with clustered shorts)
            near_zone_above = abs(liq_high - price) / price < self.zone_proximity_pct * 2
            
            # Breakdown: close below zone lower edge (swing high - small buffer)
            zone_lower = liq_high * (1 - self.zone_proximity_pct)
            breakout_down = self.data.Close[-1] < zone_lower and self.data.Close[-2] >= zone_lower
            
            vol_confirm = vol > self.volume_mult * vol_avg
            
            if near_zone_above and breakout_down and vol_confirm:
                stop = liq_high * (1 + self.zone_proximity_pct)  # just above zone upper edge
                risk_per_unit = stop - price
                
                if risk_per_unit > 0:
                    equity = self.equity
                    risk_amount = equity * self.risk_pct
                    position_size = int(round(risk_amount / risk_per_unit))
                    
                    if position_size > 0:
                        print(f"🔻🌙 SHORT SIGNAL | Price={price:.2f} | Zone High={liq_high:.2f} | "
                              f"Stop={stop:.2f} | ATR_ratio={atr_ratio:.2f} | Vol_ratio={vol/vol_avg:.2f} | "
                              f"Size={position_size}")
                        self.sell(size=position_size)
                        self.entry_bar = len(self.data)
                        self.entry_price = price
                        self.stop_price = stop
                        self.trail_stop = stop
                        return
    
    def _manage_position(self, price, high, low, atr, atr_avg, is_expansion):
        """Manage open position exits."""
        bars_in_trade = len(self.data) - self.entry_bar if self.entry_bar else 0
        
        if self.position.is_long:
            # Trailing stop using 1.5 × ATR once in profit
            if price > self.entry_price:
                new_trail = price - self.trail_atr_mult * atr
                if new_trail > self.trail_stop:
                    self.trail_stop = new_trail
            
            # Hard stop
            if low <= self.stop_price:
                print(f"🛑 LONG STOP HIT | Price={price:.2f} | Stop={self.stop_price:.2f}")
                self.position.close()
                self._reset()
                return
            
            # Trailing stop
            if low <= self.trail_stop and self.trail_stop > self.stop_price:
                print(f"📉 LONG TRAIL EXIT | Price={price:.2f} | Trail={self.trail_stop:.2f}")
                self.position.close()
                self._reset()
                return
            
            # Primary exit: ATR expansion + bearish rejection (close < open)
            bearish_rejection = self.data.Close[-1] < self.data.Open[-1] and (high - max(self.data.Open[-1], self.data.Close[-1])) > (min(self.data.Open[-1], self.data.Close[-1]) - low)
            if is_expansion and bearish_rejection:
                print(f"🎯 LONG TP (ATR expansion + rejection) | Price={price:.2f} | ATR_ratio={atr/atr_avg:.2f}")
                self.position.close()
                self._reset()
                return
            
            # Time stop: exit at breakeven if no expansion
            if bars_in_trade >= self.max_bars_in_trade and not is_expansion:
                print(f"⏰ LONG TIME EXIT (no expansion) | Price={price:.2f}")
                self.position.close()
                self._reset()
                return
        
        elif self.position.is_short:
            # Trailing stop
            if price < self.entry_price:
                new_trail = price + self.trail_atr_mult * atr
                if new_trail < self.trail_stop:
                    self.trail_stop = new_trail
            
            # Hard stop
            if high >= self.stop_price:
                print(f"🛑 SHORT STOP HIT | Price={price:.2f} | Stop={self.stop_price:.2f}")
                self.position.close()
                self._reset()
                return
            
            # Trailing stop
            if high >= self.trail_stop and self.trail_stop < self.stop_price:
                print(f"📈 SHORT TRAIL EXIT | Price={price:.2f} | Trail={self.trail_stop:.2f}")
                self.position.close()
                self._reset()
                return
            
            # Primary exit: ATR expansion + bullish rejection (close > open)
            bullish_rejection = self.data.Close[-1] > self.data.Open[-1] and (min(self.data.Open[-1], self.data.Close[-1]) - low) > (high - max(self.data.Open[-1], self.data.Close[-1]))
            if is_expansion and bullish_rejection:
                print(f"🎯 SHORT TP (ATR expansion + rejection) | Price={price:.2f} | ATR_ratio={atr/atr_avg:.2f}")
                self.position.close()
                self._reset()
                return
            
            # Time stop
            if bars_in_trade >= self.max_bars_in_trade and not is_expansion:
                print(f"⏰ SHORT TIME EXIT (no expansion) | Price={price:.2f}")
                self.position.close()
                self._reset()
                return
    
    def _reset(self):
        self.entry_bar = None
        self.entry_price = None
        self.stop_price = None
        self.trail_stop = None


# 🌙 Run the backtest 🚀
if __name__ == '__main__':
    DATA_PATH = '/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv'
    
    print("🌙✨ Moon Dev's LiquiditySqueeze Backtest Starting ✨🌙")
    data = load_data(DATA_PATH)
    
    bt = Backtest(
        data,
        LiquiditySqueeze,
        cash=1_000_000,
        commission=0.001,
        exclusive_orders=True
    )
    
    print("🚀 Running backtest...")
    stats = bt.run()
    
    print("\n" + "="*80)
    print("🌙 MOON DEV LIQUIDITYSQUEEZE - FULL STATS 🚀")
    print("="*80)
    print(stats)
    print("="*80)
    print(stats._strategy)
    print("="*80)