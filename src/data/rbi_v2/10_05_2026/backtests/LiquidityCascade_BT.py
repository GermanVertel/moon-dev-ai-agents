import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Load and clean data
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
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
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

print("🌙 Moon Dev LiquidityCascade Backtest Initializing... 🚀")
print(f"📊 Data loaded: {len(data)} candles")
print(f"📅 Range: {data.index[0]} to {data.index[-1]}")


class LiquidityCascade(Strategy):
    """
    🌙 LiquidityCascade Strategy 🌙
    
    Since on-chain liquidation heatmap data is not available in the CSV,
    we approximate liquidation clusters using swing highs/lows and volume
    profile density. A "cluster" = price zone with high swing-point density.
    
    Long Entry:
      - Price closes above a swing-high cluster boundary (proxy for upside liquidation cluster)
      - Volume >= 1.5x 20-period average
      - ATR above threshold (volatility filter)
    
    Short Entry:
      - Price closes below a swing-low cluster boundary
      - Volume >= 1.5x 20-period average
    
    Exit:
      - Initial stop just inside broken boundary
      - Trail using recent swing lows (longs) / swing highs (shorts)
      - Take profit at 2R, scale out
      - Time stop after N candles
    """
    
    # --- Strategy Params ---
    cluster_lookback = 50        # bars used to detect swing cluster zones
    swing_period = 5             # swing high/low detection period
    volume_ma_period = 20
    volume_mult = 1.5
    atr_period = 14
    atr_min_pct = 0.003          # 0.3% ATR minimum (volatility filter)
    stop_buffer_pct = 0.004      # 0.4% beyond boundary
    rr_target = 2.0              # take profit at 2R
    time_stop_bars = 10          # exit if not 1R in N bars
    risk_pct = 0.0075            # 0.75% equity risk per trade
    max_positions = 2
    
    def init(self):
        print("🌙 Initializing indicators for LiquidityCascade...")
        
        # Volume MA
        self.vol_ma = self.I(talib.SMA, self.data.Volume, timeperiod=self.volume_ma_period)
        
        # ATR
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close,
                          timeperiod=self.atr_period)
        
        # Swing highs / lows via rolling MAX/MIN
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=self.swing_period)
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=self.swing_period)
        
        # Cluster boundaries: use longer lookback MAX/MIN as proxy for
        # liquidation cluster boundaries (dense swing zones)
        self.cluster_high = self.I(talib.MAX, self.data.High, timeperiod=self.cluster_lookback)
        self.cluster_low = self.I(talib.MIN, self.data.Low, timeperiod=self.cluster_lookback)
        
        # Recent swing low for trailing (longs) / swing high (shorts)
        self.trail_low = self.I(talib.MIN, self.data.Low, timeperiod=10)
        self.trail_high = self.I(talib.MAX, self.data.High, timeperiod=10)
        
        print("✨ Indicators ready: vol_ma, atr, swing_high/low, cluster_high/low, trail_low/high")
    
    def next(self):
        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        volume = self.data.Volume[-1]
        vol_ma = self.vol_ma[-1]
        atr = self.atr[-1]
        
        if np.isnan(atr) or np.isnan(vol_ma) or vol_ma == 0:
            return
        
        # --- Volatility filter ---
        atr_pct = atr / price
        if atr_pct < self.atr_min_pct:
            return
        
        # --- Volume surge filter ---
        volume_surge = volume >= self.volume_mult * vol_ma
        
        # --- Manage open trades ---
        for trade in self.trades:
            self._manage_trade(trade, price, atr)
        
        # --- Entry logic ---
        if len(self.trades) >= self.max_positions:
            return
        
        # Long entry: close above cluster-high boundary proxy
        cluster_high = self.cluster_high[-1]
        cluster_low = self.cluster_low[-1]
        
        if np.isnan(cluster_high) or np.isnan(cluster_low):
            return
        
        # Prior candle closed below cluster_high, current candle closes above
        prev_close = self.data.Close[-2]
        
        long_breakout = (prev_close <= cluster_high) and (price > cluster_high) and volume_surge
        short_breakout = (prev_close >= cluster_low) and (price < cluster_low) and volume_surge
        
        if long_breakout:
            stop = cluster_high * (1 - self.stop_buffer_pct)
            risk = price - stop
            if risk <= 0:
                return
            size = self._calc_size(risk)
            if size <= 0:
                return
            print(f"🚀🌙 LONG BREAKOUT | price={price:.2f} cluster_high={cluster_high:.2f} "
                  f"vol_surge={volume/vol_ma:.2f}x ATR%={atr_pct*100:.2f} size={size}")
            self.buy(size=size, sl=stop, tp=price + self.rr_target * risk)
        
        elif short_breakout:
            stop = cluster_low * (1 + self.stop_buffer_pct)
            risk = stop - price
            if risk <= 0:
                return
            size = self._calc_size(risk)
            if size <= 0:
                return
            print(f"🔻🌙 SHORT BREAKOUT | price={price:.2f} cluster_low={cluster_low:.2f} "
                  f"vol_surge={volume/vol_ma:.2f}x ATR%={atr_pct*100:.2f} size={size}")
            self.sell(size=size, sl=stop, tp=price - self.rr_target * risk)
    
    def _calc_size(self, risk_per_unit):
        equity = self.equity
        risk_amount = equity * self.risk_pct
        if risk_per_unit <= 0:
            return 0
        size = risk_amount / risk_per_unit
        # Cap to notional of 1,000,000 per rules
        max_size = int(1_000_000 / max(self.data.Close[-1], 1))
        size = min(int(round(size)), max(1, max_size))
        return int(size)
    
    def _manage_trade(self, trade, price, atr):
        # Time stop: if not reached 1R within N bars
        bars_open = len(self.data) - trade.entry_bar
        if bars_open >= self.time_stop_bars:
            risk = abs(trade.entry_price - trade.sl) if trade.sl else atr
            if risk > 0:
                r_multiple = (price - trade.entry_price) / risk if trade.is_long else \
                             (trade.entry_price - price) / risk
                if r_multiple < 1.0:
                    print(f"⏰🌙 TIME STOP | bars_open={bars_open} R={r_multiple:.2f}")
                    trade.close()
                    return
        
        # Trailing stop using recent swing lows/highs
        if trade.is_long:
            new_sl = self.trail_low[-1]
            if not np.isnan(new_sl) and new_sl > (trade.sl or 0):
                trade.sl = new_sl
        else:
            new_sl = self.trail_high[-1]
            if not np.isnan(new_sl) and (trade.sl is None or new_sl < trade.sl):
                trade.sl = new_sl


print("🌙✨ Starting LiquidityCascade backtest... 🚀")
bt = Backtest(data, LiquidityCascade, cash=1_000_000, commission=0.0002, exclusive_orders=False)
stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙✨ Backtest complete! 🚀")