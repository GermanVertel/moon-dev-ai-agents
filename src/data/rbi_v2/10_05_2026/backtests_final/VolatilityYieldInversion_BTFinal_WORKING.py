import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# 🌙 Moon Dev's VolatilityYieldInversion Backtest 🚀

print("🌙✨ Loading Moon Dev data...")

data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper column mapping
data = data.rename(columns={
    'datetime': 'datetime',
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

print(f"🌙 Data loaded: {len(data)} bars ✨")
print(data.head())


class VolatilityYieldInversion(Strategy):
    """
    🌙 VolatilityYieldInversion Strategy 🚀
    
    Since we only have BTC-USD data (no VIX futures or Treasury yield),
    we proxy the VIX-Treasury spread using realized volatility vs a 
    synthetic short-rate proxy. The mean-reversion logic is preserved:
    when the volatility-yield spread inverts (RV proxy < rate proxy),
    we go long on the equity instrument.
    """
    
    # Strategy parameters
    ma_period = 50
    atr_period = 14
    atr_multiplier = 2.0
    risk_pct = 0.02  # 2% risk per trade
    time_stop_bars = 20
    vol_window = 30  # 30-day volatility proxy (for 15m bars, we scale)
    
    def init(self):
        print("🌙 Initializing Moon Dev indicators... ✨")
        
        # 50-period SMA on close (exit signal)
        self.ma50 = self.I(talib.SMA, self.data.Close, timeperiod=self.ma_period)
        
        # ATR(14) for stop loss
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close,
                          timeperiod=self.atr_period)
        
        # Realized volatility proxy (rolling std of returns) - acts as VIX proxy
        close = pd.Series(self.data.Close)
        returns = close.pct_change()
        self.rv = self.I(lambda x: x.rolling(self.vol_window).std() * np.sqrt(252 * 96),
                         returns, name='RV_Proxy')
        
        # Short-rate / yield proxy: use a smoothed inverse of price trend as "yield"
        # In low-vol regimes, this acts like a positive yield; in stress, it inverts.
        # We use a slow SMA of returns annualized as the "yield" proxy.
        self.yield_proxy = self.I(lambda x: x.rolling(self.vol_window).mean() * 252 * 96,
                                   returns, name='Yield_Proxy')
        
        # Spread = RV - Yield (proxy for VIX_Futures - 3M_Treasury)
        self.spread = self.I(lambda: self.rv - self.yield_proxy, name='Spread')
        
        print("🌙 Indicators ready! 🚀")
    
    def next(self):
        price = self.data.Close[-1]
        ma = self.ma50[-1]
        atr = self.atr[-1]
        spread = self.spread[-1]
        
        # Skip if indicators not ready
        if np.isnan(ma) or np.isnan(atr) or np.isnan(spread):
            return
        
        # 🌙 ENTRY: spread inversion (spread < 0) and price above MA for confirmation
        if not self.position:
            if spread < 0 and price > ma:
                # Position sizing: risk 2% of equity with 2x ATR stop
                risk_amount = self.equity * self.risk_pct
                stop_distance = self.atr_multiplier * atr
                
                if stop_distance > 0:
                    position_size = risk_amount / stop_distance
                    position_size = int(round(position_size))
                    
                    # Cap at 50% of portfolio (max exposure rule)
                    max_size = int((self.equity * 0.5) / price)
                    position_size = min(position_size, max_size)
                    
                    if position_size > 0:
                        sl_price = price - stop_distance
                        tp_price = price + (stop_distance * 2)  # 2:1 RR
                        
                        print(f"🌙✨ ENTRY LONG | Price: {price:.2f} | Spread: {spread:.6f} | "
                              f"Size: {position_size} | SL: {sl_price:.2f} | TP: {tp_price:.2f} 🚀")
                        
                        self.buy(size=position_size, sl=sl_price, tp=tp_price)
                        self.entry_bar = len(self.data)
        
        # 🌙 EXIT: price crosses below 50-MA (primary neutral signal)
        else:
            if price < ma:
                print(f"🌙🔻 EXIT | Price crossed below 50-MA | Price: {price:.2f} | MA: {ma:.2f}")
                self.position.close()
            
            # Time stop: if held too long without progress
            elif hasattr(self, 'entry_bar') and (len(self.data) - self.entry_bar) >= self.time_stop_bars:
                if self.position.pl < 0:
                    print(f"🌙⏰ TIME STOP | Held {self.time_stop_bars} bars with loss | PnL: {self.position.pl:.2f}")
                    self.position.close()


print("🌙✨ Running Moon Dev backtest... 🚀")

bt = Backtest(data, VolatilityYieldInversion, cash=1_000_000, commission=0.001)

stats = bt.run()

print("\n" + "="*60)
print("🌙✨ MOON DEV BACKTEST RESULTS ✨🌙")
print("="*60)
print(stats)
print("\n" + "="*60)
print("🌙 STRATEGY DETAILS 🚀")
print("="*60)
print(stats._strategy)
print("="*60)