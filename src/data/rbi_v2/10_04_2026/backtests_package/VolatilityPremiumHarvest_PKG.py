import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from datetime import time

# Load and prepare data
data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Ensure proper column mapping
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
else:
    data.index = pd.to_datetime(data.index)

data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

print("🌙 Moon Dev VolatilityPremiumHarvest Backtest Initializing... ✨")
print(f"📊 Data loaded: {len(data)} bars from {data.index[0]} to {data.index[-1]} 🚀")


class VolatilityPremiumHarvest(Strategy):
    """
    Volatility Premium Harvest Strategy (Adapted for spot/futures proxy)
    
    Original thesis: Sell puts when VIX futures > 20 and OI declining.
    Adaptation: Use realized volatility proxy (ATR-based) as VIX substitute.
    When volatility is elevated AND price momentum is weakening (proxy for OI decline),
    enter a contrarian long position (simulating short put premium capture).
    """
    
    # Strategy parameters
    vol_threshold = 20.0          # Proxy for VIX futures > 20
    atr_period = 14
    ema_fast = 9
    ema_slow = 21
    rsi_period = 14
    rsi_upper = 70
    profit_target_pct = 0.60      # Target 60% of premium (mid of 50-75%)
    stop_loss_pct = 1.00          # Stop if premium doubles (2x credit = 100% loss on credit)
    risk_pct = 0.03               # 3% of equity risk per trade
    morning_start = time(9, 0)
    morning_end = time(11, 0)
    
    def init(self):
        print("🌙 Initializing Moon Dev indicators... ✨")
        
        # Realized volatility proxy (annualized)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 
                          timeperiod=self.atr_period, name='ATR')
        
        # Annualized volatility proxy
        self.realized_vol = self.I(
            lambda h, l, c: talib.ATR(h, l, c, timeperiod=14) / c * np.sqrt(252 * 96) * 100,
            self.data.High, self.data.Low, self.data.Close,
            name='RealizedVol'
        )
        
        # EMAs for trend/momentum
        self.ema_f = self.I(talib.EMA, self.data.Close, timeperiod=self.ema_fast, name='EMA_Fast')
        self.ema_s = self.I(talib.EMA, self.data.Close, timeperiod=self.ema_slow, name='EMA_Slow')
        
        # RSI
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=self.rsi_period, name='RSI')
        
        # Volume moving average (proxy for participation / OI decline)
        self.vol_ma = self.I(talib.SMA, self.data.Volume, timeperiod=20, name='VolumeMA')
        
        # Swing low for stop reference
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=10, name='SwingLow')
        
        print("✅ All indicators initialized! 🚀")
    
    def next(self):
        # Need enough data
        if len(self.data) < 30:
            return
        
        current_time = self.data.index[-1].time()
        in_morning_window = self.morning_start <= current_time <= self.morning_end
        
        # Skip if already in position
        if self.position:
            # Profit target: close at 60% of premium collected (in long equivalent: +60% move)
            # Stop loss: 2x credit (100% loss equivalent)
            entry_price = self.position.entry_price
            current_price = self.data.Close[-1]
            pnl_pct = (current_price - entry_price) / entry_price
            
            if pnl_pct >= self.profit_target_pct:
                print(f"💰 Moon Dev PROFIT TARGET hit! PnL: {pnl_pct*100:.2f}% 🌙✨")
                self.position.close()
            elif pnl_pct <= -self.stop_loss_pct:
                print(f"🛑 Moon Dev STOP LOSS triggered! PnL: {pnl_pct*100:.2f}% ⚠️")
                self.position.close()
            return
        
        # ENTRY CONDITIONS
        if not in_morning_window:
            return
        
        current_vol = self.realized_vol[-1]
        current_rsi = self.rsi[-1]
        current_ema_f = self.ema_f[-1]
        current_ema_s = self.ema_s[-1]
        current_volume = self.data.Volume[-1]
        vol_ma_val = self.vol_ma[-1]
        
        # Condition 1: Elevated volatility (proxy for VIX futures > 20)
        vol_condition = current_vol > self.vol_threshold
        
        # Condition 2: OI decline proxy = declining volume participation
        oi_decline_proxy = current_volume < vol_ma_val
        
        # Condition 3: Momentum weakening (RSI not overbought, EMAs converging/bearish)
        momentum_condition = current_rsi < self.rsi_upper and current_ema_f < current_ema_s
        
        # Condition 4: Price near swing low (simulating put strike being approached)
        price_near_low = self.data.Close[-1] <= self.swing_low[-1] * 1.005
        
        if vol_condition and oi_decline_proxy and momentum_condition and price_near_low:
            print(f"🌙✨ Moon Dev SIGNAL DETECTED! ✨🌙")
            print(f"   📈 Realized Vol: {current_vol:.2f}% (threshold: {self.vol_threshold})")
            print(f"   📉 Volume: {current_volume:.2f} < MA: {vol_ma_val:.2f} (OI decline proxy)")
            print(f"   🔻 RSI: {current_rsi:.2f} | EMA_F: {current_ema_f:.2f} < EMA_S: {current_ema_s:.2f}")
            print(f"   🎯 Price near swing low: {self.data.Close[-1]:.2f} vs {self.swing_low[-1]:.2f}")
            
            # Position sizing: 1,000,000 units as specified
            size = 1_000_000
            
            # Calculate stop and target prices
            entry = self.data.Close[-1]
            stop_price = entry * (1 - self.stop_loss_pct)
            target_price = entry * (1 + self.profit_target_pct)
            
            print(f"🚀 ENTERING LONG (proxy for short put) at {entry:.2f}")
            print(f"   🎯 Target: {target_price:.2f} | 🛑 Stop: {stop_price:.2f}")
            
            self.buy(size=size, sl=stop_price, tp=target_price)


# Run backtest
print("\n🌙✨🚀 Starting Moon Dev VolatilityPremiumHarvest Backtest 🚀✨🌙\n")

bt = Backtest(
    data,
    VolatilityPremiumHarvest,
    cash=10_000_000,
    commission=0.0002,
    exclusive=False
)

stats = bt.run()

print("\n" + "="*70)
print("🌙 MOON DEV VOLATILITY PREMIUM HARVEST — FULL STATS 🌙")
print("="*70)
print(stats)
print("\n" + "="*70)
print("🌙 STRATEGY DETAILS 🌙")
print("="*70)
print(stats._strategy)
print("\n✨ Moon Dev Backtest Complete! 🚀🌙")