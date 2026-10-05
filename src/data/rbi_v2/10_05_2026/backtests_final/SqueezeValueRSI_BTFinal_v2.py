import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Load and prepare data
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Map columns to proper case
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

# Ensure numeric types are float64 (talib requires double)
for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    data[col] = data[col].astype(np.float64)

print("🌙✨ Moon Dev Data Loaded! ✨🌙")
print(f"📊 Total bars: {len(data)}")
print(f"📅 Date range: {data.index[0]} to {data.index[-1]}")
print(f"💰 Price range: {data['Close'].min():.2f} - {data['Close'].max():.2f}")
print("🚀 SqueezeValueRSI Strategy Initializing...")


class SqueezeValueRSI(Strategy):
    """
    SqueezeValueRSI Strategy 🌙
    Combines short-squeeze dynamics with short-term mean reversion.
    
    Since we're using BTC-USD data (no short interest data available),
    we approximate the SIR (Short Interest Ratio) using volume-based 
    proxy: high volume relative to average = crowded positioning proxy.
    """
    
    # Strategy parameters
    rsi_period = 2              # RSI period (using 2 as practical proxy for 1-day)
    rsi_oversold = 20           # Oversold threshold
    rsi_overbought = 70         # Overbought exit threshold
    volume_ma_period = 20       # Volume moving average period
    volume_spike_mult = 1.5     # Volume spike multiplier (proxy for high SIR)
    profit_target = 0.05        # +5% profit target
    stop_loss_pct = 0.03        # -3% stop loss
    time_stop_days = 3          # Time stop in days
    risk_pct = 0.02             # 2% risk per trade
    trailing_trigger = 0.03     # Trailing stop activates at +3%
    trailing_stop = 0.02        # 2% trailing stop
    
    def init(self):
        print("🌙 Initializing indicators...")
        
        # RSI indicator - ensure float64 input
        close_arr = np.asarray(self.data.Close, dtype=np.float64)
        self.rsi = self.I(talib.RSI, close_arr, timeperiod=self.rsi_period,
                          name='RSI')
        
        # Volume moving average - ensure float64 input
        vol_arr = np.asarray(self.data.Volume, dtype=np.float64)
        self.vol_ma = self.I(talib.SMA, vol_arr, timeperiod=self.volume_ma_period,
                             name='VolMA')
        
        # Average dollar volume proxy
        self.dollar_vol = self.I(lambda c, v: np.asarray(c, dtype=np.float64) * np.asarray(v, dtype=np.float64),
                                 self.data.Close, self.data.Volume,
                                 name='DollarVol')
        self.avg_dollar_vol = self.I(talib.SMA, np.asarray(self.dollar_vol, dtype=np.float64),
                                     timeperiod=20, name='AvgDollarVol')
        
        # Track entry info for time stop and trailing
        self.entry_bar = None
        self.entry_price = None
        self.trailing_active = False
        self.trailing_high = None
        
        print("✨ Indicators initialized! RSI, Volume MA, Dollar Volume ready 🚀")
    
    def next(self):
        # Skip if not enough data
        if len(self.data) < self.volume_ma_period + 2:
            return
        
        current_price = self.data.Close[-1]
        current_rsi = self.rsi[-1]
        current_vol = self.data.Volume[-1]
        avg_vol = self.vol_ma[-1]
        avg_dv = self.avg_dollar_vol[-1]
        
        # ============ POSITION MANAGEMENT ============
        if self.position:
            entry_price = self.entry_price
            pnl_pct = (current_price - entry_price) / entry_price
            bars_held = len(self.data) - self.entry_bar
            
            # Update trailing stop
            if pnl_pct >= self.trailing_trigger:
                if not self.trailing_active:
                    self.trailing_active = True
                    self.trailing_high = current_price
                    print(f"🎯 Trailing stop ACTIVATED at +{pnl_pct*100:.2f}% | Price: {current_price:.2f}")
                else:
                    self.trailing_high = max(self.trailing_high, current_price)
            
            # Check exits
            exit_reason = None
            
            # 1. Profit target
            if pnl_pct >= self.profit_target:
                exit_reason = f"💰 PROFIT TARGET +{pnl_pct*100:.2f}%"
            
            # 2. Stop loss
            elif pnl_pct <= -self.stop_loss_pct:
                exit_reason = f"🛑 STOP LOSS {pnl_pct*100:.2f}%"
            
            # 3. RSI overbought (mean reversion complete)
            elif current_rsi >= self.rsi_overbought:
                exit_reason = f"📈 RSI OVERBOUGHT ({current_rsi:.1f}) | PnL: {pnl_pct*100:.2f}%"
            
            # 4. Trailing stop
            elif self.trailing_active and self.trailing_high:
                trail_stop_price = self.trailing_high * (1 - self.trailing_stop)
                if current_price <= trail_stop_price:
                    exit_reason = f"🔻 TRAILING STOP | PnL: {pnl_pct*100:.2f}%"
            
            # 5. Time stop
            elif bars_held >= self.time_stop_days * 96:  # 96 bars per day on 15m
                exit_reason = f"⏰ TIME STOP ({bars_held} bars) | PnL: {pnl_pct*100:.2f}%"
            
            if exit_reason:
                self.position.close()
                print(f"🌙 EXIT: {exit_reason} | Price: {current_price:.2f}")
                self.entry_bar = None
                self.entry_price = None
                self.trailing_active = False
                self.trailing_high = None
        
        # ============ ENTRY LOGIC ============
        else:
            # Volume spike proxy for high short interest
            vol_spike = current_vol > (avg_vol * self.volume_spike_mult) if avg_vol > 0 else False
            
            # Liquidity filter: avg dollar volume > $5M (scaled for crypto)
            liquidity_ok = avg_dv > 5000000 if avg_dv > 0 else False
            
            # Price filter > $5
            price_ok = current_price > 5
            
            # RSI oversold
            rsi_oversold = current_rsi <= self.rsi_oversold
            
            # Entry signal
            if rsi_oversold and vol_spike and liquidity_ok and price_ok:
                # Position sizing: risk 2% of equity
                risk_amount = self.equity * self.risk_pct
                stop_distance = current_price * self.stop_loss_pct
                
                if stop_distance > 0:
                    position_size = risk_amount / stop_distance
                    position_size = int(round(position_size))
                    
                    if position_size > 0:
                        self.buy(size=position_size)
                        self.entry_bar = len(self.data)
                        self.entry_price = current_price
                        self.trailing_active = False
                        self.trailing_high = None
                        
                        print(f"🚀🌙 ENTRY SIGNAL! 🎯")
                        print(f"   💵 Price: {current_price:.2f}")
                        print(f"   📊 RSI({self.rsi_period}): {current_rsi:.2f} (OVERSOLD)")
                        print(f"   📦 Vol: {current_vol:.0f} vs Avg: {avg_vol:.0f} (Spike: {current_vol/avg_vol:.2f}x)")
                        print(f"   💰 Size: {position_size} units")
                        print(f"   🛑 Stop: {current_price*(1-self.stop_loss_pct):.2f} | 🎯 Target: {current_price*(1+self.profit_target):.2f}")


print("\n🌙✨🚀 Running SqueezeValueRSI Backtest... 🚀✨🌙\n")

bt = Backtest(
    data,
    SqueezeValueRSI,
    cash=1000000,
    commission=0.002
)

stats = bt.run()
print(stats)
print(stats._strategy)