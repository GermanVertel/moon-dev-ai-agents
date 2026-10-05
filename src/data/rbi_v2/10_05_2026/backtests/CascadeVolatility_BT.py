import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

print("🌙✨ MOON DEV BACKTEST AI - CascadeVolatility Strategy ✨🌙")

# Load data
data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
print(f"🚀 Loading data from: {data_path}")
data = pd.read_csv(data_path)

# Clean column names
data.columns = data.columns.str.strip().str.lower()
print(f"🧹 Columns after cleaning: {list(data.columns)}")

# Drop unnamed columns
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Ensure proper column mapping
data.columns = [col.capitalize() for col in data.columns]
print(f"📊 Final columns: {list(data.columns)}")

# Set datetime index if present
if 'Datetime' in data.columns:
    data['Datetime'] = pd.to_datetime(data['Datetime'])
    data = data.set_index('Datetime')
    print("⏰ Datetime index set")

print(f"🌙 Data shape: {data.shape}")
print(f"✨ Data head:\n{data.head()}")


class CascadeVolatility(Strategy):
    """
    🌙 CascadeVolatility Strategy 🌙
    
    Detects liquidation cascades via OI cliffs (proxied via volume spikes + wick candles),
    confirms with HVOL band breaks, and enters straddle-style positions.
    
    Since we only have OHLCV data, we proxy:
    - OI cliff → volume spike + large candle range (wick-heavy)
    - HVOL → rolling realized volatility with bands
    - Funding flip → momentum reversal (proxy)
    - VIX term structure → volatility regime detection
    """
    
    # Strategy parameters
    hvol_window = 20
    hvol_band_std = 1.5
    volume_spike_mult = 2.5
    oi_cliff_threshold = 0.05  # 5% proxy via range
    atr_period = 14
    risk_pct = 0.02
    profit_target = 0.75
    stop_loss_pct = 0.40
    time_stop_bars = 192  # 48 hours in 15m bars
    
    def init(self):
        print("🌙 Initializing CascadeVolatility indicators...")
        
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume
        
        # ATR for volatility and stops
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)
        
        # HVOL - rolling realized volatility (std of returns)
        def realized_vol(c, window):
            ret = np.zeros(len(c))
            ret[1:] = np.diff(np.log(c))
            rv = pd.Series(ret).rolling(window).std().values * np.sqrt(96 * 365)  # annualized 15m
            return rv
        
        self.hvol = self.I(realized_vol, close, self.hvol_window)
        
        # HVOL bands
        def hvol_upper(h, window, mult):
            s = pd.Series(h).rolling(window).mean() + mult * pd.Series(h).rolling(window).std()
            return s.values
        
        def hvol_mid(h, window):
            return pd.Series(h).rolling(window).mean().values
        
        self.hvol_upper = self.I(hvol_upper, self.hvol, self.hvol_window, self.hvol_band_std)
        self.hvol_mid = self.I(hvol_mid, self.hvol, self.hvol_window)
        
        # Volume moving average for spike detection
        self.vol_ma = self.I(talib.SMA, volume, timeperiod=20)
        
        # Candle range (proxy for wick-heavy candles / OI cliff)
        candle_range = (high - low) / close
        self.range_pct = self.I(lambda x: x, candle_range)
        
        # Range moving average
        self.range_ma = self.I(talib.SMA, candle_range, timeperiod=20)
        
        # Momentum flip proxy for funding polarity
        self.mom = self.I(talib.MOM, close, timeperiod=10)
        
        # EMA for trend
        self.ema_fast = self.I(talib.EMA, close, timeperiod=10)
        self.ema_slow = self.I(talib.EMA, close, timeperiod=30)
        
        print("✨ Indicators initialized! Ready to hunt cascades 🚀")
    
    def next(self):
        # Skip if not enough data
        if len(self.data) < max(self.hvol_window * 2, self.atr_period + 5):
            return
        
        price = self.data.Close[-1]
        atr = self.atr[-1]
        
        if np.isnan(atr) or atr <= 0:
            return
        
        # ============ ENTRY LOGIC ============
        if not self.position:
            # 1. OI Cliff proxy: volume spike + large candle range
            vol_spike = self.data.Volume[-1] > self.volume_spike_mult * self.vol_ma[-1]
            range_spike = self.range_pct[-1] > 1.5 * self.range_ma[-1]
            oi_cliff = vol_spike and range_spike
            
            # 2. HVOL breaks above upper band
            hvol_break = (not np.isnan(self.hvol_upper[-1]) and 
                          not np.isnan(self.hvol[-1]) and
                          self.hvol[-1] > self.hvol_upper[-1])
            
            # 3. Funding polarity flip proxy: momentum reversal
            mom_flip = False
            if len(self.mom) > 3:
                mom_flip = ((self.mom[-1] > 0 and self.mom[-3] < 0) or 
                            (self.mom[-1] < 0 and self.mom[-3] > 0))
            
            # Cascade detection
            if oi_cliff and hvol_break and mom_flip:
                print(f"🌙✨ CASCADE DETECTED! Price: {price:.2f} | Vol Spike: {vol_spike} | HVOL Break: {hvol_break} | Mom Flip: {mom_flip}")
                
                # Position sizing based on risk
                risk_amount = self.equity * self.risk_pct
                stop_distance = atr * 2.0
                
                if stop_distance > 0:
                    position_size = int(round(risk_amount / stop_distance))
                    position_size = max(1, min(position_size, int(self.equity / price)))
                    
                    # Determine direction based on momentum after cascade
                    if self.mom[-1] > 0:
                        print(f"🚀 LONG entry (cascade exhaustion to upside) | Size: {position_size} | Stop: {price - stop_distance:.2f} | TP: {price + atr*3:.2f}")
                        self.buy(size=position_size, sl=price - stop_distance, tp=price + atr * 3)
                    else:
                        print(f"🔻 SHORT entry (cascade exhaustion to downside) | Size: {position_size} | Stop: {price + stop_distance:.2f} | TP: {price - atr*3:.2f}")
                        self.sell(size=position_size, sl=price + stop_distance, tp=price - atr * 3)
        
        # ============ EXIT LOGIC ============
        else:
            # Time-based stop
            bars_in_trade = len(self.data) - self.trades[-1].entry_bar if self.trades else 0
            
            if bars_in_trade >= self.time_stop_bars:
                print(f"⏰ TIME STOP hit after {bars_in_trade} bars | Exiting at {price:.2f}")
                self.position.close()
                return
            
            # HVOL reversion exit
            if not np.isnan(self.hvol_mid[-1]) and not np.isnan(self.hvol[-1]):
                if self.hvol[-1] < self.hvol_mid[-1]:
                    print(f"🌊 HVOL REVERTED to mean ({self.hvol[-1]:.4f} < {self.hvol_mid[-1]:.4f}) | Exiting at {price:.2f}")
                    self.position.close()
                    return
            
            # Emergency exit: OI recovery proxy (volume collapse + small range)
            vol_collapse = self.data.Volume[-1] < 0.5 * self.vol_ma[-1]
            range_collapse = self.range_pct[-1] < 0.5 * self.range_ma[-1]
            if vol_collapse and range_collapse:
                print(f"🚨 OI RECOVERY proxy detected | Exiting at {price:.2f}")
                self.position.close()
                return


# 🌙 RUN BACKTEST 🌙
print("\n" + "="*60)
print("🌙 MOON DEV BACKTEST - CascadeVolatility 🚀")
print("="*60)

bt = Backtest(
    data,
    CascadeVolatility,
    cash=1_000_000,
    commission=0.002,
    exclusive=False
)

print("🚀 Running backtest...")
stats = bt.run()

print("\n" + "="*60)
print("🌙 FULL BACKTEST STATISTICS ✨")
print("="*60)
print(stats)
print("\n" + "="*60)
print("🌙 STRATEGY DETAILS 🚀")
print("="*60)
print(stats._strategy)
print("\n🌙✨ Moon Dev Backtest Complete! ✨🌙")