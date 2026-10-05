import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# ==============================================================================
# 🌙 MOON DEV'S VOLATILITY CRUSH BACKTEST 🌙
# ==============================================================================
# Strategy: Sell premium after a sharp IV decline (mean reversion in vol)
# Since we don't have options data, we proxy IV with realized volatility (ATR-based)
# and simulate the "short vol" thesis by going long the underlying when IV crushes
# (volatility crush typically coincides with upward drift / range-bound markets).
# ==============================================================================

print("🌙✨ MOON DEV BACKTEST INITIALIZING... ✨🌙")

# -------------------------------
# DATA LOADING & CLEANING
# -------------------------------
data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"

print(f"🚀 Loading data from: {data_path}")
data = pd.read_csv(data_path)

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Ensure proper column mapping (capitalize first letter)
data.columns = [col.capitalize() for col in data.columns]

# Handle datetime index
if 'Datetime' in data.columns:
    data['Datetime'] = pd.to_datetime(data['Datetime'])
    data = data.set_index('Datetime')
    data = data.sort_index()

print(f"✅ Data loaded: {len(data)} bars")
print(f"📊 Columns: {list(data.columns)}")
print(f"🕐 Date range: {data.index[0]} → {data.index[-1]}")

# -------------------------------
# STRATEGY CLASS
# -------------------------------
class VolatilityCrush(Strategy):
    """
    🌙 VolatilityCrush Strategy 🌙
    
    Core thesis: When implied volatility drops sharply (~3% over a week),
    it tends to mean-revert lower. We express a "short vol" bias by going long
    the underlying (since vol crush regimes are typically range-bound / bullish),
    with a hard stop at entry price and a profit target.
    
    Proxy for IV: ATR-based realized volatility (annualized).
    """
    
    # --- Tunable Parameters ---
    iv_lookback = 20           # window for volatility baseline
    iv_change_window = 5       # 5 trading days for weekly IV change (proxy: 5*96 bars for 15m)
    iv_drop_threshold = -0.03  # 3% weekly IV decline triggers entry
    stop_loss_pct = 0.02       # 2% hard stop (mirrors entry-price stop on short option)
    take_profit_pct = 0.03     # 3% profit target (mirrors ~50% premium decay)
    risk_per_trade = 0.02      # risk 2% of capital per trade
    
    def init(self):
        print("🌙✨ Initializing VolatilityCrush indicators... ✨🌙")
        
        # --- ATR as volatility proxy ---
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14)
        
        # --- Annualized realized volatility proxy (rolling std of log returns) ---
        close = pd.Series(self.data.Close)
        log_ret = np.log(close / close.shift(1)).fillna(0)
        self.vol = self.I(
            lambda s: pd.Series(s).rolling(self.iv_lookback).std() * np.sqrt(96 * 365),
            log_ret,
            name="RealizedVol"
        )
        
        # --- Weekly IV change (5 trading days ≈ 5*96 = 480 bars for 15m data) ---
        # We use a rolling window of 480 bars to approximate 1 trading week
        self.iv_change = self.I(
            lambda s: pd.Series(s).pct_change(480),
            self.vol,
            name="IV_Weekly_Change"
        )
        
        # --- SMA for trend filter (only trade with trend / range) ---
        self.sma_fast = self.I(talib.SMA, self.data.Close, timeperiod=20)
        self.sma_slow = self.I(talib.SMA, self.data.Close, timeperiod=50)
        
        print("✅ Indicators ready: ATR, RealizedVol, IV_Weekly_Change, SMA20, SMA50")
    
    def next(self):
        # Ensure we have enough data
        if len(self.data) < 500:
            return
        
        price = self.data.Close[-1]
        
        # Skip if indicators not ready
        if np.isnan(self.iv_change[-1]) or np.isnan(self.vol[-1]):
            return
        
        # --- ENTRY LOGIC ---
        # IV has declined ≥3% over the weekly window → volatility crush signal
        if not self.position:
            iv_crush = self.iv_change[-1] <= self.iv_drop_threshold
            # Trend filter: only enter longs in non-bearish regime (proxy for range-bound)
            trend_ok = self.sma_fast[-1] >= self.sma_slow[-1] * 0.98
            
            if iv_crush and trend_ok:
                # --- RISK-BASED POSITION SIZING ---
                # Risk 2% of equity with a 2% stop → position size = risk_amount / stop_distance
                equity = self.equity
                risk_amount = equity * self.risk_per_trade
                stop_distance = price * self.stop_loss_pct
                
                if stop_distance > 0:
                    position_size = risk_amount / stop_distance
                    position_size = int(round(position_size))
                    
                    # Cap position size to avoid over-leverage
                    max_size = int(equity / price)
                    position_size = min(position_size, max_size)
                    
                    if position_size > 0:
                        sl = price * (1 - self.stop_loss_pct)
                        tp = price * (1 + self.take_profit_pct)
                        
                        print(f"🌙✨ VOL CRUSH SIGNAL! IV Δ={self.iv_change[-1]*100:.2f}% "
                              f"| Price={price:.2f} | Size={position_size} | "
                              f"SL={sl:.2f} | TP={tp:.2f} 🚀")
                        
                        self.buy(size=position_size, sl=sl, tp=tp)
        
        # --- EXIT LOGIC (backup time-based exit) ---
        else:
            # If IV has re-expanded sharply, exit (vol spike = risk)
            if self.iv_change[-1] > 0.05:
                print(f"⚠️ Vol spike detected (IV Δ={self.iv_change[-1]*100:.2f}%) — "
                      f"exiting position 🌙")
                self.position.close()

# -------------------------------
# RUN BACKTEST
# -------------------------------
print("🌙🚀 Running VolatilityCrush backtest... 🚀🌙")

bt = Backtest(
    data,
    VolatilityCrush,
    cash=1_000_000,
    commission=0.001,
    exclusive=False
)

stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙✨ Backtest complete! Moon Dev out. ✨🌙")