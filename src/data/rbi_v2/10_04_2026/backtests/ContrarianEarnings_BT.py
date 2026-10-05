import pandas as pd
import numpy as np
import talib
from backtesting import Backtest, Strategy

# ============================================================
# 🌙 MOON DEV BACKTEST AI - ContrarianEarnings Strategy 🌙
# ============================================================

print("🌙 Moon Dev Backtest AI initializing...")
print("🚀 Loading data from Moon Dev vault...")

# Data path
data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"

# Load data
data = pd.read_csv(data_path)

# Clean column names
data.columns = data.columns.str.strip().str.lower()
print(f"✨ Columns after cleaning: {list(data.columns)}")

# Drop unnamed columns
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
print(f"🧹 Dropped unnamed columns. Remaining: {list(data.columns)}")

# Ensure datetime handling
if 'datetime' in data.columns:
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')
elif 'date' in data.columns:
    data['date'] = pd.to_datetime(data['date'])
    data = data.set_index('date')

# Proper column mapping
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

# Keep only required columns
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
print(f"🌙 Data ready: {len(data)} bars from {data.index[0]} to {data.index[-1]}")

# ============================================================
# Strategy: ContrarianEarnings
# ============================================================
class ContrarianEarnings(Strategy):
    """
    🌙 Contrarian Earnings Strategy
    
    Since we don't have earnings calendar data for BTC-USD, we adapt the
    PEAD-reversal concept using a rolling 3-day underperformance window:
    if price declined over the last 3 bars (post-"event" proxy), enter long
    on the next bar open, targeting a mean-reversion bounce.
    """
    
    # Parameters
    lookback_window = 3          # 3-day observation window (post-earnings proxy)
    decline_threshold = -0.005   # -0.5% cumulative decline threshold (adapted for 15m)
    stop_loss_pct = 0.02         # 2% hard stop
    take_profit_pct = 0.04       # 4% take profit (mean-reversion target)
    max_hold_bars = 40           # Time stop (10 trading days ~ 40 15m bars)
    atr_period = 14
    atr_max_mult = 3.0           # Volatility filter
    risk_per_trade = 0.01        # 1% of equity risked per trade
    position_size = 1_000_000    # Fixed size per instructions

    def init(self):
        print("🌙 Initializing ContrarianEarnings indicators...")
        
        # ATR for volatility filter
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close,
                          timeperiod=self.atr_period)
        
        # Rolling 3-bar cumulative return proxy
        close = pd.Series(self.data.Close)
        self.ret_3 = self.I(
            lambda c: (c / c.shift(self.lookback_window) - 1) * 100,
            close,
            name="Ret3"
        )
        
        # Volume SMA for confirmation
        self.vol_sma = self.I(talib.SMA, self.data.Volume, timeperiod=20,
                              name="VolSMA")
        
        # Track entry bar for time stop
        self.entry_bar = None
        print("✨ Indicators ready: ATR, Ret3, VolSMA")

    def next(self):
        # Ensure enough data
        if len(self.data) < self.lookback_window + 2:
            return
        
        price = self.data.Close[-1]
        ret3 = self.ret_3[-1]
        atr = self.atr[-1]
        atr_avg = np.nanmean(self.atr[-20:]) if len(self.atr) >= 20 else atr
        vol = self.data.Volume[-1]
        vol_avg = self.vol_sma[-1]
        
        # Skip NaN
        if np.isnan(ret3) or np.isnan(atr) or np.isnan(atr_avg):
            return
        
        # ========================================
        # 🌙 ENTRY LOGIC
        # ========================================
        if not self.position:
            # Contrarian signal: 3-bar decline below threshold
            decline_signal = ret3 <= (self.decline_threshold * 100)
            
            # Volatility filter: don't enter during abnormal spikes
            vol_ok = atr <= (atr_avg * self.atr_max_mult) if atr_avg > 0 else True
            
            # Volume filter: selling pressure confirmation
            volume_ok = vol > (vol_avg * 0.5) if not np.isnan(vol_avg) else True
            
            if decline_signal and vol_ok and volume_ok:
                print(f"🌙✨ CONTRARIAN SIGNAL DETECTED! 3-bar return: {ret3:.2f}% | "
                      f"Price: {price:.2f} | ATR: {atr:.2f}")
                
                # Calculate stop / TP prices
                stop_price = price * (1 - self.stop_loss_pct)
                tp_price = price * (1 + self.take_profit_pct)
                
                # Position sizing: risk-based but capped by fixed size
                risk_per_unit = price - stop_price
                if risk_per_unit > 0:
                    risk_amount = self.equity * self.risk_per_trade
                    calc_size = int(round(risk_amount / risk_per_unit))
                    # Cap at 1,000,000 per instructions
                    size = min(calc_size, self.position_size)
                    size = max(size, 1)
                else:
                    size = 1
                
                print(f"🚀 ENTERING LONG | Size: {size} | Stop: {stop_price:.2f} | "
                      f"TP: {tp_price:.2f}")
                
                self.buy(size=size, sl=stop_price, tp=tp_price)
                self.entry_bar = len(self.data)
        
        # ========================================
        # 🌙 EXIT LOGIC (Time Stop)
        # ========================================
        else:
            bars_held = len(self.data) - self.entry_bar if self.entry_bar else 0
            
            if bars_held >= self.max_hold_bars:
                print(f"⏰ TIME STOP HIT | Bars held: {bars_held} | "
                      f"Price: {price:.2f} | Exiting...")
                self.position.close()
                self.entry_bar = None


# ============================================================
# 🌙 RUN BACKTEST
# ============================================================
print("\n🌙🚀 Launching Moon Dev Backtest Engine...")

bt = Backtest(
    data,
    ContrarianEarnings,
    cash=1_000_000,
    commission=0.002,
    exclusive=False
)

stats = bt.run()
print("\n" + "=" * 60)
print("🌙 MOON DEV BACKTEST RESULTS 🌙")
print("=" * 60)
print(stats)
print("\n" + "=" * 60)
print("🌙 STRATEGY DETAILS 🌙")
print("=" * 60)
print(stats._strategy)
print("\n✨ Moon Dev Backtest Complete! ✨")