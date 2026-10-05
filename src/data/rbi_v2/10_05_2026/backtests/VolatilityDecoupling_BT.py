import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

print("🌙 Moon Dev Backtest AI initializing... ✨")
print("🚀 Loading VolatilityDecoupling strategy...")

# Load data
data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper column mapping
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

print(f"🌙 Data loaded: {len(data)} bars ✨")
print(f"🚀 Columns: {list(data.columns)}")


class VolatilityDecoupling(Strategy):
    """
    VolatilityDecoupling Strategy 🌙
    Since we only have BTC data, we proxy:
    - S&P 500 price action -> BTC Close
    - VIX futures -> realized volatility proxy (ATR-based)
    - T-Bill yield -> rolling mean reversal proxy
    """
    
    # Strategy parameters
    zscore_window = 20
    corr_window = 20
    zscore_threshold = 1.5
    rsi_period = 14
    rsi_threshold = 55
    roc_period = 5
    atr_period = 14
    risk_pct = 0.005  # 0.5% risk per trade
    profit_target_pct = 0.004  # 0.4%
    stop_loss_pct = 0.002  # 0.2%
    time_stop_bars = 15
    position_size = 1000000

    def init(self):
        print("🌙 Initializing indicators... ✨")
        
        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        
        # RSI(14)
        self.rsi = self.I(talib.RSI, close, timeperiod=self.rsi_period)
        
        # 5-period ROC
        self.roc = self.I(talib.ROC, close, timeperiod=self.roc_period)
        
        # ATR(14)
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)
        
        # VIX proxy = normalized ATR (fear gauge proxy)
        def vix_proxy(h, l, c):
            atr = talib.ATR(h, l, c, timeperiod=14)
            return atr / c * 100
        
        self.vix_proxy = self.I(vix_proxy, high, low, close)
        
        # T-Bill yield proxy = rolling mean of returns (rate proxy)
        def tbill_proxy(c):
            returns = np.zeros(len(c))
            returns[1:] = np.diff(c) / c[:-1]
            return pd.Series(returns).rolling(10).mean().values
        
        self.tbill_proxy = self.I(tbill_proxy, close)
        
        # Normalized returns for divergence
        def sp_returns(c):
            r = np.zeros(len(c))
            r[1:] = np.diff(c) / c[:-1]
            return r
        
        sp_ret = sp_returns(np.array(close))
        
        # VIX futures returns (proxy)
        vix_arr = np.array(self.vix_proxy)
        vix_ret = np.zeros(len(vix_arr))
        vix_ret[1:] = np.diff(vix_arr) / (vix_arr[:-1] + 1e-9)
        
        # Divergence spread: VIX returns - inverse of SP returns
        divergence = vix_ret + sp_ret  # VIX up + SP up = divergence; VIX down + SP up = negative
        
        # Z-score of divergence
        def zscore(arr, window):
            s = pd.Series(arr)
            mean = s.rolling(window).mean()
            std = s.rolling(window).std()
            return ((s - mean) / (std + 1e-9)).values
        
        self.divergence = self.I(lambda: divergence)
        self.zscore = self.I(zscore, divergence, self.zscore_window)
        
        # Rolling correlation between VIX returns and T-Bill changes
        def rolling_corr(v, t, window):
            s_v = pd.Series(v)
            s_t = pd.Series(t)
            return s_v.rolling(window).corr(s_t).values
        
        tbill_arr = np.array(self.tbill_proxy)
        tbill_change = np.zeros(len(tbill_arr))
        tbill_change[1:] = np.diff(tbill_arr)
        
        self.corr = self.I(rolling_corr, vix_ret, tbill_change, self.corr_window)
        
        # Correlation slope (is it getting more negative?)
        def corr_slope(corr_arr):
            s = pd.Series(corr_arr)
            return (s - s.shift(5)).values
        
        self.corr_slope = self.I(corr_slope, self.corr)
        
        # VWAP
        def vwap(h, l, c, v):
            tp = (h + l + c) / 3
            return (tp * v).cumsum() / (v.cumsum() + 1e-9)
        
        self.vwap = self.I(vwap, high, low, close, pd.Series(self.data.Volume))
        
        # Track entry bar for time stop
        self.entry_bar = None
        self.entry_price = None
        
        print("🌙 All indicators ready! 🚀")

    def next(self):
        price = self.data.Close[-1]
        
        # Skip if not enough data
        if len(self.data) < 50:
            return
        
        # Check for valid indicator values
        if (np.isnan(self.zscore[-1]) or np.isnan(self.corr[-1]) or 
            np.isnan(self.corr_slope[-1]) or np.isnan(self.rsi[-1]) or
            np.isnan(self.roc[-1]) or np.isnan(self.atr[-1])):
            return
        
        # === EXIT LOGIC ===
        if self.position:
            bars_held = len(self.data) - self.entry_bar if self.entry_bar else 0
            pnl_pct = (price - self.entry_price) / self.entry_price
            
            # Profit target
            if pnl_pct >= self.profit_target_pct:
                print(f"🌙✨ PROFIT TARGET HIT! +{pnl_pct*100:.2f}% | Price: {price:.2f}")
                self.position.close()
                self.entry_bar = None
                return
            
            # Stop loss
            if pnl_pct <= -self.stop_loss_pct:
                print(f"🛑 STOP LOSS HIT! {pnl_pct*100:.2f}% | Price: {price:.2f}")
                self.position.close()
                self.entry_bar = None
                return
            
            # Time stop
            if bars_held >= self.time_stop_bars:
                print(f"⏰ TIME STOP! Held {bars_held} bars | PnL: {pnl_pct*100:.2f}%")
                self.position.close()
                self.entry_bar = None
                return
            
            # Signal invalidation: correlation flips positive
            if self.corr[-1] > 0:
                print(f"🚨 SIGNAL INVALIDATION: Correlation flipped positive ({self.corr[-1]:.3f})")
                self.position.close()
                self.entry_bar = None
                return
            
            # Divergence Z-score reverts to 0
            if abs(self.zscore[-1]) < 0.3:
                print(f"🔄 DIVERGENCE REVERTED: Z={self.zscore[-1]:.3f}")
                self.position.close()
                self.entry_bar = None
                return
        
        # === ENTRY LOGIC ===
        if not self.position:
            # Condition 1: Divergence Z-score > 1.5 (VIX lagging S&P upside)
            cond1 = self.zscore[-1] > self.zscore_threshold
            
            # Condition 2: Rolling correlation negative and slope decreasing (more negative)
            cond2 = self.corr[-1] < 0 and self.corr_slope[-1] < 0
            
            # Condition 3: Momentum confirmation - ROC > 0 and RSI > 55
            cond3 = self.roc[-1] > 0 and self.rsi[-1] > self.rsi_threshold
            
            # Condition 4: Price above VWAP (micro pullback to VWAP ideal)
            cond4 = price > self.vwap[-1]
            
            if cond1 and cond2 and cond3 and cond4:
                # Calculate position size based on ATR risk
                atr_val = self.atr[-1]
                if atr_val > 0:
                    risk_amount = self.equity * self.risk_pct
                    stop_distance = atr_val * 1.5
                    if stop_distance > 0:
                        position_size = int(round(risk_amount / stop_distance))
                        position_size = min(position_size, self.position_size)
                        if position_size < 1:
                            position_size = 1
                    else:
                        position_size = 1
                else:
                    position_size = 1
                
                print(f"🌙🚀 ENTRY SIGNAL DETECTED!")
                print(f"   Z-score: {self.zscore[-1]:.3f} (> {self.zscore_threshold})")
                print(f"   Correlation: {self.corr[-1]:.3f} (slope: {self.corr_slope[-1]:.4f})")
                print(f"   RSI: {self.rsi[-1]:.2f} | ROC: {self.roc[-1]:.3f}%")
                print(f"   Price: {price:.2f} | VWAP: {self.vwap[-1]:.2f}")
                print(f"   Size: {position_size} | ATR: {atr_val:.4f}")
                
                self.buy(size=position_size)
                self.entry_bar = len(self.data)
                self.entry_price = price


print("🌙 Setting up backtest... ✨")
bt = Backtest(
    data,
    VolatilityDecoupling,
    cash=1000000,
    commission=0.002,
    exclusive_orders=True
)

print("🚀 Running backtest... 🌙")
stats = bt.run()
print(stats)
print(stats._strategy)