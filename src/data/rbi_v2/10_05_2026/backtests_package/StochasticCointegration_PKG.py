import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# 🌙 Moon Dev Backtest Configuration
DATA_PATH = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"

print("🌙 Moon Dev: Loading BTC-USD data...")
data = pd.read_csv(DATA_PATH)

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper case mapping
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

# Parse datetime index
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

print(f"🌙 Moon Dev: Data loaded with {len(data)} bars ✨")
print(f"🚀 Moon Dev: Date range {data.index[0]} to {data.index[-1]}")


class StochasticCointegration(Strategy):
    """
    🌙 StochasticCointegration Strategy 🌙
    
    Since we only have BTC-USD data, we simulate the cointegrated pair
    by constructing a synthetic hedge using a rolling OLS regression of
    Close against a lagged/smoothed version of itself (proxy for the
    second leg X). This gives us a stationary spread to trade.
    
    In production, X would be a truly cointegrated asset (e.g., ETH).
    """
    
    # Strategy parameters
    zscore_lookback = 40      # Rolling window for z-score of spread
    stoch_k = 14              # Stochastic %K period
    stoch_d = 3               # Stochastic %D smoothing
    stoch_smooth = 3          # Stochastic slowing
    z_entry = 2.0             # Z-score entry threshold
    z_exit = 0.5              # Z-score exit threshold
    z_stop = 3.5              # Hard stop at extreme z-score
    stoch_low = 20            # Oversold threshold
    stoch_high = 80           # Overbought threshold
    time_stop_bars = 30       # Max bars held
    risk_pct = 0.02           # 2% risk per trade
    atr_period = 14
    
    def init(self):
        print("🌙 Moon Dev: Initializing StochasticCointegration indicators... ✨")
        
        close = pd.Series(self.data.Close)
        
        # --- Synthetic cointegration pair ---
        # X = smoothed proxy of price (EMA as the "other leg")
        self.ema_x = self.I(talib.EMA, self.data.Close, timeperiod=20)
        
        # Rolling OLS hedge ratio beta: Y = Close, X = ema_x
        # beta = cov(Y,X) / var(X) computed on rolling window
        def rolling_beta(y, x, window):
            y = pd.Series(y)
            x = pd.Series(x)
            cov = y.rolling(window).cov(x)
            var = x.rolling(window).var()
            beta = cov / var
            return beta.values
        
        self.beta = self.I(rolling_beta, self.data.Close, self.ema_x, self.zscore_lookback,
                           name="Beta")
        
        # Spread = Y - beta * X
        def compute_spread(y, x, beta):
            return y - beta * x
        
        self.spread = self.I(compute_spread, self.data.Close, self.ema_x, self.beta,
                             name="Spread")
        
        # --- Z-score of spread ---
        def rolling_zscore(s, window):
            s = pd.Series(s)
            mean = s.rolling(window).mean()
            std = s.rolling(window).std()
            return ((s - mean) / std).values
        
        self.zscore = self.I(rolling_zscore, self.spread, self.zscore_lookback,
                             name="ZScore")
        
        # --- Stochastic on the spread ---
        spread_series = pd.Series(self.spread)
        
        # %K on spread
        def stoch_k_func(s, period, smooth):
            s = pd.Series(s)
            lowest = s.rolling(period).min()
            highest = s.rolling(period).max()
            k = 100 * (s - lowest) / (highest - lowest + 1e-10)
            k = k.rolling(smooth).mean()
            return k.values
        
        def stoch_d_func(s, period, smooth, d_period):
            s = pd.Series(s)
            lowest = s.rolling(period).min()
            highest = s.rolling(period).max()
            k = 100 * (s - lowest) / (highest - lowest + 1e-10)
            k = k.rolling(smooth).mean()
            d = k.rolling(d_period).mean()
            return d.values
        
        self.stoch_k = self.I(stoch_k_func, self.spread, self.stoch_k,
                              self.stoch_smooth, name="StochK")
        self.stoch_d = self.I(stoch_d_func, self.spread, self.stoch_k,
                              self.stoch_smooth, self.stoch_d, name="StochD")
        
        # --- ATR of spread for risk sizing ---
        self.spread_high = self.I(talib.MAX, self.spread, timeperiod=self.atr_period)
        self.spread_low = self.I(talib.MIN, self.spread, timeperiod=self.atr_period)
        
        # Track bars in trade
        self.bars_in_trade = 0
        
        print("🌙 Moon Dev: Indicators ready! 🚀")
    
    def next(self):
        price = self.data.Close[-1]
        z = self.zscore[-1]
        k = self.stoch_k[-1]
        d = self.stoch_d[-1]
        k_prev = self.stoch_k[-2] if len(self.stoch_k) > 1 else k
        d_prev = self.stoch_d[-2] if len(self.stoch_d) > 1 else d
        
        # Skip if indicators not ready
        if np.isnan(z) or np.isnan(k) or np.isnan(d):
            return
        
        # --- Manage open position ---
        if self.position:
            self.bars_in_trade += 1
            
            # Time stop
            if self.bars_in_trade >= self.time_stop_bars:
                print(f"⏰ Moon Dev TIME STOP at bar {self.bars_in_trade}, z={z:.2f} 🌙")
                self.position.close()
                self.bars_in_trade = 0
                return
            
            # Hard stop-loss on z-score
            if abs(z) >= self.z_stop:
                print(f"🛑 Moon Dev HARD STOP z={z:.2f} >= {self.z_stop} ✨")
                self.position.close()
                self.bars_in_trade = 0
                return
            
            # Primary exit: z-score reverts to mean (crosses ±0.5)
            if self.position.is_long:
                # Long spread: exit when z rises back toward 0
                if z >= -self.z_exit:
                    print(f"🎯 Moon Dev MEAN REVERSION EXIT (long) z={z:.2f} 🌙")
                    self.position.close()
                    self.bars_in_trade = 0
                    return
                # Stochastic exit: %K crosses below %D while above 50
                if self.stoch_d[-2] < self.stoch_k[-2] and self.stoch_d[-1] > self.stoch_k[-1] and k > 50:
                    print(f"📉 Moon Dev STOCH EXIT (long) K={k:.1f} D={d:.1f} ✨")
                    self.position.close()
                    self.bars_in_trade = 0
                    return
            else:  # short
                # Short spread: exit when z falls back toward 0
                if z <= self.z_exit:
                    print(f"🎯 Moon Dev MEAN REVERSION EXIT (short) z={z:.2f} 🌙")
                    self.position.close()
                    self.bars_in_trade = 0
                    return
                # Stochastic exit: %K crosses above %D while below 50
                if self.stoch_k[-2] < self.stoch_d[-2] and self.stoch_k[-1] > self.stoch_d[-1] and k < 50:
                    print(f"📈 Moon Dev STOCH EXIT (short) K={k:.1f} D={d:.1f} ✨")
                    self.position.close()
                    self.bars_in_trade = 0
                    return
            return
        
        # --- Entry logic ---
        # Long spread: z < -2.0 AND %K crosses above %D from below 20
        long_signal = (
            z < -self.z_entry and
            self.stoch_k[-2] < self.stoch_d[-2] and self.stoch_k[-1] > self.stoch_d[-1] and
            k_prev < self.stoch_low
        )
        
        # Short spread: z > +2.0 AND %K crosses below %D from above 80
        short_signal = (
            z > self.z_entry and
            self.stoch_d[-2] < self.stoch_k[-2] and self.stoch_d[-1] > self.stoch_k[-1] and
            k_prev > self.stoch_high
        )
        
        if long_signal:
            # Position sizing: risk-based using z-score distance to stop
            risk_per_unit = abs(self.z_stop - z) * np.std(self.spread[-self.zscore_lookback:])
            if risk_per_unit <= 0:
                risk_per_unit = price * 0.02
            risk_amount = self.equity * self.risk_pct
            size = int(round(risk_amount / max(risk_per_unit, 1e-6)))
            size = max(1, min(size, 1000000))
            print(f"🚀 Moon Dev LONG SPREAD entry! z={z:.2f} K={k:.1f} D={d:.1f} size={size} 🌙")
            self.buy(size=size)
            self.bars_in_trade = 0
        
        elif short_signal:
            risk_per_unit = abs(z - (-self.z_stop)) * np.std(self.spread[-self.zscore_lookback:])
            if risk_per_unit <= 0:
                risk_per_unit = price * 0.02
            risk_amount = self.equity * self.risk_pct
            size = int(round(risk_amount / max(risk_per_unit, 1e-6)))
            size = max(1, min(size, 1000000))
            print(f"🔻 Moon Dev SHORT SPREAD entry! z={z:.2f} K={k:.1f} D={d:.1f} size={size} 🌙")
            self.sell(size=size)
            self.bars_in_trade = 0


print("🌙 Moon Dev: Starting backtest... 🚀")
bt = Backtest(
    data,
    StochasticCointegration,
    cash=1_000_000,
    commission=0.001,
    exclusive=False
)

stats = bt.run()
print(stats)
print(stats._strategy)