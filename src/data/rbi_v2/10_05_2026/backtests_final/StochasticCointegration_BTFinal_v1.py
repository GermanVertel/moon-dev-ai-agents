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
    """
    
    # Strategy parameters
    zscore_lookback = 40
    stoch_k = 14
    stoch_d = 3
    stoch_smooth = 3
    z_entry = 2.0
    z_exit = 0.5
    z_stop = 3.5
    stoch_low = 20
    stoch_high = 80
    time_stop_bars = 30
    risk_pct = 0.02
    atr_period = 14
    
    def init(self):
        print("🌙 Moon Dev: Initializing StochasticCointegration indicators... ✨")
        
        # --- Synthetic cointegration pair ---
        self.ema_x = self.I(talib.EMA, self.data.Close, timeperiod=20)
        
        def rolling_beta(y, x, window):
            y = pd.Series(y)
            x = pd.Series(x)
            cov = y.rolling(window).cov(x)
            var = x.rolling(window).var()
            beta = cov / var
            return beta.values
        
        self.beta = self.I(rolling_beta, self.data.Close, self.ema_x, self.zscore_lookback,
                           name="Beta")
        
        def compute_spread(y, x, beta):
            return y - beta * x
        
        self.spread = self.I(compute_spread, self.data.Close, self.ema_x, self.beta,
                             name="Spread")
        
        def rolling_zscore(s, window):
            s = pd.Series(s)
            mean = s.rolling(window).mean()
            std = s.rolling(window).std()
            return ((s - mean) / std).values
        
        self.zscore = self.I(rolling_zscore, self.spread, self.zscore_lookback,
                             name="ZScore")
        
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
        
        self.spread_high = self.I(talib.MAX, self.spread, timeperiod=self.atr_period)
        self.spread_low = self.I(talib.MIN, self.spread, timeperiod=self.atr_period)
        
        self.bars_in_trade = 0
        
        print("🌙 Moon Dev: Indicators ready! 🚀")
    
    def next(self):
        price = self.data.Close[-1]
        z = self.zscore[-1]
        k = self.stoch_k[-1]
        d = self.stoch_d[-1]
        k_prev = self.stoch_k[-2] if len(self.stoch_k) > 1 else k
        d_prev = self.stoch_d[-2] if len(self.stoch_d) > 1 else d
        
        if np.isnan(z) or np.isnan(k) or np.isnan(d):
            return
        
        # --- Manage open position ---
        if self.position:
            self.bars_in_trade += 1
            
            if self.bars_in_trade >= self.time_stop_bars:
                print(f"⏰ Moon Dev TIME STOP at bar {self.bars_in_trade}, z={z:.2f} 🌙")
                self.position.close()
                self.bars_in_trade = 0
                return
            
            if abs(z) >= self.z_stop:
                print(f"🛑 Moon Dev HARD STOP z={z:.2f} >= {self.z_stop} ✨")
                self.position.close()
                self.bars_in_trade = 0
                return
            
            if self.position.is_long:
                if z >= -self.z_exit:
                    print(f"🎯 Moon Dev MEAN REVERSION EXIT (long) z={z:.2f} 🌙")
                    self.position.close()
                    self.bars_in_trade = 0
                    return
                if self.stoch_d[-2] < self.stoch_k[-2] and self.stoch_d[-1] > self.stoch_k[-1] and k > 50:
                    print(f"📉 Moon Dev STOCH EXIT (long) K={k:.1f} D={d:.1f} ✨")
                    self.position.close()
                    self.bars_in_trade = 0
                    return
            else:
                if z <= self.z_exit:
                    print(f"🎯 Moon Dev MEAN REVERSION EXIT (short) z={z:.2f} 🌙")
                    self.position.close()
                    self.bars_in_trade = 0
                    return
                if self.stoch_k[-2] < self.stoch_d[-2] and self.stoch_k[-1] > self.stoch_d[-1] and k < 50:
                    print(f"📈 Moon Dev STOCH EXIT (short) K={k:.1f} D={d:.1f} ✨")
                    self.position.close()
                    self.bars_in_trade = 0
                    return
            return
        
        # --- Entry logic ---
        long_signal = (
            z < -self.z_entry and
            self.stoch_k[-2] < self.stoch_d[-2] and self.stoch_k[-1] > self.stoch_d[-1] and
            k_prev < self.stoch_low
        )
        
        short_signal = (
            z > self.z_entry and
            self.stoch_d[-2] < self.stoch_k[-2] and self.stoch_d[-1] > self.stoch_k[-1] and
            k_prev > self.stoch_high
        )
        
        if long_signal:
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
    commission=0.001
)

stats = bt.run()
print(stats)
print(stats._strategy)