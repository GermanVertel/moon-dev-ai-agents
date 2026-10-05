import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

print("🌙✨ Moon Dev's VolatilityConfluence Strategy Initializing... 🚀")

# Load data
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

print(f"🌙 Data loaded: {len(data)} rows")
print(f"🚀 Columns: {list(data.columns)}")


class VolatilityConfluence(Strategy):
    # Strategy parameters
    rsi_period = 14
    rsi_threshold = 70
    vix_threshold = 20
    vix_spike_threshold = 25
    sma_period = 200
    vix_sma_period = 20
    corr_window = 60
    corr_exit_threshold = -0.2
    atr_period = 14
    atr_multiplier = 2.0
    ema_period = 10
    risk_pct = 0.015

    def init(self):
        print("🌙✨ Initializing VolatilityConfluence indicators... 🚀")
        
        close = pd.Series(self.data.Close)
        
        # RSI(14)
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=self.rsi_period)
        
        # 200-day SMA of price
        self.sma200 = self.I(talib.SMA, self.data.Close, timeperiod=self.sma_period)
        
        # ATR(14)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period)
        
        # 10-day EMA
        self.ema10 = self.I(talib.EMA, self.data.Close, timeperiod=self.ema_period)
        
        # Synthetic VIX proxy using volatility of returns (since VIX not in data)
        # Use rolling std of returns scaled as VIX proxy
        returns = close.pct_change()
        vix_proxy = returns.rolling(20).std() * np.sqrt(252) * 100
        vix_proxy = vix_proxy.fillna(method='bfill').fillna(0)
        self.vix = self.I(lambda x: x.values, vix_proxy)
        
        # VIX 20-day SMA
        self.vix_sma = self.I(talib.SMA, self.vix, timeperiod=self.vix_sma_period)
        
        # Rolling 60-day correlation between VIX and 200-day SMA
        def rolling_corr(vix_series, sma_series):
            vix_s = pd.Series(vix_series)
            sma_s = pd.Series(sma_series)
            return vix_s.rolling(self.corr_window).corr(sma_s).fillna(0).values
        
        self.corr = self.I(rolling_corr, self.vix, self.sma200)
        
        # VIX 5-day change for regime check
        def vix_change(vix_series):
            vix_s = pd.Series(vix_series)
            return (vix_s.pct_change(5) * 100).fillna(0).values
        
        self.vix_change_5d = self.I(vix_change, self.vix)
        
        print("🌙✨ All indicators initialized! 🚀")

    def next(self):
        # Skip if indicators not ready
        if len(self.data) < self.sma_period + 10:
            return
        
        price = self.data.Close[-1]
        rsi_val = self.rsi[-1]
        sma200_val = self.sma200[-1]
        vix_val = self.vix[-1]
        vix_sma_val = self.vix_sma[-1]
        corr_val = self.corr[-1]
        atr_val = self.atr[-1]
        vix_change = self.vix_change_5d[-1]
        
        # Skip if any indicator is nan
        if (np.isnan(rsi_val) or np.isnan(sma200_val) or np.isnan(vix_val) 
            or np.isnan(corr_val) or np.isnan(atr_val) or np.isnan(vix_sma_val)):
            return
        
        # ============ ENTRY LOGIC ============
        if not self.position:
            # Entry conditions
            cond_trend = price > sma200_val
            cond_rsi = rsi_val > self.rsi_threshold
            cond_vix = vix_val < self.vix_threshold
            cond_vix_sma = vix_val < vix_sma_val
            cond_corr = corr_val < 0
            cond_vix_spike_avoid = vix_change < 20  # avoid if VIX rose >20% in 5 days
            
            if (cond_trend and cond_rsi and cond_vix and cond_corr 
                and cond_vix_spike_avoid):
                
                # Position sizing: risk 1.5% of equity
                risk_amount = self.equity * self.risk_pct
                stop_distance = self.atr_multiplier * atr_val
                
                if stop_distance > 0 and price > 0:
                    position_size = risk_amount / stop_distance
                    position_size = int(round(position_size))
                    # Cap position size
                    max_size = int(self.equity * 0.95 / price)
                    position_size = min(position_size, max_size)
                    
                    if position_size > 0:
                        self.buy(size=position_size)
                        print(f"🌙✨🚀 MOON DEV LONG ENTRY! Price={price:.2f} RSI={rsi_val:.1f} "
                              f"VIX={vix_val:.2f} Corr={corr_val:.3f} Size={position_size} "
                              f"ATR-Stop={stop_distance:.2f}")
        
        # ============ EXIT LOGIC ============
        else:
            entry_price = self.trades[-1].entry_price if self.trades else price
            stop_price = entry_price - self.atr_multiplier * atr_val
            
            # Primary exit: correlation regime shift
            exit_corr = corr_val > self.corr_exit_threshold
            
            # Stop-loss: VIX spike or price below 200 SMA
            exit_vix_spike = vix_val > self.vix_spike_threshold
            exit_below_sma = price < sma200_val
            
            # Trailing stop via ATR
            exit_atr_stop = price < stop_price
            
            # RSI profit-taking
            exit_rsi = rsi_val < 50
            
            if exit_corr:
                self.position.close()
                print(f"🌙🔴 EXIT - Correlation regime shift! Corr={corr_val:.3f} Price={price:.2f}")
            elif exit_vix_spike:
                self.position.close()
                print(f"🌙🔴 EXIT - VIX spike! VIX={vix_val:.2f} Price={price:.2f}")
            elif exit_below_sma:
                self.position.close()
                print(f"🌙🔴 EXIT - Price below 200SMA! Price={price:.2f} SMA200={sma200_val:.2f}")
            elif exit_atr_stop:
                self.position.close()
                print(f"🌙🔴 EXIT - ATR stop hit! Price={price:.2f} Stop={stop_price:.2f}")
            elif exit_rsi:
                self.position.close()
                print(f"🌙🔴 EXIT - RSI dropped below 50! RSI={rsi_val:.1f}")


# Run backtest
print("🌙✨ Running VolatilityConfluence backtest... 🚀")
bt = Backtest(data, VolatilityConfluence, cash=1_000_000, commission=0.002)
stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙✨ Backtest complete! 🚀")