import numpy as np
import pandas as pd
import talib
import pywt
from backtesting import Backtest, Strategy

# 🌙 Moon Dev's WaveletFractal Strategy ✨

def wavelet_energy_ratio(close_prices, wavelet='db4', level=4, k=10):
    """Compute wavelet energy ratio E_detail3 / E_detail1"""
    n = len(close_prices)
    if n < 2**level:
        return np.nan, np.nan
    
    try:
        coeffs = pywt.wavedec(close_prices, wavelet, level=level)
        # coeffs: [a4, d4, d3, d2, d1]
        d1 = coeffs[-1]
        d3 = coeffs[-3] if len(coeffs) >= 3 else coeffs[-1]
        a4 = coeffs[0]
        
        kk = min(k, len(d1), len(d3))
        e_d1 = np.sum(d1[-kk:]**2) if kk > 0 else np.nan
        e_d3 = np.sum(d3[-kk:]**2) if kk > 0 else np.nan
        e_a = np.sum(a4[-kk:]**2) if kk > 0 else np.nan
        
        er = e_d3 / e_d1 if e_d1 > 0 else np.nan
        return er, e_a
    except Exception:
        return np.nan, np.nan


def rolling_wavelet_features(close_arr, window=128, wavelet='db4', level=4, k=10):
    """Rolling wavelet features: energy ratio and approx slope"""
    n = len(close_arr)
    er_arr = np.full(n, np.nan)
    slope_arr = np.full(n, np.nan)
    
    for i in range(window, n):
        seg = close_arr[i-window:i]
        er, _ = wavelet_energy_ratio(seg, wavelet, level, k)
        er_arr[i] = er
        
        try:
            coeffs = pywt.wavedec(seg, wavelet, level=level)
            a = coeffs[0]
            if len(a) >= 2:
                slope = a[-1] - a[-2]
                slope_arr[i] = slope
        except Exception:
            pass
    
    return er_arr, slope_arr


class WaveletFractal(Strategy):
    donchian_period = 30
    atr_period = 14
    vol_ma_period = 20
    wavelet_window = 128
    wavelet_level = 4
    wavelet_type = 'db4'
    energy_k = 10
    er_threshold = 1.5
    vol_mult = 1.5
    risk_pct = 0.01
    rr_target = 2.0
    sl_atr_mult = 1.0
    timestop_bars = 10
    fakeout_bars = 3
    
    def init(self):
        print("🌙✨ Initializing WaveletFractal Strategy... 🚀")
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        vol = self.data.Volume
        
        # Donchian channels
        self.dc_upper = self.I(talib.MAX, high, timeperiod=self.donchian_period)
        self.dc_lower = self.I(talib.MIN, low, timeperiod=self.donchian_period)
        
        # ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)
        
        # Volume MA
        self.vol_ma = self.I(talib.SMA, vol, timeperiod=self.vol_ma_period)
        
        # Wavelet features (rolling)
        close_arr = np.asarray(close, dtype=float)
        er_arr, slope_arr = rolling_wavelet_features(
            close_arr,
            window=self.wavelet_window,
            wavelet=self.wavelet_type,
            level=self.wavelet_level,
            k=self.energy_k
        )
        self.er = self.I(lambda: er_arr, name='ER')
        self.approx_slope = self.I(lambda: slope_arr, name='ApproxSlope')
        
        print("🌙 Indicators ready! Donchian, ATR, VolMA, Wavelet ER & Slope computed ✨")
    
    def next(self):
        price = self.data.Close[-1]
        
        # Skip if indicators not ready
        if np.isnan(self.er[-1]) or np.isnan(self.approx_slope[-1]):
            return
        if np.isnan(self.atr[-1]) or np.isnan(self.vol_ma[-1]):
            return
        
        # Regime filter: extreme volatility
        atr_ma = np.mean(self.atr[-20:]) if len(self.atr) >= 20 else self.atr[-1]
        if atr_ma > 0 and self.atr[-1] > 3 * atr_ma:
            return
        
        er = self.er[-1]
        slope = self.approx_slope[-1]
        vol_ok = self.data.Volume[-1] > self.vol_mult * self.vol_ma[-1]
        
        # Manage existing position
        if self.position:
            entry_price = self.trades[0].entry_price if self.trades else None
            bars_held = len(self.data) - self.trades[0].entry_bar if self.trades else 0
            
            if entry_price is not None:
                atr = self.atr[-1]
                is_long = self.position.is_long
                
                # Fakeout invalidation
                if is_long and price < self.dc_upper[-1] and bars_held <= self.fakeout_bars:
                    print(f"🌙 Fakeout! Long invalidated, exiting at {price:.2f} ⚠️")
                    self.position.close()
                    return
                if not is_long and price > self.dc_lower[-1] and bars_held <= self.fakeout_bars:
                    print(f"🌙 Fakeout! Short invalidated, exiting at {price:.2f} ⚠️")
                    self.position.close()
                    return
                
                # Time stop
                if bars_held >= self.timestop_bars:
                    if is_long and price <= entry_price:
                        print(f"🌙 Time stop hit for long at {price:.2f} ⏰")
                        self.position.close()
                        return
                    if not is_long and price >= entry_price:
                        print(f"🌙 Time stop hit for short at {price:.2f} ⏰")
                        self.position.close()
                        return
                
                # Profit target / stop loss
                if is_long:
                    tp = entry_price + self.rr_target * atr
                    sl = entry_price - self.sl_atr_mult * atr
                    if price >= tp:
                        print(f"🚀 TP hit! Long exit at {price:.2f} 💰")
                        self.position.close()
                        return
                    if price <= sl:
                        print(f"🌙 SL hit! Long exit at {price:.2f} ⚠️")
                        self.position.close()
                        return
                else:
                    tp = entry_price - self.rr_target * atr
                    sl = entry_price + self.sl_atr_mult * atr
                    if price <= tp:
                        print(f"🚀 TP hit! Short exit at {price:.2f} 💰")
                        self.position.close()
                        return
                    if price >= sl:
                        print(f"🌙 SL hit! Short exit at {price:.2f} ⚠️")
                        self.position.close()
                        return
            return
        
        # Entry logic
        long_breakout = price > self.dc_upper[-1] and self.data.Close[-2] <= self.dc_upper[-2]
        short_breakout = price < self.dc_lower[-1] and self.data.Close[-2] >= self.dc_lower[-2]
        
        if long_breakout and er > self.er_threshold and slope > 0 and vol_ok:
            size = int(round(1_000_000 / price))
            if size > 0:
                print(f"🌙🚀 LONG breakout! Price={price:.2f} ER={er:.2f} Slope={slope:.4f} Size={size}")
                self.buy(size=size)
        
        elif short_breakout and er > self.er_threshold and slope < 0 and vol_ok:
            size = int(round(1_000_000 / price))
            if size > 0:
                print(f"🌙🚀 SHORT breakout! Price={price:.2f} ER={er:.2f} Slope={slope:.4f} Size={size}")
                self.sell(size=size)


# 🌙 Load data
data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open', 'high': 'High', 'low': 'Low',
    'close': 'Close', 'volume': 'Volume'
})

if 'datetime' in data.columns:
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')

data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

print(f"🌙 Data loaded: {len(data)} bars ✨")

bt = Backtest(data, WaveletFractal, cash=1_000_000, commission=0.001, exclusive_orders=True)
stats = bt.run()
print(stats)
print(stats._strategy)