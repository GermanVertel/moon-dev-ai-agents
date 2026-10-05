import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# 🌙 Moon Dev's VortexConvergence Backtest 🌙

def vortex(high, low, close, n=14):
    """Calculate Vortex Indicator VI+ and VI-"""
    high = pd.Series(high)
    low = pd.Series(low)
    close = pd.Series(close)
    
    vm_plus = np.abs(high - low.shift(1))
    vm_minus = np.abs(low - high.shift(1))
    
    tr = pd.concat([
        high - low,
        np.abs(high - close.shift(1)),
        np.abs(low - close.shift(1))
    ], axis=1).max(axis=1)
    
    tr_sum = tr.rolling(n).sum()
    
    vi_plus = vm_plus.rolling(n).sum() / tr_sum
    vi_minus = vm_minus.rolling(n).sum() / tr_sum
    
    return vi_plus.values, vi_minus.values


class VortexConvergence(Strategy):
    # Strategy parameters
    vortex_period = 14
    slope_lookback = 7
    slope_threshold = 0.0002  # 0.02% per bar
    atr_period = 14
    band_mult = 2.0
    risk_pct = 0.01
    rr_ratio = 2.0

    def init(self):
        print("🌙✨ Initializing VortexConvergence Strategy ✨🌙")
        
        # Vortex Indicator
        self.vi_plus, self.vi_minus = self.I(
            vortex,
            self.data.High, self.data.Low, self.data.Close,
            self.vortex_period,
            name='Vortex'
        )
        
        # ATR for dynamic bands and stops
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close,
                          timeperiod=self.atr_period, name='ATR')
        
        # Session VWAP (anchored daily)
        self.vwap = self.I(self._calc_vwap, name='VWAP')
        
        # VWAP slope
        self.vwap_slope = self.I(self._calc_slope, name='VWAP_Slope')
        
        # Dynamic bands
        self.upper_band = self.I(lambda: self.vwap + self.band_mult * self.atr, name='UpperBand')
        self.lower_band = self.I(lambda: self.vwap - self.band_mult * self.atr, name='LowerBand')
        
        print("🚀 Moon Dev indicators loaded! Vortex, VWAP, ATR ready 🌙")

    def _calc_vwap(self):
        """Session-anchored VWAP, reset daily"""
        price = (self.data.High + self.data.Low + self.data.Close) / 3
        vol = self.data.Volume
        dates = pd.Series(self.data.index).dt.date.values
        
        vwap_vals = np.full(len(price), np.nan)
        cum_pv = 0.0
        cum_v = 0.0
        prev_date = None
        
        for i in range(len(price)):
            d = dates[i]
            if d != prev_date:
                cum_pv = 0.0
                cum_v = 0.0
                prev_date = d
            
            p = price[i]
            v = vol[i]
            if np.isnan(p) or np.isnan(v):
                vwap_vals[i] = vwap_vals[i-1] if i > 0 else np.nan
                continue
            
            cum_pv += p * v
            cum_v += v
            vwap_vals[i] = cum_pv / cum_v if cum_v > 0 else p
        
        return vwap_vals

    def _calc_slope(self):
        """VWAP slope over lookback as pct change per bar"""
        vwap = self.vwap
        n = self.slope_lookback
        slope = np.full(len(vwap), np.nan)
        for i in range(n, len(vwap)):
            if not np.isnan(vwap[i]) and not np.isnan(vwap[i - n]) and vwap[i - n] != 0:
                slope[i] = (vwap[i] - vwap[i - n]) / vwap[i - n] / n
        return slope

    def next(self):
        if len(self.data) < max(self.vortex_period, self.slope_lookback, self.atr_period) + 5:
            return
        
        price = self.data.Close[-1]
        vwap = self.vwap[-1]
        slope = self.vwap_slope[-1]
        atr = self.atr[-1]
        vi_p = self.vi_plus[-1]
        vi_m = self.vi_minus[-1]
        vi_p_prev = self.vi_plus[-2]
        vi_m_prev = self.vi_minus[-2]
        
        if np.isnan(vwap) or np.isnan(slope) or np.isnan(atr):
            return
        
        # Vortex crossovers (replaced backtesting.lib.crossover)
        bull_cross = vi_p_prev <= vi_m_prev and vi_p > vi_m
        bear_cross = vi_m_prev <= vi_p_prev and vi_m > vi_p
        
        # VWAP slope classification
        steep_up = slope > self.slope_threshold
        steep_down = slope < -self.slope_threshold
        
        # ==================== POSITION MANAGEMENT ====================
        if self.position:
            # Long exits
            if self.position.is_long:
                # Primary: opposing Vortex crossover
                if bear_cross:
                    print(f"🌙💥 LONG EXIT: Bearish Vortex crossover @ {price:.2f}")
                    self.position.close()
                    return
                # Secondary: price closes below VWAP
                if price < vwap:
                    print(f"🌙📉 LONG EXIT: Price below VWAP @ {price:.2f}")
                    self.position.close()
                    return
                # Stop loss handled by broker via sl= parameter; no manual check needed
                # (Position object has no .sl attribute in backtesting.py)
            
            # Short exits
            if self.position.is_short:
                # Primary: opposing Vortex crossover
                if bull_cross:
                    print(f"🌙💥 SHORT EXIT: Bullish Vortex crossover @ {price:.2f}")
                    self.position.close()
                    return
                # Secondary: price closes above VWAP
                if price > vwap:
                    print(f"🌙📈 SHORT EXIT: Price above VWAP @ {price:.2f}")
                    self.position.close()
                    return
                # Stop loss handled by broker via sl= parameter
            return
        
        # ==================== ENTRY LOGIC ====================
        # Long entry: bullish Vortex crossover + steep up VWAP slope + price above VWAP
        if bull_cross and steep_up and price > vwap:
            # Avoid chasing: don't enter if already above upper band
            if price > self.upper_band[-1]:
                print(f"🌙⚠️ Skip LONG: price above upper band (chasing) @ {price:.2f}")
                return
            
            sl = min(self.data.Low[-1], vwap) - 0.5 * atr
            risk = price - sl
            if risk <= 0:
                return
            
            size = int(round(1_000_000 * self.risk_pct / risk))
            if size < 1:
                size = 1
            tp = price + self.rr_ratio * risk
            
            print(f"🌙🚀 LONG ENTRY: VI+ cross + VWAP slope UP @ {price:.2f} | SL={sl:.2f} TP={tp:.2f} Size={size}")
            self.buy(size=size, sl=sl, tp=tp)
        
        # Short entry: bearish Vortex crossover + steep down VWAP slope + price below VWAP
        elif bear_cross and steep_down and price < vwap:
            if price < self.lower_band[-1]:
                print(f"🌙⚠️ Skip SHORT: price below lower band (chasing) @ {price:.2f}")
                return
            
            sl = max(self.data.High[-1], vwap) + 0.5 * atr
            risk = sl - price
            if risk <= 0:
                return
            
            size = int(round(1_000_000 * self.risk_pct / risk))
            if size < 1:
                size = 1
            tp = price - self.rr_ratio * risk
            
            print(f"🌙🔻 SHORT ENTRY: VI- cross + VWAP slope DOWN @ {price:.2f} | SL={sl:.2f} TP={tp:.2f} Size={size}")
            self.sell(size=size, sl=sl, tp=tp)
        
        # Divergence warning
        elif bull_cross and not steep_up:
            print(f"🌙⚠️ DIVERGENCE: Bullish Vortex but VWAP slope flat/down - SKIP")
        elif bear_cross and not steep_down:
            print(f"🌙⚠️ DIVERGENCE: Bearish Vortex but VWAP slope flat/up - SKIP")


# ==================== DATA LOADING ====================
print("🌙 Moon Dev loading BTC-USD 15m data... 🚀")
data = pd.read_csv(
    '/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv'
)

# Clean columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper case mapping
data = data.rename(columns={
    'datetime': 'datetime',
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

print(f"🌙 Data loaded: {len(data)} bars ✨")

# ==================== RUN BACKTEST ====================
bt = Backtest(data, VortexConvergence, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙✨ Moon Dev VortexConvergence backtest complete! 🚀")