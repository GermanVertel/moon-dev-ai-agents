import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# ============================================================
# 🌙 Moon Dev's VolatilitySentiment Divergence Backtest 🚀
# ============================================================

print("🌙 Moon Dev: Loading BTC-USD data for VIX/COT strategy proxy...")
data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper case mapping
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

# Parse datetime index
if 'datetime' in data.columns:
    data['datetime'] = pd.to_datetime(data['datetime'])
    data.set_index('datetime', inplace=True)

data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
print(f"🌙 Moon Dev: Data loaded! Shape: {data.shape} ✨")


class VolatilitySentimentDivergence(Strategy):
    """
    🌙 VolatilitySentiment Divergence Strategy
    Combines VIX-like volatility extremes (via Bollinger Bands on price volatility proxy)
    with positioning divergence (approximated via momentum divergence).
    
    Since we don't have actual VIX/COT data in this dataset, we use:
    - Volatility proxy: ATR-based Bollinger Bands on Close
    - Sentiment divergence: RSI extremes + price momentum divergence
    """
    # Risk management
    risk_pct = 0.01          # 1% risk per trade
    atr_period = 14
    bb_period = 20
    bb_std_mult = 2.0
    rsi_period = 14
    ema_period = 20
    swing_lookback = 10
    reward_ratio = 2.0       # 2R target
    
    def init(self):
        print("🌙 Moon Dev: Initializing indicators... ✨")
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        
        # ATR for volatility & stop sizing
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)
        
        # 20-EMA trend filter
        self.ema = self.I(talib.EMA, close, timeperiod=self.ema_period)
        
        # RSI for sentiment/divergence proxy
        self.rsi = self.I(talib.RSI, close, timeperiod=self.rsi_period)
        
        # Volatility Bollinger Bands (proxy for VIX BB)
        self.bb_mid = self.I(talib.SMA, close, timeperiod=self.bb_period)
        self.bb_stddev = self.I(talib.STDDEV, close, timeperiod=self.bb_period, nbdev=1)
        
        # BB upper/lower computed from mid + k*stddev
        self.bb_upper = self.I(
            lambda m, s: m + self.bb_std_mult * s,
            self.bb_mid, self.bb_stddev
        )
        self.bb_lower = self.I(
            lambda m, s: m - self.bb_std_mult * s,
            self.bb_mid, self.bb_stddev
        )
        
        # Swing highs/lows via talib MAX/MIN
        self.swing_high = self.I(talib.MAX, high, timeperiod=self.swing_lookback)
        self.swing_low = self.I(talib.MIN, low, timeperiod=self.swing_lookback)
        
        # Volatility percentile proxy (rolling stddev of returns)
        self.vol = self.I(talib.STDDEV, close, timeperiod=self.bb_period, nbdev=1)
        
        print("🌙 Moon Dev: All indicators ready! 🚀")
    
    def next(self):
        # Need enough bars
        if len(self.data) < self.bb_period + 5:
            return
        
        price = self.data.Close[-1]
        ema = self.ema[-1]
        rsi = self.rsi[-1]
        atr = self.atr[-1]
        
        if np.isnan(atr) or atr <= 0:
            return
        
        # Skip if already in position (max 1 position)
        if self.position:
            return
        
        # ============================================
        # 🌙 Volatility Regime Detection (VIX proxy)
        # ============================================
        # High volatility = "VIX spike" = oversold S&P → potential LONG
        # Low volatility = "VIX crush" = overbought S&P → potential SHORT
        vol_now = self.vol[-1]
        vol_mean = np.nanmean(self.vol[-50:]) if len(self.vol) >= 50 else vol_now
        vol_std = np.nanstd(self.vol[-50:]) if len(self.vol) >= 50 else 0
        
        if vol_std == 0 or np.isnan(vol_std):
            return
        
        vol_z = (vol_now - vol_mean) / vol_std
        
        # ============================================
        # 🌙 Sentiment Divergence (COT proxy via RSI)
        # ============================================
        # RSI extreme + price divergence = crowd positioning extreme
        rsi_extreme_long = rsi < 30    # Crowd panic (oversold)
        rsi_extreme_short = rsi > 70   # Crowd euphoria (overbought)
        
        # ============================================
        # 🌙 Bullish Reversal Candle Confirmation
        # ============================================
        o = self.data.Open[-1]
        h = self.data.High[-1]
        l = self.data.Low[-1]
        c = self.data.Close[-1]
        body = abs(c - o)
        
        hammer = (min(o, c) - l) > 2 * body and (h - max(o, c)) < body
        bullish_engulf = (c > o and self.data.Close[-2] < self.data.Open[-2] and
                          c > self.data.Open[-2] and o < self.data.Close[-2])
        
        # ============================================
        # 🌙 Bearish Reversal Candle Confirmation
        # ============================================
        shooting_star = (h - max(o, c)) > 2 * body and (min(o, c) - l) < body
        bearish_engulf = (c < o and self.data.Close[-2] > self.data.Open[-2] and
                          c < self.data.Open[-2] and o > self.data.Close[-2])
        
        # ============================================
        # 🌙 LONG ENTRY
        # Conditions:
        # 1. Volatility extreme high (VIX spike proxy)
        # 2. RSI oversold (crowd panic proxy)
        # 3. Bullish reversal confirmation
        # 4. Price reclaiming EMA or above swing low
        # ============================================
        long_vol_signal = vol_z > 1.5
        long_sentiment = rsi_extreme_long
        long_confirm = hammer or bullish_engulf
        long_trend = price > ema or price > self.swing_low[-1]
        
        if long_vol_signal and long_sentiment and long_confirm and long_trend:
            entry = price
            stop = min(self.swing_low[-1], entry - atr) - 0.01 * atr
            risk = entry - stop
            
            if risk > 0:
                # Position sizing: risk 1% of equity
                equity = self.equity
                risk_amount = equity * self.risk_pct
                position_size = int(round(risk_amount / risk))
                position_size = max(1, min(position_size, int(equity / entry)))
                
                target = entry + self.reward_ratio * risk
                
                print(f"🌙🚀 LONG SIGNAL! Price: {entry:.2f} | "
                      f"RSI: {rsi:.1f} | VolZ: {vol_z:.2f} | "
                      f"Size: {position_size} | Stop: {stop:.2f} | Target: {target:.2f}")
                
                self.buy(size=position_size, sl=stop, tp=target)
        
        # ============================================
        # 🌙 SHORT ENTRY
        # Conditions:
        # 1. Volatility extreme low (VIX crush proxy)
        # 2. RSI overbought (crowd euphoria proxy)
        # 3. Bearish reversal confirmation
        # 4. Price rejecting EMA or below swing high
        # ============================================
        short_vol_signal = vol_z < -1.5
        short_sentiment = rsi_extreme_short
        short_confirm = shooting_star or bearish_engulf
        short_trend = price < ema or price < self.swing_high[-1]
        
        if short_vol_signal and short_sentiment and short_confirm and short_trend:
            entry = price
            stop = max(self.swing_high[-1], entry + atr) + 0.01 * atr
            risk = stop - entry
            
            if risk > 0:
                equity = self.equity
                risk_amount = equity * self.risk_pct
                position_size = int(round(risk_amount / risk))
                position_size = max(1, min(position_size, int(equity / entry)))
                
                target = entry - self.reward_ratio * risk
                
                print(f"🌙🔻 SHORT SIGNAL! Price: {entry:.2f} | "
                      f"RSI: {rsi:.1f} | VolZ: {vol_z:.2f} | "
                      f"Size: {position_size} | Stop: {stop:.2f} | Target: {target:.2f}")
                
                self.sell(size=position_size, sl=stop, tp=target)


# ============================================================
# 🌙 Run Backtest
# ============================================================
print("🌙 Moon Dev: Starting backtest... 🚀")
bt = Backtest(
    data,
    VolatilitySentimentDivergence,
    cash=1_000_000,
    commission=0.002,
    exclusive_orders=True
)

stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙 Moon Dev: Backtest complete! ✨🚀")