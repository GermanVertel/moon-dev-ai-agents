import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

print("🌙 Moon Dev Backtest AI initializing... ✨")
print("🚀 Loading VolatilityChikou strategy... 🌙")

# Load data
data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper column mapping
data.columns = [col.capitalize() for col in data.columns]
if 'Datetime' in data.columns:
    data = data.set_index('Datetime')
data.index = pd.to_datetime(data.index)

print(f"🌙 Data loaded: {len(data)} rows ✨")
print(f"🚀 Date range: {data.index[0]} to {data.index[-1]} 🌙")


class VolatilityChikou(Strategy):
    """
    VolatilityChikou Strategy 🌙
    FCI volatility divergence + Bollinger Bands + Chikou Span confirmation
    """
    
    # Strategy parameters
    fci_vol_period = 20
    bb_period = 20
    bb_std = 2.0
    chikou_shift = 26
    sma_period = 26
    atr_period = 14
    
    risk_pct = 0.02
    reward_risk = 2.0
    
    def init(self):
        print("🌙 Initializing indicators... ✨")
        
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        
        # FCI proxy: use returns as a proxy for financial conditions changes
        # In absence of actual FCI data, we use price returns as proxy
        self.fci = self.I(lambda x: pd.Series(x).pct_change().fillna(0).values, close)
        
        # FCI changes
        self.fci_change = self.I(lambda x: pd.Series(x).diff().fillna(0).values, self.fci)
        
        # Rolling std of FCI changes = FCI volatility
        self.fci_vol = self.I(
            lambda x: pd.Series(x).rolling(self.fci_vol_period).std().fillna(0).values,
            self.fci_change
        )
        
        # Bollinger Bands on FCI volatility
        self.vol_sma = self.I(talib.SMA, self.fci_vol, timeperiod=self.bb_period)
        self.vol_std = self.I(talib.STDDEV, self.fci_vol, timeperiod=self.bb_period)
        self.vol_upper = self.I(
            lambda sma, std: sma + self.bb_std * std,
            self.vol_sma, self.vol_std
        )
        self.vol_lower = self.I(
            lambda sma, std: sma - self.bb_std * std,
            self.vol_sma, self.vol_std
        )
        
        # Chikou Span: current close shifted back 26 periods
        # At time t, Chikou = close[t+26]. We use it as confirmation:
        # bullish if close[t] > close[t-26] (equivalent to chikou above past price)
        self.chikou_past = self.I(
            lambda x: pd.Series(x).shift(self.chikou_shift).values,
            close
        )
        
        # 26-period SMA for trend filter
        self.sma26 = self.I(talib.SMA, close, timeperiod=self.sma_period)
        
        # ATR for stops
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)
        
        # Price Bollinger Bands for stop placement
        self.price_sma = self.I(talib.SMA, close, timeperiod=self.bb_period)
        self.price_std = self.I(talib.STDDEV, close, timeperiod=self.bb_period)
        self.price_upper = self.I(
            lambda sma, std: sma + self.bb_std * std,
            self.price_sma, self.price_std
        )
        self.price_lower = self.I(
            lambda sma, std: sma - self.bb_std * std,
            self.price_sma, self.price_std
        )
        
        # Swing high/low for stops
        self.swing_high = self.I(talib.MAX, high, timeperiod=10)
        self.swing_low = self.I(talib.MIN, low, timeperiod=10)
        
        print("🌙 All indicators initialized! 🚀✨")
    
    def next(self):
        # Wait for enough data
        if len(self.data) < self.chikou_shift + self.bb_period + 5:
            return
        
        price = self.data.Close[-1]
        
        # Chikou confirmation: current close vs close 26 bars ago
        chikou_bullish = price > self.chikou_past[-1]
        chikou_bearish = price < self.chikou_past[-1]
        
        # FCI volatility divergence detection
        vol_now = self.fci_vol[-1]
        vol_prev = self.fci_vol[-2]
        upper = self.vol_upper[-1]
        lower = self.vol_lower[-1]
        
        # FCI level trend (proxy: price trend)
        fci_rising = self.fci[-1] > self.fci[-5]
        fci_falling = self.fci[-1] < self.fci[-5]
        
        # Bullish divergence: vol pierces lower band while FCI rising
        # OR vol makes lower low while FCI makes higher low (simplified)
        bullish_div = (vol_now < lower and fci_rising) or \
                      (vol_now < vol_prev and self.fci[-1] > self.fci[-5])
        
        # Bearish divergence: vol pierces upper band while FCI falling
        bearish_div = (vol_now > upper and fci_falling) or \
                      (vol_now > vol_prev and self.fci[-1] < self.fci[-5])
        
        # Trend filter: price above/below 26-SMA
        above_sma = price > self.sma26[-1]
        below_sma = price < self.sma26[-1]
        
        # Long entry
        if not self.position:
            if bullish_div and chikou_bullish and above_sma:
                # Stop below swing low or lower band, whichever tighter
                stop_price = max(self.swing_low[-1], self.price_lower[-1])
                risk = price - stop_price
                if risk > 0:
                    target = price + self.reward_risk * risk
                    size = int(round(1_000_000 / price))
                    if size > 0:
                        print(f"🌙✨ LONG ENTRY @ {price:.2f} | Stop: {stop_price:.2f} | Target: {target:.2f} 🚀")
                        self.buy(size=size, sl=stop_price, tp=target)
            
            # Short entry
            elif bearish_div and chikou_bearish and below_sma:
                stop_price = min(self.swing_high[-1], self.price_upper[-1])
                risk = stop_price - price
                if risk > 0:
                    target = price - self.reward_risk * risk
                    size = int(round(1_000_000 / price))
                    if size > 0:
                        print(f"🌙✨ SHORT ENTRY @ {price:.2f} | Stop: {stop_price:.2f} | Target: {target:.2f} 🚀")
                        self.sell(size=size, sl=stop_price, tp=target)
        
        # Exit logic for open positions
        else:
            if self.position.is_long:
                # Chikou crossed below past price -> exit
                if chikou_bearish:
                    print(f"🌙 Long exit: Chikou bearish @ {price:.2f} ✨")
                    self.position.close()
            elif self.position.is_short:
                if chikou_bullish:
                    print(f"🌙 Short exit: Chikou bullish @ {price:.2f} ✨")
                    self.position.close()


print("🌙 Setting up backtest... ✨")
bt = Backtest(
    data,
    VolatilityChikou,
    cash=1_000_000,
    commission=0.002,
    exclusive_orders=True
)

print("🚀 Running backtest... 🌙")
stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙✨ Backtest complete! 🚀")