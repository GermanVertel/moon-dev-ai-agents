import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# 🌙 Moon Dev's KineticCrossover Backtest 🌙

print("🌙 Moon Dev is loading the cosmic data... ✨")
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper column mapping
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

# Set datetime index
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

print(f"🌙 Data loaded: {len(data)} bars from {data.index[0]} to {data.index[-1]} ✨")


class KineticCrossover(Strategy):
    """
    🌙 KineticCrossover Strategy 🌙
    - Entry: 50-SMA crosses above 200-EMA (long only)
    - Stop-Loss: Recent swing low (structural)
    - Take-Profit: Entry + 0.20 * ATR (tight volatility target)
    - Position sizing: fractional equity
    """
    sma_period = 50
    ema_period = 200
    atr_period = 14
    swing_lookback = 20
    tp_atr_mult = 0.20
    risk_pct = 0.01  # 1% risk per trade (informational)

    def init(self):
        print("🌙 Initializing cosmic indicators... ✨")
        self.sma = self.I(talib.SMA, self.data.Close, timeperiod=self.sma_period)
        self.ema = self.I(talib.EMA, self.data.Close, timeperiod=self.ema_period)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close,
                          timeperiod=self.atr_period)
        # Swing low using talib.MIN on Low
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=self.swing_lookback)
        print("🚀 Indicators ready: SMA, EMA, ATR, SwingLow ✨")

    def next(self):
        price = self.data.Close[-1]

        # Skip if indicators not ready
        if np.isnan(self.sma[-1]) or np.isnan(self.ema[-1]) or np.isnan(self.atr[-1]):
            return
        if len(self.data) < 2:
            return

        prev_sma = self.sma[-2]
        prev_ema = self.ema[-2]
        curr_sma = self.sma[-1]
        curr_ema = self.ema[-1]

        # 🌙 Entry Logic: bullish crossover + price above EMA filter
        if not self.position:
            bullish_cross = (prev_sma <= prev_ema) and (curr_sma > curr_ema)
            price_above_ema = price > curr_ema

            if bullish_cross and price_above_ema:
                sl = self.swing_low[-1]
                if sl >= price:
                    # Invalid stop (above/at entry), fallback to ATR-based stop
                    sl = price - 1.0 * self.atr[-1]
                tp = price + (self.tp_atr_mult * self.atr[-1])

                if sl < price < tp:
                    print(f"🌙✨ LONG SIGNAL @ {price:.2f} | SL: {sl:.2f} | TP: {tp:.2f} 🚀")
                    # 🌙 Use fractional equity sizing (90% of available equity)
                    self.buy(size=0.9, sl=sl, tp=tp)

        # 🌙 Exit fallback: if in position and no SL/TP triggered, backtesting.py handles it
        # Optional time-based exit could be added here if desired


print("🚀 Moon Dev launching backtest... ✨")
bt = Backtest(data, KineticCrossover, cash=1000000, commission=0.001, exclusive_orders=True)
stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙✨ Backtest complete! To the moon! 🚀🌙")