import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Load data
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper column mapping
data.columns = [col.capitalize() for col in data.columns]

# Ensure required columns
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

print("🌙✨ DeltaSkewReversal Backtest Starting... 🚀")
print(f"📊 Data shape: {data.shape}")
print(f"📈 Data columns: {list(data.columns)}")


class DeltaSkewReversal(Strategy):
    """
    🌙 DeltaSkewReversal Strategy 🌙
    Mean-reversion on overbought conditions.
    Since options data is not available, we simulate the bearish
    net-delta position by shorting the underlying when RSI > 70
    and price is above the upper Bollinger Band.
    """

    rsi_period = 14
    rsi_overbought = 70
    rsi_exit = 50
    bb_period = 20
    bb_dev = 2
    sma_period = 20
    risk_pct = 0.02
    atr_period = 14
    stop_atr_mult = 2.0
    tp_pct = 0.03  # 3% profit target

    def init(self):
        print("🌙 Initializing indicators... ✨")
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=self.rsi_period)
        self.bb_upper, self.bb_mid, self.bb_lower = self.I(
            talib.BBANDS, self.data.Close,
            timeperiod=self.bb_period, nbdevup=self.bb_dev, nbdevdn=self.bb_dev
        )
        self.sma = self.I(talib.SMA, self.data.Close, timeperiod=self.sma_period)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period)

    def next(self):
        price = self.data.Close[-1]
        rsi = self.rsi[-1]
        upper = self.bb_upper[-1]
        sma = self.sma[-1]
        atr = self.atr[-1]

        # Skip if indicators not ready
        if np.isnan(rsi) or np.isnan(upper) or np.isnan(atr):
            return

        # 🚀 ENTRY: Overbought condition -> short (bearish net delta)
        if not self.position:
            if rsi > self.rsi_overbought and price > upper and price > sma:
                # Risk-based sizing
                risk_amount = self.equity * self.risk_pct
                stop_distance = atr * self.stop_atr_mult
                if stop_distance <= 0:
                    return
                position_size = int(round(risk_amount / stop_distance))
                if position_size <= 0:
                    position_size = 1

                sl = price + stop_distance
                tp = price * (1 - self.tp_pct)

                print(f"🌙✨ SHORT SIGNAL | RSI={rsi:.2f} > {self.rsi_overbought} | "
                      f"Price={price:.2f} > BB_Upper={upper:.2f} | "
                      f"Size={position_size} | SL={sl:.2f} | TP={tp:.2f} 🚀")

                self.sell(size=position_size, sl=sl, tp=tp)

        # 🌙 EXIT: Mean reversion resolution
        else:
            if rsi < self.rsi_exit or price < sma:
                print(f"🌙 EXIT SIGNAL | RSI={rsi:.2f} < {self.rsi_exit} or Price < SMA | "
                      f"Closing position at {price:.2f} ✨")
                self.position.close()


# Run backtest
bt = Backtest(data, DeltaSkewReversal, cash=1_000_000, commission=0.001)

print("🚀 Running backtest... 🌙")
stats = bt.run()
print(stats)
print(stats._strategy)