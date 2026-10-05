import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

print("🌙 Moon Dev Backtest AI initializing... ✨")
print("🚀 Loading BandedOscillator strategy... 🎯")

data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"

data = pd.read_csv(data_path)
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
data = data.set_index(pd.to_datetime(data['datetime']))
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
data = data.dropna()

print(f"🌙 Data loaded: {len(data)} bars ✨")


class BandedOscillator(Strategy):
    bb_period = 20
    bb_std = 2.0
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    risk_pct = 0.02
    proximity_pct = 0.01  # 1% proximity to lower band (widened for trade execution)

    def init(self):
        print("🌙 Initializing indicators... ✨")
        close = self.data.Close

        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS, close,
            timeperiod=self.bb_period,
            nbdevup=self.bb_std,
            nbdevdn=self.bb_std,
            matype=0
        )

        self.macd_line, self.macd_signal_line, self.macd_hist = self.I(
            talib.MACD, close,
            fastperiod=self.macd_fast,
            slowperiod=self.macd_slow,
            signalperiod=self.macd_signal
        )

        print("🚀 Indicators ready! Bollinger + MACD loaded 🎯")

    def next(self):
        price = self.data.Close[-1]
        lower = self.bb_lower[-1]
        middle = self.bb_middle[-1]
        upper = self.bb_upper[-1]
        macd = self.macd_line[-1]

        if np.isnan(lower) or np.isnan(macd):
            return

        # Entry logic
        if not self.position:
            macd_bullish = macd > 0
            near_lower = price <= lower * (1 + self.proximity_pct)

            if macd_bullish and near_lower:
                sl = lower * 0.995
                risk_per_unit = price - sl
                if risk_per_unit <= 0:
                    return
                risk_amount = self.equity * self.risk_pct
                size = int(round(risk_amount / risk_per_unit))
                if size > 0:
                    print(f"🌙✨ BUY signal! Price={price:.2f} Lower={lower:.2f} MACD={macd:.2f} Size={size} 🚀")
                    self.buy(size=size, sl=sl)

        # Exit logic
        else:
            # Exit if price closes below lower band (invalidation)
            if price < lower:
                print(f"🌙❌ Exit: price {price:.2f} below lower band {lower:.2f} 🛑")
                self.position.close()
            # Optional profit exit at middle band
            elif price >= middle:
                print(f"🌙💰 Profit exit at middle band! Price={price:.2f} Middle={middle:.2f} 🎯")
                self.position.close()
            # Exit if MACD crosses below zero
            elif macd < 0:
                print(f"🌙⚠️ MACD below zero, exiting. MACD={macd:.2f}")
                self.position.close()


bt = Backtest(
    data,
    BandedOscillator,
    cash=1_000_000,
    commission=0.001,
    exclusive_orders=True
)

print("🚀 Running backtest... 🌙")
stats = bt.run()
print(stats)
print(stats._strategy)
print("✨ Moon Dev backtest complete! 🌙")