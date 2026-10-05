import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Load data
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')

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

print("🌙 Moon Dev Data Loaded! Shape: {}".format(data.shape))
print("✨ Columns: {}".format(list(data.columns)))
print("🚀 First few rows:\n{}".format(data.head()))


class CompressionVolt(Strategy):
    bb_period = 20
    bb_std = 2.0
    rsi_period = 14
    bbw_lookback = 20
    atr_period = 14
    risk_pct = 0.02
    atr_mult = 2.0

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low

        # Bollinger Bands
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS, close, timeperiod=self.bb_period,
            nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0
        )

        # Bollinger Band Width
        def compute_bbw(upper, middle, lower):
            return (upper - lower) / middle

        self.bbw = self.I(compute_bbw, self.bb_upper, self.bb_middle, self.bb_lower)

        # BBW 20-period low
        self.bbw_low = self.I(talib.MIN, self.bbw, timeperiod=self.bbw_lookback)

        # RSI
        self.rsi = self.I(talib.RSI, close, timeperiod=self.rsi_period)

        # ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        print("🌙✨ CompressionVolt indicators initialized! 🚀")

    def next(self):
        price = self.data.Close[-1]

        # Skip if indicators not ready
        if np.isnan(self.bbw_low[-1]) or np.isnan(self.rsi[-1]) or np.isnan(self.atr[-1]):
            return
        if np.isnan(self.bb_upper[-1]) or np.isnan(self.bb_lower[-1]):
            return

        bbw_is_low = self.bbw[-1] <= self.bbw_low[-1]
        rsi_val = self.rsi[-1]

        # ===== EXIT LOGIC =====
        if self.position:
            if self.position.is_long:
                # Exit long when close below lower band
                if price < self.bb_lower[-1]:
                    print("🌙💥 LONG EXIT @ {} | Close < Lower BB ({:.2f})".format(
                        price, self.bb_lower[-1]))
                    self.position.close()
                    return
            elif self.position.is_short:
                # Exit short when close above upper band
                if price > self.bb_upper[-1]:
                    print("🌙💥 SHORT EXIT @ {} | Close > Upper BB ({:.2f})".format(
                        price, self.bb_upper[-1]))
                    self.position.close()
                    return

        # ===== ENTRY LOGIC =====
        if not self.position:
            # Long entry: squeeze + RSI > 70
            if bbw_is_low and rsi_val > 70:
                sl = price - self.atr_mult * self.atr[-1]
                risk = price - sl
                if risk <= 0:
                    return
                # Use fractional position sizing (percentage of equity)
                size = 0.95
                print("🌙🚀 LONG ENTRY @ {} | RSI={:.2f} | BBW squeeze | SL={:.2f} | Size={}".format(
                    price, rsi_val, sl, size))
                self.buy(size=size, sl=sl)

            # Short entry: squeeze + RSI < 30
            elif bbw_is_low and rsi_val < 30:
                sl = price + self.atr_mult * self.atr[-1]
                risk = sl - price
                if risk <= 0:
                    return
                # Use fractional position sizing (percentage of equity)
                size = 0.95
                print("🌙🚀 SHORT ENTRY @ {} | RSI={:.2f} | BBW squeeze | SL={:.2f} | Size={}".format(
                    price, rsi_val, sl, size))
                self.sell(size=size, sl=sl)


# Run backtest
bt = Backtest(data, CompressionVolt, cash=1000000, commission=0.002)

print("🌙✨ Starting CompressionVolt Backtest... 🚀")
stats = bt.run()
print(stats)
print(stats._strategy)