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
data = data.rename(columns={
    'datetime': 'Date',
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

data['Date'] = pd.to_datetime(data['Date'])
data = data.set_index('Date')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

print("🌙✨ Moon Dev Data Loaded! Shape:", data.shape)
print("🚀 First few rows:\n", data.head())


class BollingerBandMomentum(Strategy):
    bb_period = 20
    bb_std = 2.0
    ema_period = 200
    rsi_period = 14
    atr_period = 14
    risk_pct = 0.02
    atr_mult = 1.5
    rr_ratio = 2.0

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low

        # Bollinger Bands - using talib wrapped in self.I
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS, close, timeperiod=self.bb_period,
            nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0
        )

        # Trend filter - talib EMA wrapped in self.I
        self.ema200 = self.I(talib.EMA, close, timeperiod=self.ema_period)

        # Momentum - talib RSI wrapped in self.I
        self.rsi = self.I(talib.RSI, close, timeperiod=self.rsi_period)

        # Volatility - talib ATR wrapped in self.I
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        print("🌙✨ Bollinger Band Momentum indicators initialized! 🚀")

    def next(self):
        price = self.data.Close[-1]
        upper = self.bb_upper[-1]
        middle = self.bb_middle[-1]
        lower = self.bb_lower[-1]
        ema = self.ema200[-1]
        rsi = self.rsi[-1]
        atr = self.atr[-1]

        if np.isnan(upper) or np.isnan(ema) or np.isnan(rsi) or np.isnan(atr):
            return

        # ===== EXIT LOGIC =====
        if self.position:
            if self.position.is_long:
                # Exit long: close below middle band OR RSI flips bearish
                if price < middle or rsi < 50:
                    print(f"🌙 EXIT LONG @ {price:.2f} | RSI={rsi:.1f} | Mid={middle:.2f}")
                    self.position.close()
            elif self.position.is_short:
                # Exit short: close above middle band OR RSI flips bullish
                if price > middle or rsi > 50:
                    print(f"🌙 EXIT SHORT @ {price:.2f} | RSI={rsi:.1f} | Mid={middle:.2f}")
                    self.position.close()
            return

        # ===== ENTRY LOGIC =====
        # LONG: touch upper band + bullish momentum + above 200 EMA
        if price >= upper and rsi > 50 and price > ema:
            sl = price - self.atr_mult * atr
            tp = price + self.rr_ratio * (price - sl)
            risk_per_unit = price - sl
            if risk_per_unit > 0:
                size = int(round((self.equity * self.risk_pct) / risk_per_unit))
                if size > 0:
                    print(f"🚀 LONG ENTRY @ {price:.2f} | RSI={rsi:.1f} | Upper={upper:.2f} | SL={sl:.2f} | TP={tp:.2f} | Size={size}")
                    self.buy(size=size, sl=sl, tp=tp)

        # SHORT: price crosses above middle band from below + bearish momentum + below 200 EMA
        # Crossover detection without backtesting.lib
        elif (self.data.Close[-2] < self.bb_middle[-2] and price > middle
              and rsi < 50 and price < ema):
            sl = price + self.atr_mult * atr
            tp = price - self.rr_ratio * (sl - price)
            risk_per_unit = sl - price
            if risk_per_unit > 0:
                size = int(round((self.equity * self.risk_pct) / risk_per_unit))
                if size > 0:
                    print(f"🔻 SHORT ENTRY @ {price:.2f} | RSI={rsi:.1f} | Mid={middle:.2f} | SL={sl:.2f} | TP={tp:.2f} | Size={size}")
                    self.sell(size=size, sl=sl, tp=tp)


bt = Backtest(data, BollingerBandMomentum, cash=1_000_000, commission=0.001)

print("🌙✨ Running Moon Dev Backtest... 🚀")
stats = bt.run()
print(stats)
print(stats._strategy)