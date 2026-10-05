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
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

print("🌙✨ Moon Dev Backtest Initializing... VolatilityContraction Strategy Loading 🚀")
print(f"📊 Data shape: {data.shape}")
print(f"📈 Date range: {data.index[0]} → {data.index[-1]}")


class VolatilityContraction(Strategy):
    # MACD params
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    # BB params
    bb_period = 20
    bb_std = 2.0
    # ATR
    atr_period = 14
    atr_mult = 1.5
    # Risk
    risk_pct = 0.02
    size = 1_000_000

    def init(self):
        print("🌙 Initializing indicators...")
        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)

        # MACD
        macd, macd_sig, macd_hist = talib.MACD(
            close.values,
            fastperiod=self.macd_fast,
            slowperiod=self.macd_slow,
            signalperiod=self.macd_signal
        )
        self.macd = self.I(lambda: macd, name='MACD')
        self.macd_sig = self.I(lambda: macd_sig, name='MACD_Signal')
        self.macd_hist = self.I(lambda: macd_hist, name='MACD_Hist')

        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(
            close.values,
            timeperiod=self.bb_period,
            nbdevup=self.bb_std,
            nbdevdn=self.bb_std,
            matype=0
        )
        self.bb_upper = self.I(lambda: upper, name='BB_Upper')
        self.bb_middle = self.I(lambda: middle, name='BB_Middle')
        self.bb_lower = self.I(lambda: lower, name='BB_Lower')
        bb_width = upper - lower
        self.bb_width = self.I(lambda: bb_width, name='BB_Width')

        # ATR
        atr = talib.ATR(high.values, low.values, close.values, timeperiod=self.atr_period)
        self.atr = self.I(lambda: atr, name='ATR')

        print("✨ Indicators ready!")

    def next(self):
        price = self.data.Close[-1]

        # Need enough history
        if len(self.data) < self.bb_period + 5:
            return

        # --- Momentum condition ---
        macd_bullish = (
            (self.macd[-1] > self.macd_sig[-1] and self.macd[-2] <= self.macd_sig[-2])
            or (self.macd_hist[-1] > 0 and self.macd_hist[-2] <= 0)
        )

        # --- BB downtrend: middle band sloping down ---
        bb_downtrend = self.bb_middle[-1] < self.bb_middle[-2]

        # --- BB width expanding for 2+ consecutive bars ---
        width_expanding = (
            self.bb_width[-1] > self.bb_width[-2]
            and self.bb_width[-2] > self.bb_width[-3]
        )

        # --- Exit: close below lower band ---
        if self.position:
            if price < self.bb_lower[-1]:
                print(f"🚪🌙 EXIT — close {price:.2f} < lower band {self.bb_lower[-1]:.2f}")
                self.position.close()
            return

        # --- Entry ---
        if macd_bullish and bb_downtrend and width_expanding:
            atr_val = self.atr[-1]
            if np.isnan(atr_val) or atr_val <= 0:
                return

            # 2% risk per trade
            risk_amount = self.equity * self.risk_pct
            stop_distance = self.atr_mult * atr_val
            position_size = int(round(risk_amount / stop_distance)) if stop_distance > 0 else 0

            # Fallback to fixed size if calc fails
            if position_size <= 0:
                position_size = int(round(self.size / price))

            if position_size <= 0:
                return

            sl = price - stop_distance
            print(f"🚀🌙 LONG ENTRY — price: {price:.2f} | size: {position_size} | SL: {sl:.2f} | ATR: {atr_val:.2f}")
            self.buy(size=position_size, sl=sl)


bt = Backtest(data, VolatilityContraction, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)