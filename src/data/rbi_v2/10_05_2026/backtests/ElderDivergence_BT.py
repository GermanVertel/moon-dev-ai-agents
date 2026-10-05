import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Load data
data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper mapping
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

print("🌙✨ Moon Dev Data Loaded! Shape:", data.shape)
print(data.head())


class ElderDivergence(Strategy):
    ema_period = 13
    bp_sma_period = 5
    vol_sma_period = 20
    adx_period = 14
    atr_period = 14
    swing_lookback = 20
    adx_threshold = 25
    risk_pct = 0.02
    atr_mult = 1.5

    def init(self):
        print("🌙🚀 Initializing ElderDivergence Strategy Indicators...")
        self.ema13 = self.I(talib.EMA, self.data.Close, timeperiod=self.ema_period)
        self.bull_power = self.I(lambda h, e: h - e, self.data.High, self.ema13)
        self.bp_sma = self.I(talib.SMA, self.bull_power, timeperiod=self.bp_sma_period)
        self.vol_sma = self.I(talib.SMA, self.data.Volume, timeperiod=self.vol_sma_period)
        self.adx = self.I(talib.ADX, self.data.High, self.data.Low, self.data.Close, timeperiod=self.adx_period)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period)
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=self.swing_lookback)
        print("🌙✅ Indicators ready!")

    def next(self):
        if len(self.data) < max(self.swing_lookback, self.vol_sma_period, self.adx_period, self.atr_period) + 5:
            return

        price = self.data.Close[-1]
        high = self.data.High[-1]
        bp = self.bull_power[-1]
        bp_sma = self.bp_sma[-1]
        vol = self.data.Volume[-1]
        vol_sma = self.vol_sma[-1]
        adx = self.adx[-1]
        atr = self.atr[-1]
        swing_high = self.swing_high[-1]

        # Position management
        if self.position:
            if self.position.is_short and bp > bp_sma:
                print(f"🌙💥 EXIT SHORT: Bull Power {bp:.2f} crossed above SMA {bp_sma:.2f} at {price:.2f}")
                self.position.close()
            return

        # Entry logic
        # Higher high: current high is at/near the swing high of lookback
        higher_high = high >= swing_high * 0.999

        # Bull power lower high: current bp < bp 5 bars ago
        bp_prev = self.bull_power[-5]
        bp_lower_high = bp < bp_prev

        # Volume declining
        vol_declining = vol < vol_sma or (self.data.Volume[-1] < self.data.Volume[-3] < self.data.Volume[-5])

        # ADX trending
        adx_trending = adx > self.adx_threshold

        # Bull power still positive
        bp_positive = bp > 0

        if higher_high and bp_lower_high and vol_declining and adx_trending and bp_positive:
            # Risk sizing
            stop_price = high + self.atr_mult * atr
            risk_per_unit = stop_price - price
            if risk_per_unit <= 0:
                return

            equity = self.equity
            risk_amount = equity * self.risk_pct
            position_size = int(round(risk_amount / risk_per_unit))
            if position_size < 1:
                position_size = 1

            print(f"🌙🚀 SHORT ENTRY: Price={price:.2f} HH={higher_high} BP_LH={bp_lower_high} "
                  f"VolDec={vol_declining} ADX={adx:.2f} BP={bp:.2f} Size={position_size}")

            self.sell(size=position_size, sl=stop_price)


bt = Backtest(data, ElderDivergence, cash=1_000_000, commission=0.001, exclusive_orders=True)
stats = bt.run()
print(stats)
print(stats._strategy)