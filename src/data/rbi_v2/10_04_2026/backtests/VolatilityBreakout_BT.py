import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Load and clean data
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
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
print("🌙✨ Data loaded and cleaned! Shape:", data.shape)
print(data.head())


class VolatilityBreakout(Strategy):
    bb_period = 20
    bb_std = 2.0
    atr_period = 14
    atr_ma_period = 20
    risk_pct = 0.02
    atr_stop_mult = 2.0

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low

        # Bollinger Bands
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS, close, timeperiod=self.bb_period,
            nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0
        )
        # ATR and its SMA
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)
        self.atr_ma = self.I(talib.SMA, self.atr, timeperiod=self.atr_ma_period)

        print("🚀🌙 VolatilityBreakout indicators initialized!")

    def next(self):
        price = self.data.Close[-1]

        # Skip if indicators not ready
        if np.isnan(self.bb_upper[-1]) or np.isnan(self.atr[-1]) or np.isnan(self.atr_ma[-1]):
            return

        upper = self.bb_upper[-1]
        lower = self.bb_lower[-1]
        middle = self.bb_middle[-1]
        atr = self.atr[-1]
        atr_ma = self.atr_ma[-1]

        # Manage open position exits
        if self.position:
            if self.position.is_long:
                if price < middle:
                    print(f"🌙 EXIT LONG | price {price:.2f} < middle {middle:.2f} | 💰")
                    self.position.close()
                    return
            elif self.position.is_short:
                if price > middle:
                    print(f"🌙 EXIT SHORT | price {price:.2f} > middle {middle:.2f} | 💰")
                    self.position.close()
                    return

        # Entry logic
        if not self.position:
            vol_filter = atr < atr_ma
            long_signal = price > upper and vol_filter
            short_signal = price < lower and vol_filter

            if long_signal:
                stop = price - self.atr_stop_mult * atr
                risk_per_unit = price - stop
                if risk_per_unit > 0:
                    risk_amount = self.equity * self.risk_pct
                    size = int(round(risk_amount / risk_per_unit))
                    if size > 0:
                        print(f"🚀 LONG BREAKOUT | price {price:.2f} > upper {upper:.2f} | ATR {atr:.2f} < ATR_MA {atr_ma:.2f} | size {size}")
                        self.buy(size=size, sl=stop)

            elif short_signal:
                stop = price + self.atr_stop_mult * atr
                risk_per_unit = stop - price
                if risk_per_unit > 0:
                    risk_amount = self.equity * self.risk_pct
                    size = int(round(risk_amount / risk_per_unit))
                    if size > 0:
                        print(f"🔻 SHORT BREAKOUT | price {price:.2f} < lower {lower:.2f} | ATR {atr:.2f} < ATR_MA {atr_ma:.2f} | size {size}")
                        self.sell(size=size, sl=stop)


bt = Backtest(data, VolatilityBreakout, cash=1_000_000, commission=0.001, exclusive_orders=True)
stats = bt.run()
print(stats)
print(stats._strategy)