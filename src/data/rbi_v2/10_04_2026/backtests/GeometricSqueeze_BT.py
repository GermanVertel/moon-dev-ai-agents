import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

print("🌙 ✨ Moon Dev's GeometricSqueeze Backtest Loading... 🚀")

data_path = '/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv'
data = pd.read_csv(data_path)

data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

if 'datetime' in data.columns:
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')

data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

print(f"🌙 Data loaded: {len(data)} bars ✨")


class GeometricSqueeze(Strategy):
    swing_lookback = 20
    sma_period = 5
    atr_period = 14
    risk_pct = 0.01
    rr_target = 1.5
    gann_scale = 1.0

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low

        self.sma = self.I(talib.SMA, close, timeperiod=self.sma_period)
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        self.swing_high = self.I(talib.MAX, high, timeperiod=self.swing_lookback)
        self.swing_low = self.I(talib.MIN, low, timeperiod=self.swing_lookback)

        self.sma_prev = np.roll(self.sma, 1)
        self.sma_prev[0] = self.sma[0]

        print("🌙 ✨ GeometricSqueeze indicators initialized 🚀")

    def next(self):
        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        open_ = self.data.Open[-1]
        sma = self.sma[-1]
        sma_prev = self.sma_prev[-1] if len(self.sma) > 1 else sma
        atr = self.atr[-1]

        if np.isnan(sma) or np.isnan(atr) or atr <= 0:
            return

        swing_low = self.swing_low[-1]
        swing_high = self.swing_high[-1]

        bars_since_low = 1
        for i in range(2, min(self.swing_lookback + 1, len(self.data))):
            if self.data.Low[-i] == swing_low:
                bars_since_low = i
                break

        bars_since_high = 1
        for i in range(2, min(self.swing_lookback + 1, len(self.data))):
            if self.data.High[-i] == swing_high:
                bars_since_high = i
                break

        gann_up = swing_low + bars_since_low * atr * 0.5 * self.gann_scale
        gann_down = swing_high - bars_since_high * atr * 0.5 * self.gann_scale

        sma_rising = sma > sma_prev
        sma_falling = sma < sma_prev

        bull_reversal = (price > open_) and (low < min(open_, price)) and (price > (high + low) / 2)
        bear_reversal = (price < open_) and (high > max(open_, price)) and (price < (high + low) / 2)

        if not self.position:
            long_setup = (
                price > gann_up and
                price > sma and
                sma_rising and
                bull_reversal
            )

            short_setup = (
                price < gann_down and
                price < sma and
                sma_falling and
                bear_reversal
            )

            equity = self.equity
            risk_amount = equity * self.risk_pct

            if long_setup:
                stop = min(swing_low, gann_up - atr * 0.25)
                risk_per_unit = price - stop
                if risk_per_unit > 0:
                    size = int(round(risk_amount / risk_per_unit))
                    if size > 0:
                        tp = price + risk_per_unit * self.rr_target
                        print(f"🌙 🚀 LONG ENTRY @ {price:.2f} | SL {stop:.2f} | TP {tp:.2f} | size {size} ✨")
                        self.buy(size=size, sl=stop, tp=tp)

            elif short_setup:
                stop = max(swing_high, gann_down + atr * 0.25)
                risk_per_unit = stop - price
                if risk_per_unit > 0:
                    size = int(round(risk_amount / risk_per_unit))
                    if size > 0:
                        tp = price - risk_per_unit * self.rr_target
                        print(f"🌙 🚀 SHORT ENTRY @ {price:.2f} | SL {stop:.2f} | TP {tp:.2f} | size {size} ✨")
                        self.sell(size=size, sl=stop, tp=tp)

        else:
            if self.position.is_long:
                if price < sma and sma_falling:
                    print(f"🌙 ✨ LONG EXIT via SMA trailing stop @ {price:.2f}")
                    self.position.close()
            elif self.position.is_short:
                if price > sma and sma_rising:
                    print(f"🌙 ✨ SHORT EXIT via SMA trailing stop @ {price:.2f}")
                    self.position.close()


bt = Backtest(data, GeometricSqueeze, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)