import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

print("🌙 Moon Dev SqueezeImpulse Backtest Starting... ✨")

data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
print(f"🚀 Data loaded: {len(data)} candles")


class SqueezeImpulse(Strategy):
    bb_period = 20
    bb_std = 2.0
    adx_period = 14
    atr_period = 14
    bbw_squeeze_threshold = 2.0
    bbw_expand_threshold = 6.0
    adx_entry = 30
    adx_exit = 25
    risk_pct = 0.02
    size = 0.95

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low

        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS, close, timeperiod=self.bb_period,
            nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0
        )
        self.bbw = self.I(
            lambda u, m, l: (u - l) / m * 100.0,
            self.bb_upper, self.bb_middle, self.bb_lower
        )
        self.adx = self.I(talib.ADX, high, low, close, timeperiod=self.adx_period)
        self.di_plus = self.I(talib.PLUS_DI, high, low, close, timeperiod=self.adx_period)
        self.di_minus = self.I(talib.MINUS_DI, high, low, close, timeperiod=self.adx_period)
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        self.squeeze_high = np.nan
        self.squeeze_low = np.nan
        self.in_squeeze = False

    def next(self):
        price = self.data.Close[-1]
        bbw = self.bbw[-1]
        adx = self.adx[-1]
        prev_adx = self.adx[-2] if len(self.adx) > 1 else 0.0
        di_p = self.di_plus[-1]
        di_m = self.di_minus[-1]
        atr = self.atr[-1]

        if np.isnan(bbw) or np.isnan(adx) or np.isnan(atr):
            return

        # Track squeeze range
        if bbw < self.bbw_squeeze_threshold:
            if not self.in_squeeze:
                self.squeeze_high = self.data.High[-1]
                self.squeeze_low = self.data.Low[-1]
                self.in_squeeze = True
                print(f"🌙 Squeeze detected! BBW={bbw:.2f}% price={price:.2f}")
            else:
                self.squeeze_high = max(self.squeeze_high, self.data.High[-1])
                self.squeeze_low = min(self.squeeze_low, self.data.Low[-1])

        adx_cross_up = prev_adx < self.adx_entry and adx >= self.adx_entry

        # Manage open positions
        if self.position:
            self._manage_position(price, bbw, adx, atr)
            return

        # Entry logic
        if not self.in_squeeze:
            return
        if np.isnan(self.squeeze_high) or np.isnan(self.squeeze_low):
            return

        rng = self.squeeze_high - self.squeeze_low
        if rng <= 0:
            return

        # Long
        if adx_cross_up and di_p > di_m and price > self.bb_upper[-1]:
            stop = max(self.bb_lower[-1], price - 1.5 * atr)
            target = price + rng * 1.618
            risk = price - stop
            reward = target - price
            if risk > 0 and reward >= 2 * risk:
                self.buy(size=self.size, sl=stop, tp=target)
                print(f"🚀 LONG entry! price={price:.2f} stop={stop:.2f} target={target:.2f} R:R={reward/risk:.2f}")
                self.in_squeeze = False
                return

        # Short
        if adx_cross_up and di_m > di_p and price < self.bb_lower[-1]:
            stop = min(self.bb_upper[-1], price + 1.5 * atr)
            target = price - rng * 1.618
            risk = stop - price
            reward = price - target
            if risk > 0 and reward >= 2 * risk:
                self.sell(size=self.size, sl=stop, tp=target)
                print(f"🔻 SHORT entry! price={price:.2f} stop={stop:.2f} target={target:.2f} R:R={reward/risk:.2f}")
                self.in_squeeze = False
                return

    def _manage_position(self, price, bbw, adx, atr):
        # Time stop conditions
        if adx < self.adx_exit or bbw > self.bbw_expand_threshold:
            print(f"⏰ Time stop exit: ADX={adx:.2f} BBW={bbw:.2f}%")
            self.position.close()
            self.in_squeeze = False


bt = Backtest(data, SqueezeImpulse, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)