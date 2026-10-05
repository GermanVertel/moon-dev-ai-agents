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
print("🌙✨ Moon Dev data loaded:", data.shape)
print(data.head())


class VolatilityFilteredCrossover(Strategy):
    ema_fast = 50
    ema_slow = 200
    atr_period = 14
    atr_ma_period = 50
    donchian_period = 20
    risk_pct = 0.01
    sl_mult = 2.0
    tp_mult = 1.5

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low

        self.ema50 = self.I(talib.EMA, close, timeperiod=self.ema_fast)
        self.ema200 = self.I(talib.EMA, close, timeperiod=self.ema_slow)
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)
        self.atr_ma = self.I(talib.SMA, self.atr, timeperiod=self.atr_ma_period)
        self.dc_high = self.I(talib.MAX, high, timeperiod=self.donchian_period)
        self.dc_low = self.I(talib.MIN, low, timeperiod=self.donchian_period)

        print("🚀 Moon Dev indicators initialized 🌙")

    def next(self):
        if len(self.data) < max(self.ema_slow, self.atr_ma_period, self.donchian_period) + 2:
            return

        price = self.data.Close[-1]
        prev_ema50 = self.ema50[-2]
        prev_ema200 = self.ema200[-2]
        curr_ema50 = self.ema50[-1]
        curr_ema200 = self.ema200[-1]
        atr = self.atr[-1]
        atr_ma = self.atr_ma[-1]

        # Skip if indicators invalid
        if np.isnan(atr) or np.isnan(atr_ma) or atr_ma == 0:
            return
        if np.isnan(prev_ema50) or np.isnan(prev_ema200) or np.isnan(curr_ema50) or np.isnan(curr_ema200):
            return

        low_vol = atr < atr_ma

        bullish_cross = prev_ema50 <= prev_ema200 and curr_ema50 > curr_ema200
        bearish_cross = prev_ema50 >= prev_ema200 and curr_ema50 < curr_ema200

        # Breakout confirmation using prior bar's donchian
        breakout_up = price > self.dc_high[-2]
        breakout_down = price < self.dc_low[-2]

        # Long entry
        if not self.position and bullish_cross and low_vol and breakout_up:
            sl = price - self.sl_mult * atr
            tp = price + self.tp_mult * atr
            risk_per_unit = price - sl
            if risk_per_unit <= 0:
                return
            risk_amount = self.equity * self.risk_pct
            size = int(round(risk_amount / risk_per_unit))
            if size < 1:
                size = 1
            print(f"🌙✨ LONG SIGNAL | price={price:.2f} atr={atr:.2f} sl={sl:.2f} tp={tp:.2f} size={size} 🚀")
            self.buy(size=size, sl=sl, tp=tp)

        # Short entry
        elif not self.position and bearish_cross and low_vol and breakout_down:
            sl = price + self.sl_mult * atr
            tp = price - self.tp_mult * atr
            risk_per_unit = sl - price
            if risk_per_unit <= 0:
                return
            risk_amount = self.equity * self.risk_pct
            size = int(round(risk_amount / risk_per_unit))
            if size < 1:
                size = 1
            print(f"🌙✨ SHORT SIGNAL | price={price:.2f} atr={atr:.2f} sl={sl:.2f} tp={tp:.2f} size={size} 🚀")
            self.sell(size=size, sl=sl, tp=tp)


bt = Backtest(data, VolatilityFilteredCrossover, cash=1_000_000, commission=0.0002, trade_on_close=False, exclusive_orders=True)
stats = bt.run()
print(stats)
print(stats._strategy)