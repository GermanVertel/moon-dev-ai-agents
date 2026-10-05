import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Load and clean data
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
print("🌙✨ Moon Dev data loaded:", data.shape)


class FractalVolume(Strategy):
    vma_period = 20
    volume_mult = 1.5
    atr_period = 14
    atr_mult = 2.0
    rr_ratio = 2.0
    min_fractal_spacing = 10
    risk_pct = 0.02

    def init(self):
        self.vma = self.I(talib.SMA, self.data.Volume, timeperiod=self.vma_period)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period)

        high = self.data.High.values
        low = self.data.Low.values
        n = len(high)

        bearish_fractal = np.full(n, np.nan)
        bullish_fractal = np.full(n, np.nan)

        for i in range(2, n - 2):
            if high[i] > high[i-1] and high[i] > high[i-2] and high[i] > high[i+1] and high[i] > high[i+2]:
                bearish_fractal[i + 2] = high[i]
            if low[i] < low[i-1] and low[i] < low[i-2] and low[i] < low[i+1] and low[i] < low[i+2]:
                bullish_fractal[i + 2] = low[i]

        self.bearish_fractal = self.I(lambda: pd.Series(bearish_fractal).ffill().values, name='BearFrac')
        self.bullish_fractal = self.I(lambda: pd.Series(bullish_fractal).ffill().values, name='BullFrac')

        bf_idx = np.full(n, -1000, dtype=int)
        bl_idx = np.full(n, -1000, dtype=int)
        last_b = -1000
        last_l = -1000
        for i in range(n):
            if not np.isnan(bearish_fractal[i]):
                last_b = i
            if not np.isnan(bullish_fractal[i]):
                last_l = i
            bf_idx[i] = last_b
            bl_idx[i] = last_l
        self.bf_idx = self.I(lambda: bf_idx, name='bf_idx')
        self.bl_idx = self.I(lambda: bl_idx, name='bl_idx')

    def next(self):
        if len(self.data) < self.vma_period + 5:
            return

        price = self.data.Close[-1]
        vol = self.data.Volume[-1]
        vma = self.vma[-1]
        atr = self.atr[-1]

        if np.isnan(vma) or np.isnan(atr) or vma == 0:
            return

        vol_spike = vol >= self.volume_mult * vma

        if self.position:
            return

        bf_i = self.bf_idx[-1]
        bl_i = self.bl_idx[-1]
        current_i = len(self.data) - 1

        # Long entry
        if bf_i >= 0 and not np.isnan(self.bearish_fractal[-1]) and vol_spike:
            if price > self.bearish_fractal[-1] and (current_i - bf_i) >= self.min_fractal_spacing:
                sl = self.bullish_fractal[-1]
                if not np.isnan(sl) and sl < price:
                    risk = price - sl
                    if risk > 0:
                        tp = price + self.rr_ratio * risk
                        equity = self.equity
                        risk_amount = equity * self.risk_pct
                        size = int(round(risk_amount / risk))
                        if size > 0:
                            print(f"🌙🚀 MOON DEV LONG BREAKOUT! Price={price:.2f} FractalHigh={self.bearish_fractal[-1]:.2f} Vol={vol:.2f} VMA={vma:.2f} SL={sl:.2f} TP={tp:.2f} Size={size}")
                            self.buy(size=size, sl=sl, tp=tp)

        # Short entry
        if bl_i >= 0 and not np.isnan(self.bullish_fractal[-1]) and vol_spike:
            if price < self.bullish_fractal[-1] and (current_i - bl_i) >= self.min_fractal_spacing:
                sl = self.bearish_fractal[-1]
                if not np.isnan(sl) and sl > price:
                    risk = sl - price
                    if risk > 0:
                        tp = price - self.rr_ratio * risk
                        equity = self.equity
                        risk_amount = equity * self.risk_pct
                        size = int(round(risk_amount / risk))
                        if size > 0:
                            print(f"🌙💫 MOON DEV SHORT BREAKOUT! Price={price:.2f} FractalLow={self.bullish_fractal[-1]:.2f} Vol={vol:.2f} VMA={vma:.2f} SL={sl:.2f} TP={tp:.2f} Size={size}")
                            self.sell(size=size, sl=sl, tp=tp)


bt = Backtest(data, FractalVolume, cash=1000000, commission=0.002, exclusive_orders=True)
stats = bt.run()
print(stats)
print(stats._strategy)