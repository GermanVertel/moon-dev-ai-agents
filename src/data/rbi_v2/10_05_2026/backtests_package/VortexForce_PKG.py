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

print("🌙✨ Moon Dev VortexForce Backtest Initializing... 🚀")
print(f"📊 Data loaded: {len(data)} bars from {data.index[0]} to {data.index[-1]}")


class VortexForce(Strategy):
    vi_period = 14
    evf_ema_period = 13
    kst_ema_period = 3
    atr_period = 14
    swing_period = 10
    ema_period = 20
    risk_pct = 0.02
    atr_mult = 1.5
    time_stop = 10

    def init(self):
        high = self.data.High
        low = self.data.Low
        close = self.data.Close
        volume = self.data.Volume

        # Vortex Indicator
        vm_plus = high - np.roll(low, 1)
        vm_minus = np.roll(high, 1) - low
        vm_plus = np.where(vm_plus < 0, 0, vm_plus)
        vm_minus = np.where(vm_minus < 0, 0, vm_minus)
        tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
        tr_sum = pd.Series(tr).rolling(self.vi_period).sum().values
        vm_plus_sum = pd.Series(vm_plus).rolling(self.vi_period).sum().values
        vm_minus_sum = pd.Series(vm_minus).rolling(self.vi_period).sum().values

        self.vi_plus = self.I(lambda: vm_plus_sum / tr_sum, name='VI+')
        self.vi_minus = self.I(lambda: vm_minus_sum / tr_sum, name='VI-')

        # Elder's Volume Force
        evf = (close - np.roll(close, 1)) * volume
        self.evf = self.I(talib.EMA, pd.Series(evf), timeperiod=self.evf_ema_period, name='EVF')

        # KST Momentum Oscillator (simplified ROC-based)
        roc1 = talib.ROC(close, timeperiod=10)
        roc2 = talib.ROC(close, timeperiod=15)
        roc3 = talib.ROC(close, timeperiod=20)
        roc4 = talib.ROC(close, timeperiod=30)
        kst = roc1 * 1 + roc2 * 2 + roc3 * 3 + roc4 * 4
        self.kst_ema = self.I(talib.EMA, pd.Series(kst), timeperiod=self.kst_ema_period, name='KST_EMA')

        # ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period, name='ATR')

        # EMA trend filter
        self.ema = self.I(talib.EMA, close, timeperiod=self.ema_period, name='EMA')

        # Swing highs/lows
        self.swing_high = self.I(talib.MAX, high, timeperiod=self.swing_period, name='SwingHigh')
        self.swing_low = self.I(talib.MIN, low, timeperiod=self.swing_period, name='SwingLow')

        self.entry_bar = 0
        self.stop_price = 0
        self.take_profit = 0

    def next(self):
        price = self.data.Close[-1]
        if len(self.data) < 50:
            return

        # Manage existing position
        if self.position:
            bars_held = len(self.data) - 1 - self.entry_bar

            if self.position.is_long:
                # Time stop
                if bars_held >= self.time_stop:
                    print(f"⏰ Moon Dev TIME STOP hit long @ {price:.2f} 🌙")
                    self.position.close()
                    return
                # Stop loss
                if self.data.Low[-1] <= self.stop_price:
                    print(f"🛑 Moon Dev STOP LOSS long @ {self.stop_price:.2f} 🌙")
                    self.position.close()
                    return
                # KST EMA slope flip down
                if len(self.kst_ema) > 2 and self.kst_ema[-1] < self.kst_ema[-2] and self.kst_ema[-2] >= self.kst_ema[-3]:
                    print(f"📉 Moon Dev KST reversal exit long @ {price:.2f} ✨")
                    self.position.close()
                    return
                # Vortex bearish cross (vi_minus crossing above vi_plus)
                if self.vi_minus[-2] < self.vi_plus[-2] and self.vi_minus[-1] > self.vi_plus[-1]:
                    print(f"🔄 Moon Dev Vortex bearish cross exit long @ {price:.2f} 🚀")
                    self.position.close()
                    return

            elif self.position.is_short:
                if bars_held >= self.time_stop:
                    print(f"⏰ Moon Dev TIME STOP hit short @ {price:.2f} 🌙")
                    self.position.close()
                    return
                if self.data.High[-1] >= self.stop_price:
                    print(f"🛑 Moon Dev STOP LOSS short @ {self.stop_price:.2f} 🌙")
                    self.position.close()
                    return
                if len(self.kst_ema) > 2 and self.kst_ema[-1] > self.kst_ema[-2] and self.kst_ema[-2] <= self.kst_ema[-3]:
                    print(f"📈 Moon Dev KST reversal exit short @ {price:.2f} ✨")
                    self.position.close()
                    return
                # Vortex bullish cross (vi_plus crossing above vi_minus)
                if self.vi_plus[-2] < self.vi_minus[-2] and self.vi_plus[-1] > self.vi_minus[-1]:
                    print(f"🔄 Moon Dev Vortex bullish cross exit short @ {price:.2f} 🚀")
                    self.position.close()
                    return

        # Entry logic
        if not self.position:
            evf_rising = len(self.evf) > 2 and self.evf[-1] > self.evf[-2]
            evf_falling = len(self.evf) > 2 and self.evf[-1] < self.evf[-2]

            # Vortex bullish cross (vi_plus crossing above vi_minus)
            vi_bull_cross = self.vi_plus[-2] < self.vi_minus[-2] and self.vi_plus[-1] > self.vi_minus[-1]
            # Vortex bearish cross (vi_minus crossing above vi_plus)
            vi_bear_cross = self.vi_minus[-2] < self.vi_plus[-2] and self.vi_minus[-1] > self.vi_plus[-1]

            # Long entry
            if (vi_bull_cross
                    and self.evf[-1] > 0 and evf_rising
                    and price > self.swing_high[-2]
                    and price > self.ema[-1]):
                stop = price - self.atr[-1] * self.atr_mult
                risk = price - stop
                if risk > 0:
                    size = int(round(1000000 / price))
                    if size > 0:
                        self.buy(size=size)
                        self.stop_price = stop
                        self.entry_bar = len(self.data) - 1
                        print(f"🚀🌙 Moon Dev LONG ENTRY @ {price:.2f} | Stop: {stop:.2f} | Size: {size} ✨")

            # Short entry
            elif (vi_bear_cross
                    and self.evf[-1] < 0 and evf_falling
                    and price < self.swing_low[-2]
                    and price < self.ema[-1]):
                stop = price + self.atr[-1] * self.atr_mult
                risk = stop - price
                if risk > 0:
                    size = int(round(1000000 / price))
                    if size > 0:
                        self.sell(size=size)
                        self.stop_price = stop
                        self.entry_bar = len(self.data) - 1
                        print(f"🔻🌙 Moon Dev SHORT ENTRY @ {price:.2f} | Stop: {stop:.2f} | Size: {size} ✨")


bt = Backtest(data, VortexForce, cash=1000000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)