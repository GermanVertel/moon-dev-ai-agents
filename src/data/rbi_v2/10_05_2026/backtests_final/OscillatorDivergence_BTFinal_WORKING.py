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

print("🌙 Moon Dev OscillatorDivergence Backtest Loading... ✨")
print(f"📊 Data shape: {data.shape}")
print(f"🚀 Data range: {data.index[0]} to {data.index[-1]}")


class OscillatorDivergence(Strategy):
    rsi_period = 14
    adx_period = 14
    atr_period = 14
    ema_period = 50
    rsi_oversold = 20
    adx_threshold = 20
    atr_tp_mult = 2.0
    atr_sl_mult = 1.5
    risk_pct = 0.02
    max_bars = 12
    size = 1_000_000

    def init(self):
        print("🌙 Initializing Moon Dev OscillatorDivergence indicators... ✨")
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=self.rsi_period)
        self.adx = self.I(talib.ADX, self.data.High, self.data.Low, self.data.Close, timeperiod=self.adx_period)
        self.pdi = self.I(talib.PLUS_DI, self.data.High, self.data.Low, self.data.Close, timeperiod=self.adx_period)
        self.mdi = self.I(talib.MINUS_DI, self.data.High, self.data.Low, self.data.Close, timeperiod=self.adx_period)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period)
        self.ema = self.I(talib.EMA, self.data.Close, timeperiod=self.ema_period)
        self.bars_in_trade = 0
        print("🚀 Indicators ready to launch!")

    def next(self):
        price = self.data.Close[-1]

        # Manage open position
        if self.position:
            self.bars_in_trade += 1
            if self.bars_in_trade >= self.max_bars:
                print(f"⏰ Moon Dev Time Exit at {price:.2f} after {self.bars_in_trade} bars 🌙")
                self.position.close()
                self.bars_in_trade = 0
            return

        # Entry conditions
        rsi_ok = self.rsi[-1] < self.rsi_oversold
        adx_ok = self.adx[-1] > self.adx_threshold and self.mdi[-1] > self.pdi[-1]
        ema_ok = price < self.ema[-1]
        atr_ok = self.atr[-1] > 0

        if rsi_ok and adx_ok and ema_ok and atr_ok:
            atr_val = self.atr[-1]
            sl_price = price + self.atr_sl_mult * atr_val
            tp_price = price - self.atr_tp_mult * atr_val
            stop_dist = sl_price - price

            if stop_dist <= 0:
                return

            # Position sizing based on risk
            risk_amount = self.equity * self.risk_pct
            position_size = risk_amount / stop_dist
            position_size = int(round(position_size))

            if position_size < 1:
                position_size = 1

            print(f"🌙✨ SHORT SIGNAL! RSI={self.rsi[-1]:.2f} ADX={self.adx[-1]:.2f} -DI={self.mdi[-1]:.2f} +DI={self.pdi[-1]:.2f}")
            print(f"🚀 Entry={price:.2f} SL={sl_price:.2f} TP={tp_price:.2f} Size={position_size}")

            self.sell(size=position_size, sl=sl_price, tp=tp_price)
            self.bars_in_trade = 0


bt = Backtest(data, OscillatorDivergence, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)