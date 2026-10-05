import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

print("🌙✨ Moon Dev VolumeReversal Backtest Starting! 🚀🌙")

# Load and clean data
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
data = data.set_index(pd.to_datetime(data['datetime']))
data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
data = data.astype(float)

print(f"🌙 Data loaded: {len(data)} bars ✨")


class VolumeReversal(Strategy):
    rsi_period = 14
    rsi_oversold = 30
    rsi_exit = 50
    bb_period = 20
    bb_dev = 2
    ema_period = 20
    sma_period = 50
    max_hold = 50
    risk_pct = 0.02

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        close_arr = np.asarray(close, dtype=np.float64)
        high_arr = np.asarray(high, dtype=np.float64)
        low_arr = np.asarray(low, dtype=np.float64)
        vol_arr = np.asarray(volume, dtype=np.float64)

        self.rsi = self.I(talib.RSI, close_arr, timeperiod=self.rsi_period)
        self.ema20 = self.I(talib.EMA, close_arr, timeperiod=self.ema_period)
        self.sma50 = self.I(talib.SMA, close_arr, timeperiod=self.sma_period)
        self.bb_up, self.bb_mid, self.bb_low = self.I(
            talib.BBANDS, close_arr, timeperiod=self.bb_period,
            nbdevup=self.bb_dev, nbdevdn=self.bb_dev, matype=0
        )
        self.macd, self.macd_sig, self.macd_hist = self.I(
            talib.MACD, close_arr, fastperiod=12, slowperiod=26, signalperiod=9
        )
        self.vol_sma = self.I(talib.SMA, vol_arr, timeperiod=20)
        self.high20 = self.I(talib.MAX, high_arr, timeperiod=20)
        self.low20 = self.I(talib.MIN, low_arr, timeperiod=20)

        self.entry_bar = None
        self.stop_price = None
        self.target_price = None

        print("🌙 Indicators initialized ✨")

    def next(self):
        price = self.data.Close[-1]
        rsi = self.rsi[-1]
        ema = self.ema20[-1]
        sma = self.sma50[-1]
        bb_u = self.bb_up[-1]
        bb_l = self.bb_low[-1]
        macd = self.macd[-1]
        macd_s = self.macd_sig[-1]
        vol = self.data.Volume[-1]
        vol_avg = self.vol_sma[-1]
        high20 = self.high20[-1]
        low20 = self.low20[-1]

        if self.position:
            new_stop = ema * 0.99
            if self.stop_price is None or new_stop > self.stop_price:
                self.stop_price = new_stop

            if price <= self.stop_price:
                print(f"🛑 Stop hit at {price:.2f} | stop {self.stop_price:.2f} 🌙")
                self.position.close()
                return
            if self.target_price and price >= self.target_price:
                print(f"🎯 Target hit at {price:.2f} | target {self.target_price:.2f} 🚀")
                self.position.close()
                return
            if rsi > 70:
                print(f"✨ RSI overbought {rsi:.1f}, exiting 🌙")
                self.position.close()
                return
            if self.entry_bar and (len(self.data) - self.entry_bar) > self.max_hold:
                print(f"⏰ Max hold reached, exiting at {price:.2f} 🌙")
                self.position.close()
                return
            return

        oversold = rsi < self.rsi_oversold
        downtrend = price < high20 * 0.8
        reversal = price > ema and macd > macd_s
        vol_confirm = vol > vol_avg * 1.2
        bb_width = (bb_u - bb_l) / price
        consolidation = bb_width < 0.08
        breakout = price > self.data.Close[-2] and price > bb_u * 0.995

        if oversold and reversal and vol_confirm and (consolidation or breakout):
            risk = price - low20
            if risk <= 0:
                return
            size = 0.95
            self.stop_price = low20 * 0.99
            self.target_price = price + risk * 2.5
            self.entry_bar = len(self.data)
            print(f"🚀🌙 ENTRY at {price:.2f} | RSI {rsi:.1f} | stop {self.stop_price:.2f} | target {self.target_price:.2f} | size {size}")
            self.buy(size=size, sl=self.stop_price)


bt = Backtest(data, VolumeReversal, cash=1000000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)