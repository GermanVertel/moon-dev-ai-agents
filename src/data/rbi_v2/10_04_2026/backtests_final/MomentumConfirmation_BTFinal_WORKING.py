import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

print("🌙✨ Moon Dev MomentumConfirmation Backtest Initializing... 🚀")

data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
print(f"🌙 Data loaded: {len(data)} bars 📊")


class MomentumConfirmation(Strategy):
    rsi_period = 14
    adx_period = 14
    atr_period = 14
    rsi_overbought = 70
    adx_threshold = 25
    atr_stop_mult = 2.0
    pivot_window = 5

    def init(self):
        print("🌙✨ Initializing indicators... 🚀")
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=self.rsi_period)
        self.adx = self.I(talib.ADX, self.data.High, self.data.Low, self.data.Close, timeperiod=self.adx_period)
        self.plus_di = self.I(talib.PLUS_DI, self.data.High, self.data.Low, self.data.Close, timeperiod=self.adx_period)
        self.minus_di = self.I(talib.MINUS_DI, self.data.High, self.data.Low, self.data.Close, timeperiod=self.adx_period)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period)
        self.pivot_high = self.I(talib.MAX, self.data.High, timeperiod=self.pivot_window)
        self.adx_peak = self.I(talib.MAX, self.adx, timeperiod=10)
        print("🌙 Indicators ready! ✨")

    def next(self):
        price = self.data.Close[-1]
        rsi = self.rsi[-1]
        rsi_prev = self.rsi[-2] if len(self.rsi) > 1 else rsi
        adx = self.adx[-1]
        adx_prev = self.adx[-2] if len(self.adx) > 1 else adx
        adx_peak = self.adx_peak[-1]
        plus_di = self.plus_di[-1]
        minus_di = self.minus_di[-1]
        atr = self.atr[-1]

        if np.isnan(rsi) or np.isnan(adx) or np.isnan(atr) or np.isnan(plus_di) or np.isnan(minus_di):
            return

        if not self.position:
            rsi_cross_up = rsi_prev <= self.rsi_overbought and rsi > self.rsi_overbought
            rsi_sustained = rsi > self.rsi_overbought
            adx_strong = adx > self.adx_threshold
            adx_rising = adx > adx_prev
            di_bullish = plus_di > minus_di

            if (rsi_cross_up or rsi_sustained) and adx_strong and adx_rising and di_bullish:
                stop_price = price - self.atr_stop_mult * atr
                risk = price - stop_price
                if risk <= 0:
                    return
                size = int(round(1000000 / price))
                if size < 1:
                    size = 1
                print(f"🌙🚀 LONG ENTRY | Price: {price:.2f} | RSI: {rsi:.2f} | ADX: {adx:.2f} | +DI: {plus_di:.2f} -DI: {minus_di:.2f} | Size: {size}")
                self.buy(size=size, sl=stop_price)

        else:
            entry_price = self.trades[-1].entry_price
            bearish_div = False
            if len(self.data.Close) > 2 * self.pivot_window:
                recent_price_high = self.data.High[-1]
                prev_price_high = self.pivot_high[-self.pivot_window - 1]
                if recent_price_high >= prev_price_high and rsi < self.rsi[-self.pivot_window - 1]:
                    bearish_div = True

            adx_declining = adx < adx_peak and adx < adx_prev
            rsi_weak = rsi < 50

            if bearish_div:
                print(f"🌙⚠️ EXIT: Bearish RSI Divergence | Price: {price:.2f} | RSI: {rsi:.2f}")
                self.position.close()
            elif adx_declining:
                print(f"🌙⚠️ EXIT: ADX Declining | Price: {price:.2f} | ADX: {adx:.2f}")
                self.position.close()
            elif rsi_weak:
                print(f"🌙⚠️ EXIT: RSI below 50 | Price: {price:.2f} | RSI: {rsi:.2f}")
                self.position.close()


bt = Backtest(data, MomentumConfirmation, cash=1000000, commission=0.002)
stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙✨ Backtest complete! 🚀")