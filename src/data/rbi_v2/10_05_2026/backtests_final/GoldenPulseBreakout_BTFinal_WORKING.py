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

print("🌙✨ GoldenPulse Breakout data loaded! Rows:", len(data), "🚀")


class GoldenPulseBreakout(Strategy):
    ema_fast_period = 50
    ema_slow_period = 200
    adx_period = 14
    atr_period = 14
    rsi_period = 5
    adx_threshold = 25
    adx_exit_threshold = 20
    rsi_overbought = 70
    rsi_oversold = 30
    atr_multiplier = 2.0
    risk_pct = 0.02

    def init(self):
        self.ema_fast = self.I(talib.EMA, self.data.Close, timeperiod=self.ema_fast_period)
        self.ema_slow = self.I(talib.EMA, self.data.Close, timeperiod=self.ema_slow_period)
        self.adx = self.I(talib.ADX, self.data.High, self.data.Low, self.data.Close, timeperiod=self.adx_period)
        self.plus_di = self.I(talib.PLUS_DI, self.data.High, self.data.Low, self.data.Close, timeperiod=self.adx_period)
        self.minus_di = self.I(talib.MINUS_DI, self.data.High, self.data.Low, self.data.Close, timeperiod=self.adx_period)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period)
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=self.rsi_period)

        self.entry_price = None
        self.stop_price = None
        self.trade_direction = None

        print("🌙✨ GoldenPulse indicators initialized! 🚀")

    def next(self):
        price = self.data.Close[-1]
        atr = self.atr[-1]
        adx = self.adx[-1]
        rsi = self.rsi[-1]

        if len(self.data) < self.ema_slow_period + 2:
            return

        if self.position:
            if self.trade_direction == 'long':
                if rsi > self.rsi_overbought:
                    print(f"🌙✨ RSI overbought ({rsi:.2f}) — exiting LONG at {price:.2f} 🚀")
                    self.position.close()
                    self.trade_direction = None
                elif adx < self.adx_exit_threshold:
                    print(f"🌙✨ ADX weak ({adx:.2f}) — exiting LONG at {price:.2f} 🚀")
                    self.position.close()
                    self.trade_direction = None
                elif (self.ema_slow[-2] < self.ema_fast[-2] and self.ema_slow[-1] > self.ema_fast[-1]):
                    print(f"🌙✨ Death cross — exiting LONG at {price:.2f} 🚀")
                    self.position.close()
                    self.trade_direction = None
                elif self.stop_price and price <= self.stop_price:
                    print(f"🌙🛑 Stop hit — exiting LONG at {price:.2f} 🚀")
                    self.position.close()
                    self.trade_direction = None

            elif self.trade_direction == 'short':
                if rsi < self.rsi_oversold:
                    print(f"🌙✨ RSI oversold ({rsi:.2f}) — exiting SHORT at {price:.2f} 🚀")
                    self.position.close()
                    self.trade_direction = None
                elif adx < self.adx_exit_threshold:
                    print(f"🌙✨ ADX weak ({adx:.2f}) — exiting SHORT at {price:.2f} 🚀")
                    self.position.close()
                    self.trade_direction = None
                elif (self.ema_fast[-2] < self.ema_slow[-2] and self.ema_fast[-1] > self.ema_slow[-1]):
                    print(f"🌙✨ Golden cross — exiting SHORT at {price:.2f} 🚀")
                    self.position.close()
                    self.trade_direction = None
                elif self.stop_price and price >= self.stop_price:
                    print(f"🌙🛑 Stop hit — exiting SHORT at {price:.2f} 🚀")
                    self.position.close()
                    self.trade_direction = None
            return

        # Long entry
        if (self.ema_fast[-2] < self.ema_slow[-2] and self.ema_fast[-1] > self.ema_slow[-1]) and adx > self.adx_threshold and price > self.ema_slow[-1]:
            stop = price - self.atr_multiplier * atr
            risk_per_unit = price - stop
            if risk_per_unit > 0:
                risk_amount = self.equity * self.risk_pct
                size = int(round(risk_amount / risk_per_unit))
                if size > 0:
                    print(f"🌙🚀 LONG entry at {price:.2f} | ADX={adx:.2f} | ATR={atr:.2f} | Size={size} ✨")
                    self.buy(size=size)
                    self.entry_price = price
                    self.stop_price = stop
                    self.trade_direction = 'long'

        # Short entry
        elif (self.ema_slow[-2] < self.ema_fast[-2] and self.ema_slow[-1] > self.ema_fast[-1]) and adx > self.adx_threshold and price < self.ema_slow[-1]:
            stop = price + self.atr_multiplier * atr
            risk_per_unit = stop - price
            if risk_per_unit > 0:
                risk_amount = self.equity * self.risk_pct
                size = int(round(risk_amount / risk_per_unit))
                if size > 0:
                    print(f"🌙🚀 SHORT entry at {price:.2f} | ADX={adx:.2f} | ATR={atr:.2f} | Size={size} ✨")
                    self.sell(size=size)
                    self.entry_price = price
                    self.stop_price = stop
                    self.trade_direction = 'short'


bt = Backtest(data, GoldenPulseBreakout, cash=1000000, commission=0.002)
stats = bt.run()
print(stats)
print(stats._strategy)