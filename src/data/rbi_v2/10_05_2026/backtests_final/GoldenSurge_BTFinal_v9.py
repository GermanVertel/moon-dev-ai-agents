import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# 🌙 Moon Dev GoldenSurge Backtest ✨🚀

data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# 🌙 Clean columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# 🌙 Proper mapping
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
data = data.dropna()

print("🌙✨ Moon Dev Data Loaded:", data.shape, "🚀")


class GoldenSurge(Strategy):
    ema_fast = 50
    ema_slow = 200
    adx_period = 14
    adx_entry = 25
    adx_exit = 20
    rsi_period = 14
    rsi_overbought = 70
    atr_period = 14
    atr_mult = 2.0
    risk_pct = 0.02

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low

        self.ema50 = self.I(talib.EMA, close, timeperiod=self.ema_fast)
        self.ema200 = self.I(talib.EMA, close, timeperiod=self.ema_slow)
        self.adx = self.I(talib.ADX, high, low, close, timeperiod=self.adx_period)
        self.rsi = self.I(talib.RSI, close, timeperiod=self.rsi_period)
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        print("🌙✨ GoldenSurge indicators initialized 🚀")

    def next(self):
        price = self.data.Close[-1]

        if len(self.data) < self.ema_slow + 2:
            return

        ema50_now = self.ema50[-1]
        ema200_now = self.ema200[-1]
        ema50_prev = self.ema50[-2]
        ema200_prev = self.ema200[-2]

        adx_now = self.adx[-1]
        adx_prev = self.adx[-2]
        rsi_now = self.rsi[-1]
        atr_now = self.atr[-1]

        # 🌙 Skip if indicators are NaN (warmup)
        if (np.isnan(ema50_now) or np.isnan(ema200_now) or
                np.isnan(ema50_prev) or np.isnan(ema200_prev) or
                np.isnan(adx_now) or np.isnan(adx_prev) or
                np.isnan(rsi_now) or np.isnan(atr_now)):
            return

        # 🌙 Manual crossover detection (no backtesting.lib)
        cross_up = (ema50_prev <= ema200_prev) and (ema50_now > ema200_now)
        adx_ok = adx_now > self.adx_entry and adx_now > adx_prev
        close_above = price > ema50_now and price > ema200_now

        # 🌙 Entry
        if not self.position:
            if cross_up and adx_ok and close_above:
                stop_price = price - self.atr_mult * atr_now
                risk_per_unit = price - stop_price
                if risk_per_unit > 0:
                    risk_amount = self.equity * self.risk_pct
                    # 🌙 Moon Dev fix: fraction of equity for sizing (0 < size < 1)
                    size_units = risk_amount / risk_per_unit
                    size_frac = size_units * price / self.equity
                    size_frac = max(0.01, min(0.99, size_frac))
                    print(f"🌙🚀 GOLDEN CROSS! Entering LONG @ {price:.2f} | ADX={adx_now:.2f} | RSI={rsi_now:.2f} | size={size_frac:.4f}")
                    self.buy(size=size_frac, sl=stop_price)

        # ✨ Exit
        else:
            if rsi_now > self.rsi_overbought:
                print(f"✨💰 RSI OVERBOUGHT EXIT @ {price:.2f} | RSI={rsi_now:.2f}")
                self.position.close()
            elif adx_now < self.adx_exit:
                print(f"🌙⚠️ ADX WEAK EXIT @ {price:.2f} | ADX={adx_now:.2f}")
                self.position.close()


# 🌙 Moon Dev debug: verify data before running
print("🌙 Data head:\n", data.head())
print("🌙 Data columns:", list(data.columns))
print("🌙 Data length:", len(data))

bt = Backtest(data, GoldenSurge, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)