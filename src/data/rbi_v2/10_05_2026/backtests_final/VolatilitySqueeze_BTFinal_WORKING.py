import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

print("🌙✨ Moon Dev's VolatilitySqueeze Backtest Initializing... 🚀")

data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"

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

data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

print(f"🌙 Data loaded: {len(data)} bars from {data.index[0]} to {data.index[-1]} ✨")


class VolatilitySqueeze(Strategy):
    bb_period = 20
    bb_std = 2.0
    adx_period = 14
    adx_threshold = 25
    atr_period = 14
    sma_period = 200
    atr_mult = 1.5
    risk_reward = 2.0
    risk_pct = 0.02

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low

        self.bb_upper, self.bb_mid, self.bb_lower = self.I(
            talib.BBANDS, close, timeperiod=self.bb_period,
            nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0
        )
        self.adx = self.I(talib.ADX, high, low, close, timeperiod=self.adx_period)
        self.pdi = self.I(talib.PLUS_DI, high, low, close, timeperiod=self.adx_period)
        self.mdi = self.I(talib.MINUS_DI, high, low, close, timeperiod=self.adx_period)
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)
        self.sma200 = self.I(talib.SMA, close, timeperiod=self.sma_period)

        self.stop_price = None
        self.tp_price = None
        self.entry_price = None
        self.risk_per_unit = None

        print("🌙 Indicators initialized: BB, ADX, +DI, -DI, ATR, SMA200 ✨")

    def next(self):
        price = self.data.Close[-1]

        if len(self.data) < self.sma_period + 5:
            return

        if np.isnan(self.adx[-1]) or np.isnan(self.bb_upper[-1]) or np.isnan(self.sma200[-1]):
            return

        # Manage open position
        if self.position:
            if self.position.is_long:
                if self.stop_price is not None and self.data.Low[-1] <= self.stop_price:
                    print(f"🛑 Long stop hit at {self.stop_price:.2f} | Price: {price:.2f} 🌙")
                    self.position.close()
                    self._reset()
                    return
                if self.tp_price is not None and self.data.High[-1] >= self.tp_price:
                    print(f"🎯 Long TP hit at {self.tp_price:.2f} | Price: {price:.2f} 🚀")
                    self.position.close()
                    self._reset()
                    return
                if price < self.bb_mid[-1] and self.adx[-1] < self.adx_threshold:
                    print(f"⚠️ Long early exit: re-entered BB & ADX weak ({self.adx[-1]:.2f}) 🌙")
                    self.position.close()
                    self._reset()
                    return
            elif self.position.is_short:
                if self.stop_price is not None and self.data.High[-1] >= self.stop_price:
                    print(f"🛑 Short stop hit at {self.stop_price:.2f} | Price: {price:.2f} 🌙")
                    self.position.close()
                    self._reset()
                    return
                if self.tp_price is not None and self.data.Low[-1] <= self.tp_price:
                    print(f"🎯 Short TP hit at {self.tp_price:.2f} | Price: {price:.2f} 🚀")
                    self.position.close()
                    self._reset()
                    return
                if price > self.bb_mid[-1] and self.adx[-1] < self.adx_threshold:
                    print(f"⚠️ Short early exit: re-entered BB & ADX weak ({self.adx[-1]:.2f}) 🌙")
                    self.position.close()
                    self._reset()
                    return
            return

        # ADX crossover detection (no backtesting.lib)
        adx_now = self.adx[-1]
        adx_prev = self.adx[-2]
        adx_prev2 = self.adx[-3] if len(self.adx) > 3 else np.nan

        crossed_up = (adx_prev < self.adx_threshold and adx_now >= self.adx_threshold) or \
                     (not np.isnan(adx_prev2) and adx_prev2 < self.adx_threshold and adx_now >= self.adx_threshold)

        if not crossed_up:
            return

        atr_val = self.atr[-1]
        if np.isnan(atr_val) or atr_val <= 0:
            return

        risk_dist = self.atr_mult * atr_val

        # Long entry
        if price > self.bb_upper[-1] and price > self.sma200[-1]:
            stop = price - risk_dist
            tp = price + risk_dist * self.risk_reward
            size = 0.99
            print(f"🚀 LONG breakout! Price: {price:.2f} > BB_upper: {self.bb_upper[-1]:.2f} | ADX: {adx_now:.2f} | SL: {stop:.2f} | TP: {tp:.2f} 🌙")
            self.buy(size=size)
            self.stop_price = stop
            self.tp_price = tp
            self.entry_price = price
            self.risk_per_unit = risk_dist

        # Short entry
        elif price < self.bb_lower[-1] and price < self.sma200[-1]:
            stop = price + risk_dist
            tp = price - risk_dist * self.risk_reward
            size = 0.99
            print(f"🔻 SHORT breakout! Price: {price:.2f} < BB_lower: {self.bb_lower[-1]:.2f} | ADX: {adx_now:.2f} | SL: {stop:.2f} | TP: {tp:.2f} 🌙")
            self.sell(size=size)
            self.stop_price = stop
            self.tp_price = tp
            self.entry_price = price
            self.risk_per_unit = risk_dist

    def _reset(self):
        self.stop_price = None
        self.tp_price = None
        self.entry_price = None
        self.risk_per_unit = None


bt = Backtest(data, VolatilitySqueeze, cash=1_000_000, commission=0.001)

print("🌙✨ Running Moon Dev's VolatilitySqueeze Backtest... 🚀")
stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙✨ Backtest complete! Moon Dev out! 🚀")