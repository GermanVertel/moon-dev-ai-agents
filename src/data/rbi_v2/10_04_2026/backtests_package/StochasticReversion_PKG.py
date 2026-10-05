import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

print("🌙 Moon Dev Backtest AI - StochasticReversion Strategy Loading... ✨")

data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
print(f"🚀 Data loaded: {len(data)} bars from {data.index[0]} to {data.index[-1]}")


class StochasticReversion(Strategy):
    stoch_k = 14
    stoch_d = 3
    stoch_smooth = 3
    atr_period = 100
    vol_lookback = 100
    vol_percentile = 80
    oversold = 20
    overbought = 80
    risk_pct = 0.02
    swing_lookback = 20
    time_stop = 20
    atr_stop_mult = 1.5

    def init(self):
        print("🌙 Initializing indicators... ✨")
        self.k, self.d = self.I(
            talib.STOCH,
            self.data.High,
            self.data.Low,
            self.data.Close,
            fastk_period=self.stoch_k,
            slowk_period=self.stoch_smooth,
            slowk_matype=0,
            slowd_period=self.stoch_d,
            slowd_matype=0,
        )
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period)
        self.vol_rank = self.I(
            lambda s: pd.Series(s).rolling(self.vol_lookback).apply(
                lambda x: (x.iloc[-1] > x.quantile(self.vol_percentile / 100.0)) * 1.0, raw=False
            ).values,
            self.atr,
            name="VolRank",
        )
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=self.swing_lookback)
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=self.swing_lookback)
        self.entry_bar = None
        self.entry_price = None
        self.stop_price = None
        self.target_price = None
        print("✅ Indicators ready! 🚀")

    def next(self):
        if len(self.data) < max(self.atr_period, self.vol_lookback, self.swing_lookback) + 5:
            return

        price = self.data.Close[-1]
        k_now, k_prev = self.k[-1], self.k[-2]
        d_now, d_prev = self.d[-1], self.d[-2]
        atr_now = self.atr[-1]
        high_vol = self.vol_rank[-1] == 1.0

        # Manage open position
        if self.position:
            bars_held = len(self.data) - self.entry_bar
            # Profit target
            if self.position.is_long and k_now >= self.overbought:
                print(f"🌙✨ Long TP hit: Stoch %K={k_now:.1f} >= {self.overbought}. Closing! 🚀")
                self.position.close()
                return
            if self.position.is_short and k_now <= self.oversold:
                print(f"🌙✨ Short TP hit: Stoch %K={k_now:.1f} <= {self.oversold}. Closing! 🚀")
                self.position.close()
                return
            # Stop loss
            if self.position.is_long and self.data.Low[-1] <= self.stop_price:
                print(f"🛑 Long SL hit at {self.stop_price:.2f}. Closing! 🌙")
                self.position.close()
                return
            if self.position.is_short and self.data.High[-1] >= self.stop_price:
                print(f"🛑 Short SL hit at {self.stop_price:.2f}. Closing! 🌙")
                self.position.close()
                return
            # Time stop
            if bars_held >= self.time_stop:
                print(f"⏰ Time stop after {bars_held} bars. Closing! 🌙")
                self.position.close()
                return
            return

        # Entry logic
        if not high_vol:
            return

        long_cross = k_prev < d_prev and k_now > d_now and k_prev < self.oversold
        short_cross = k_prev > d_prev and k_now < d_now and k_prev > self.overbought

        if long_cross:
            stop = self.swing_low[-1] - self.atr_stop_mult * atr_now
            risk = price - stop
            if risk <= 0:
                return
            size = int(round((self.equity * self.risk_pct) / risk))
            if size <= 0:
                return
            print(f"🌙🚀 LONG entry @ {price:.2f} | Stoch cross up from oversold | Stop={stop:.2f} | Size={size}")
            self.buy(size=size)
            self.entry_bar = len(self.data)
            self.entry_price = price
            self.stop_price = stop
            self.target_price = price + 2 * risk

        elif short_cross:
            stop = self.swing_high[-1] + self.atr_stop_mult * atr_now
            risk = stop - price
            if risk <= 0:
                return
            size = int(round((self.equity * self.risk_pct) / risk))
            if size <= 0:
                return
            print(f"🌙🚀 SHORT entry @ {price:.2f} | Stoch cross down from overbought | Stop={stop:.2f} | Size={size}")
            self.sell(size=size)
            self.entry_bar = len(self.data)
            self.entry_price = price
            self.stop_price = stop
            self.target_price = price - 2 * risk


bt = Backtest(data, StochasticReversion, cash=1_000_000, commission=0.001, exclusive_orders=True)
stats = bt.run()
print(stats)
print(stats._strategy)