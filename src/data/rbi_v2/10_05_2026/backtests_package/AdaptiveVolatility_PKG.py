import pandas as pd
import talib
from backtesting import Backtest, Strategy
import numpy as np

# Load and clean data
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'datetime': 'Date',
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})
data['Date'] = pd.to_datetime(data['Date'])
data = data.set_index('Date')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

print("🌙✨ Moon Dev: Data loaded successfully!")
print(f"🚀 Data shape: {data.shape}")
print(f"📊 Date range: {data.index[0]} to {data.index[-1]}")


class AdaptiveVolatility(Strategy):
    # Strategy parameters
    atr_period = 14
    ma_period = 20
    trend_period = 200
    atr_multiplier = 2.0
    stop_atr_mult = 1.0
    target_atr_mult = 2.0
    time_exit_bars = 10

    def init(self):
        print("🌙 Moon Dev: Initializing AdaptiveVolatility Strategy...")
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close,
                          timeperiod=self.atr_period)
        self.ma = self.I(talib.SMA, self.data.Close, timeperiod=self.ma_period)
        self.trend_ma = self.I(talib.SMA, self.data.Close, timeperiod=self.trend_period)
        self.entry_bar = None
        self.entry_price = None
        self.stop_price = None
        self.target_price = None
        print("✨ Moon Dev: Indicators ready! ATR, MA, Trend MA loaded.")

    def next(self):
        if len(self.data) < max(self.atr_period, self.ma_period, self.trend_period) + 1:
            return

        price = self.data.Close[-1]
        atr = self.atr[-1]
        ma = self.ma[-1]
        trend = self.trend_ma[-1]

        if np.isnan(atr) or np.isnan(ma) or np.isnan(trend):
            return

        upper_band = ma + atr * self.atr_multiplier
        lower_band = ma - atr * self.atr_multiplier

        # Manage open position
        if self.position:
            bars_held = len(self.data) - self.entry_bar

            # Check stop loss
            if self.position.is_long and self.data.Low[-1] <= self.stop_price:
                print(f"🛑 Moon Dev: LONG STOP hit at {self.stop_price:.2f} | Price: {price:.2f}")
                self.position.close()
                return
            if self.position.is_short and self.data.High[-1] >= self.stop_price:
                print(f"🛑 Moon Dev: SHORT STOP hit at {self.stop_price:.2f} | Price: {price:.2f}")
                self.position.close()
                return

            # Check profit target
            if self.position.is_long and self.data.High[-1] >= self.target_price:
                print(f"🎯 Moon Dev: LONG TARGET hit at {self.target_price:.2f} | Price: {price:.2f}")
                self.position.close()
                return
            if self.position.is_short and self.data.Low[-1] <= self.target_price:
                print(f"🎯 Moon Dev: SHORT TARGET hit at {self.target_price:.2f} | Price: {price:.2f}")
                self.position.close()
                return

            # Time-based exit
            if bars_held >= self.time_exit_bars:
                print(f"⏰ Moon Dev: TIME EXIT after {bars_held} bars | Price: {price:.2f}")
                self.position.close()
                return

            # Opposite signal exit
            if self.position.is_long and price < lower_band:
                print(f"🔄 Moon Dev: Opposite signal - closing LONG | Price: {price:.2f}")
                self.position.close()
                return
            if self.position.is_short and price > upper_band:
                print(f"🔄 Moon Dev: Opposite signal - closing SHORT | Price: {price:.2f}")
                self.position.close()
                return

            return

        # Entry logic
        long_signal = price > upper_band and price > trend
        short_signal = price < lower_band and price < trend

        if long_signal:
            stop = price - atr * self.stop_atr_mult
            target = price + atr * self.target_atr_mult
            risk = price - stop
            if risk <= 0:
                return
            size = int(round(1_000_000 / price))
            if size <= 0:
                return
            print(f"🚀 Moon Dev: LONG ENTRY | Price: {price:.2f} | Upper: {upper_band:.2f} | Trend: {trend:.2f} | Stop: {stop:.2f} | Target: {target:.2f} | Size: {size}")
            self.buy(size=size)
            self.entry_bar = len(self.data)
            self.entry_price = price
            self.stop_price = stop
            self.target_price = target

        elif short_signal:
            stop = price + atr * self.stop_atr_mult
            target = price - atr * self.target_atr_mult
            risk = stop - price
            if risk <= 0:
                return
            size = int(round(1_000_000 / price))
            if size <= 0:
                return
            print(f"🔻 Moon Dev: SHORT ENTRY | Price: {price:.2f} | Lower: {lower_band:.2f} | Trend: {trend:.2f} | Stop: {stop:.2f} | Target: {target:.2f} | Size: {size}")
            self.sell(size=size)
            self.entry_bar = len(self.data)
            self.entry_price = price
            self.stop_price = stop
            self.target_price = target


print("🌙✨ Starting Moon Dev AdaptiveVolatility Backtest...")
bt = Backtest(data, AdaptiveVolatility, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)