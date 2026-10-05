import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Load data
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

print("🌙 Moon Dev VolatilitySurge Backtest Initializing... ✨")
print(f"📊 Data loaded: {len(data)} bars")
print(f"🚀 Starting backtest...")

class VolatilitySurge(Strategy):
    atr_period = 2
    adx_period = 14
    adx_threshold = 30
    adx_exit_threshold = 25
    atr_multiplier = 2.0
    trailing_atr_mult = 1.5
    risk_pct = 0.01
    atr_sma_period = 20
    size = 0.99

    def init(self):
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period)
        self.adx = self.I(talib.ADX, self.data.High, self.data.Low, self.data.Close, timeperiod=self.adx_period)
        self.atr_sma = self.I(talib.SMA, self.atr, timeperiod=self.atr_sma_period)
        self.highest_high = self.I(talib.MAX, self.data.High, timeperiod=20)
        self.lowest_low = self.I(talib.MIN, self.data.Low, timeperiod=20)
        self.entry_price = None
        self.trail_stop = None
        self.trade_direction = None
        print("🌙 Indicators initialized: ATR(2), ADX(14), ATR-SMA(20) ✨")

    def next(self):
        if len(self.data) < 25:
            return

        price = self.data.Close[-1]
        prev_close = self.data.Close[-2]
        atr_val = self.atr[-1]
        adx_val = self.adx[-1]
        adx_prev = self.adx[-2] if len(self.adx) > 1 else 0
        atr_sma_val = self.atr_sma[-1]

        if np.isnan(atr_val) or np.isnan(adx_val) or np.isnan(atr_sma_val):
            return

        upper_band = prev_close + (self.atr_multiplier * atr_val)
        lower_band = prev_close - (self.atr_multiplier * atr_val)

        atr_compressed = atr_val < atr_sma_val * 0.5

        adx_cross_up = adx_prev < self.adx_threshold and adx_val > self.adx_threshold
        adx_cross_down = adx_prev > self.adx_exit_threshold and adx_val < self.adx_exit_threshold

        # Manage existing position - trailing stop
        if self.position:
            if self.trade_direction == 'long':
                new_trail = self.data.High[-1] - (self.trailing_atr_mult * atr_val)
                if self.trail_stop is None or new_trail > self.trail_stop:
                    self.trail_stop = new_trail
                if price <= self.trail_stop:
                    print(f"🌙 Long exit via trailing stop @ {price:.2f} | Trail: {self.trail_stop:.2f} 🚀")
                    self.position.close()
                    self.trail_stop = None
                    self.trade_direction = None
                    return
                if adx_cross_down:
                    print(f"🌙 Long exit via ADX weakening @ {price:.2f} | ADX: {adx_val:.2f} ✨")
                    self.position.close()
                    self.trail_stop = None
                    self.trade_direction = None
                    return

            elif self.trade_direction == 'short':
                new_trail = self.data.Low[-1] + (self.trailing_atr_mult * atr_val)
                if self.trail_stop is None or new_trail < self.trail_stop:
                    self.trail_stop = new_trail
                if price >= self.trail_stop:
                    print(f"🌙 Short exit via trailing stop @ {price:.2f} | Trail: {self.trail_stop:.2f} 🚀")
                    self.position.close()
                    self.trail_stop = None
                    self.trade_direction = None
                    return
                if adx_cross_down:
                    print(f"🌙 Short exit via ADX weakening @ {price:.2f} | ADX: {adx_val:.2f} ✨")
                    self.position.close()
                    self.trail_stop = None
                    self.trade_direction = None
                    return
            return

        if atr_compressed:
            return

        # Long entry
        if price > upper_band and adx_cross_up:
            stop_distance = self.trailing_atr_mult * atr_val
            if stop_distance <= 0:
                return
            position_size = self.size
            if position_size <= 0 or position_size >= 1:
                return
            print(f"🌙 LONG ENTRY @ {price:.2f} | Upper Band: {upper_band:.2f} | ADX: {adx_val:.2f} 🚀")
            self.buy(size=position_size)
            self.entry_price = price
            self.trail_stop = price - stop_distance
            self.trade_direction = 'long'
            return

        # Short entry
        if price < lower_band and adx_cross_up:
            stop_distance = self.trailing_atr_mult * atr_val
            if stop_distance <= 0:
                return
            position_size = self.size
            if position_size <= 0 or position_size >= 1:
                return
            print(f"🌙 SHORT ENTRY @ {price:.2f} | Lower Band: {lower_band:.2f} | ADX: {adx_val:.2f} 🚀")
            self.sell(size=position_size)
            self.entry_price = price
            self.trail_stop = price + stop_distance
            self.trade_direction = 'short'
            return


bt = Backtest(data, VolatilitySurge, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)