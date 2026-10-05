import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Load data
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Rename columns properly
data = data.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})

# Set datetime as index
if 'datetime' in data.columns:
    data = data.set_index(pd.to_datetime(data['datetime']))
    data = data.drop(columns=['datetime'])

data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
data = data.dropna()

# Ensure float64 dtype for talib
data['Open'] = data['Open'].astype(np.float64)
data['High'] = data['High'].astype(np.float64)
data['Low'] = data['Low'].astype(np.float64)
data['Close'] = data['Close'].astype(np.float64)
data['Volume'] = data['Volume'].astype(np.float64)

print("🌙 Moon Dev VolatilitySurge Backtest Loading... ✨")
print(f"📊 Data shape: {data.shape}")
print(f"🚀 Date range: {data.index[0]} to {data.index[-1]}")


class VolatilitySurge(Strategy):
    atr_period = 20
    atr_baseline = 90
    vol_ma_period = 50
    donchian_period = 20
    risk_pct = 0.01
    rr_ratio = 2.0
    atr_stop_mult = 1.5
    time_stop_bars = 15

    def init(self):
        print("🌙 Initializing VolatilitySurge indicators... ✨")
        high = self.data.High.astype(np.float64)
        low = self.data.Low.astype(np.float64)
        close = self.data.Close.astype(np.float64)
        volume = self.data.Volume.astype(np.float64)

        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)
        self.atr_baseline_sma = self.I(talib.SMA, self.atr, timeperiod=self.atr_baseline)
        self.vol_ma = self.I(talib.SMA, volume, timeperiod=self.vol_ma_period)
        self.donchian_high = self.I(talib.MAX, high, timeperiod=self.donchian_period)
        self.donchian_low = self.I(talib.MIN, low, timeperiod=self.donchian_period)
        self.bars_in_trade = 0
        self.entry_price = 0
        self.stop_price = 0
        self.target_price = 0
        self.breakout_level = 0
        self.trade_dir = 0
        print("🚀 Indicators armed and ready!")

    def next(self):
        price = self.data.Close[-1]

        # Manage open position
        if self.position:
            self.bars_in_trade += 1

            if self.trade_dir == 1:  # Long
                if self.data.High[-1] >= self.target_price:
                    print(f"🎯 MOON DEV TARGET HIT LONG @ {self.target_price:.2f} | Price: {price:.2f} 🌙")
                    self.position.close()
                    self._reset_trade()
                    return
                if self.data.Low[-1] <= self.stop_price:
                    print(f"🛑 MOON DEV STOP HIT LONG @ {self.stop_price:.2f} | Price: {price:.2f} 💥")
                    self.position.close()
                    self._reset_trade()
                    return
                if price < self.breakout_level:
                    print(f"⚠️ MOON DEV BREAKOUT FAILURE LONG | Close {price:.2f} < Level {self.breakout_level:.2f} ✨")
                    self.position.close()
                    self._reset_trade()
                    return
                if self.bars_in_trade >= self.time_stop_bars:
                    print(f"⏰ MOON DEV TIME STOP LONG after {self.bars_in_trade} bars 🌙")
                    self.position.close()
                    self._reset_trade()
                    return

            elif self.trade_dir == -1:  # Short
                if self.data.Low[-1] <= self.target_price:
                    print(f"🎯 MOON DEV TARGET HIT SHORT @ {self.target_price:.2f} | Price: {price:.2f} 🌙")
                    self.position.close()
                    self._reset_trade()
                    return
                if self.data.High[-1] >= self.stop_price:
                    print(f"🛑 MOON DEV STOP HIT SHORT @ {self.stop_price:.2f} | Price: {price:.2f} 💥")
                    self.position.close()
                    self._reset_trade()
                    return
                if price > self.breakout_level:
                    print(f"⚠️ MOON DEV BREAKDOWN FAILURE SHORT | Close {price:.2f} > Level {self.breakout_level:.2f} ✨")
                    self.position.close()
                    self._reset_trade()
                    return
                if self.bars_in_trade >= self.time_stop_bars:
                    print(f"⏰ MOON DEV TIME STOP SHORT after {self.bars_in_trade} bars 🌙")
                    self.position.close()
                    self._reset_trade()
                    return
            return

        # Need enough data
        if len(self.data) < max(self.atr_baseline, self.vol_ma_period, self.donchian_period) + 2:
            return

        atr_now = self.atr[-1]
        atr_base = self.atr_baseline_sma[-1]
        vol_now = self.data.Volume[-1]
        vol_avg = self.vol_ma[-1]

        if np.isnan(atr_now) or np.isnan(atr_base) or np.isnan(vol_avg):
            return

        vol_expansion = atr_now > atr_base
        volume_spike = vol_now > vol_avg

        if not (vol_expansion and volume_spike):
            return

        prior_high = self.donchian_high[-2]
        prior_low = self.donchian_low[-2]

        if np.isnan(prior_high) or np.isnan(prior_low):
            return

        # Long breakout
        if price > prior_high:
            stop_dist = max(self.atr_stop_mult * atr_now, price - self.data.Low[-1])
            if stop_dist <= 0:
                return
            stop_price = price - stop_dist
            target_price = price + self.rr_ratio * stop_dist
            risk_amount = self.equity * self.risk_pct
            size = int(round(risk_amount / stop_dist))
            if size < 1:
                return

            self.entry_price = price
            self.stop_price = stop_price
            self.target_price = target_price
            self.breakout_level = prior_high
            self.trade_dir = 1
            self.bars_in_trade = 0

            print(f"🚀🌙 MOON DEV LONG BREAKOUT! Entry: {price:.2f} | Stop: {stop_price:.2f} | Target: {target_price:.2f} | Size: {size} ✨")
            self.buy(size=size)

        # Short breakdown
        elif price < prior_low:
            stop_dist = max(self.atr_stop_mult * atr_now, self.data.High[-1] - price)
            if stop_dist <= 0:
                return
            stop_price = price + stop_dist
            target_price = price - self.rr_ratio * stop_dist
            risk_amount = self.equity * self.risk_pct
            size = int(round(risk_amount / stop_dist))
            if size < 1:
                return

            self.entry_price = price
            self.stop_price = stop_price
            self.target_price = target_price
            self.breakout_level = prior_low
            self.trade_dir = -1
            self.bars_in_trade = 0

            print(f"🔻🌙 MOON DEV SHORT BREAKDOWN! Entry: {price:.2f} | Stop: {stop_price:.2f} | Target: {target_price:.2f} | Size: {size} ✨")
            self.sell(size=size)

    def _reset_trade(self):
        self.bars_in_trade = 0
        self.entry_price = 0
        self.stop_price = 0
        self.target_price = 0
        self.breakout_level = 0
        self.trade_dir = 0


bt = Backtest(data, VolatilitySurge, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)