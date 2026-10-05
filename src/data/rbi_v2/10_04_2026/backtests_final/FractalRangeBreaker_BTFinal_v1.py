import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Load data
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Ensure proper column mapping
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

print("🌙✨ Data loaded successfully! Shape:", data.shape)
print("🚀 Moon Dev FractalRangeBreaker warming up...")


class FractalRangeBreaker(Strategy):
    lookback = 20
    rsi_period = 14
    atr_period = 14
    vol_ma_period = 20
    rsi_long = 55
    rsi_short = 45
    vol_mult = 1.5
    risk_pct = 0.01
    time_stop_bars = 10

    def init(self):
        self.range_high = self.I(talib.MAX, self.data.High, timeperiod=self.lookback)
        self.range_low = self.I(talib.MIN, self.data.Low, timeperiod=self.lookback)
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=self.rsi_period)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period)
        self.vol_ma = self.I(talib.SMA, self.data.Volume, timeperiod=self.vol_ma_period)

        # Fractal detection: 5-bar pattern (2 left, 2 right)
        # Use np.asarray to convert _Array to plain numpy arrays
        high = np.asarray(self.data.High)
        low = np.asarray(self.data.Low)
        n = len(high)
        bearish_fractal = np.full(n, np.nan)
        bullish_fractal = np.full(n, np.nan)

        for i in range(2, n - 2):
            # Bearish fractal: high surrounded by lower highs
            if (high[i] > high[i-1] and high[i] > high[i-2] and
                high[i] > high[i+1] and high[i] > high[i+2]):
                bearish_fractal[i] = high[i]
            # Bullish fractal: low surrounded by higher lows
            if (low[i] < low[i-1] and low[i] < low[i-2] and
                low[i] < low[i+1] and low[i] < low[i+2]):
                bullish_fractal[i] = low[i]

        # Forward-fill fractals and store as plain arrays
        bearish_ffill = pd.Series(bearish_fractal).ffill().values
        bullish_ffill = pd.Series(bullish_fractal).ffill().values

        self.bearish_fractal = self.I(lambda: bearish_ffill, name='bearish_fractal')
        self.bullish_fractal = self.I(lambda: bullish_ffill, name='bullish_fractal')

        # Precompute ATR SMA so we don't call self.I() inside next()
        self.atr_ma = self.I(talib.SMA, self.atr, timeperiod=50)

        self.entry_bar = None
        self.entry_price = None
        self.stop_price = None
        self.target_price = None
        self.trail_active = False
        self.trail_stop = None
        self.highest_high = None
        self.lowest_low = None

    def next(self):
        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]

        # Manage open position
        if self.position:
            bars_held = len(self.data) - 1 - self.entry_bar

            if self.position.is_long:
                self.highest_high = max(self.highest_high, high)

                # Trailing stop activation after 1x range width profit
                range_width = self.range_high[-1] - self.range_low[-1]
                if not self.trail_active and self.highest_high >= self.entry_price + range_width:
                    self.trail_active = True
                    print(f"🌙 Trailing stop activated LONG at {self.highest_high:.2f}")

                if self.trail_active:
                    new_trail = self.highest_high - self.atr[-1]
                    self.trail_stop = max(self.trail_stop, new_trail) if self.trail_stop else new_trail

                # Check stops
                if self.trail_active and self.trail_stop and low <= self.trail_stop:
                    print(f"🚀 Trailing stop hit LONG @ {self.trail_stop:.2f}")
                    self.position.close()
                    self._reset()
                    return
                if self.stop_price and low <= self.stop_price:
                    print(f"💥 Stop loss hit LONG @ {self.stop_price:.2f}")
                    self.position.close()
                    self._reset()
                    return
                if self.target_price and high >= self.target_price:
                    print(f"✨ Target hit LONG @ {self.target_price:.2f}")
                    self.position.close()
                    self._reset()
                    return
                if bars_held >= self.time_stop_bars:
                    print(f"⏰ Time stop LONG after {bars_held} bars")
                    self.position.close()
                    self._reset()
                    return

            elif self.position.is_short:
                self.lowest_low = min(self.lowest_low, low)

                range_width = self.range_high[-1] - self.range_low[-1]
                if not self.trail_active and self.lowest_low <= self.entry_price - range_width:
                    self.trail_active = True
                    print(f"🌙 Trailing stop activated SHORT at {self.lowest_low:.2f}")

                if self.trail_active:
                    new_trail = self.lowest_low + self.atr[-1]
                    self.trail_stop = min(self.trail_stop, new_trail) if self.trail_stop else new_trail

                if self.trail_active and self.trail_stop and high >= self.trail_stop:
                    print(f"🚀 Trailing stop hit SHORT @ {self.trail_stop:.2f}")
                    self.position.close()
                    self._reset()
                    return
                if self.stop_price and high >= self.stop_price:
                    print(f"💥 Stop loss hit SHORT @ {self.stop_price:.2f}")
                    self.position.close()
                    self._reset()
                    return
                if self.target_price and low <= self.target_price:
                    print(f"✨ Target hit SHORT @ {self.target_price:.2f}")
                    self.position.close()
                    self._reset()
                    return
                if bars_held >= self.time_stop_bars:
                    print(f"⏰ Time stop SHORT after {bars_held} bars")
                    self.position.close()
                    self._reset()
                    return
            return

        # Entry logic
        if len(self.data) < self.lookback + 5:
            return

        range_high = self.range_high[-1]
        range_low = self.range_low[-1]
        rsi = self.rsi[-1]
        atr = self.atr[-1]
        vol_ma = self.vol_ma[-1]
        volume = self.data.Volume[-1]
        bear_fr = self.bearish_fractal[-1]
        bull_fr = self.bullish_fractal[-1]

        if np.isnan(range_high) or np.isnan(range_low) or np.isnan(rsi) or np.isnan(atr) or np.isnan(vol_ma):
            return

        # Volatility filter (use precomputed atr_ma from init)
        atr_ma = self.atr_ma[-1]
        if not np.isnan(atr_ma) and atr_ma > 0:
            if atr < 0.2 * atr_ma or atr > 3.0 * atr_ma:
                return

        vol_ok = volume > self.vol_mult * vol_ma
        range_width = range_high - range_low

        # Long entry
        if (price > range_high and not np.isnan(bear_fr) and price > bear_fr
                and rsi > self.rsi_long and vol_ok):
            stop_fractal = bull_fr if not np.isnan(bull_fr) and bull_fr < price else price - atr
            stop = max(stop_fractal, price - atr)
            risk = price - stop
            if risk > 0:
                size = int(round((self.equity * self.risk_pct) / risk))
                if size > 0:
                    self.buy(size=size)
                    self.entry_bar = len(self.data) - 1
                    self.entry_price = price
                    self.stop_price = stop
                    self.target_price = price + 2 * range_width
                    self.trail_active = False
                    self.trail_stop = None
                    self.highest_high = high
                    print(f"🚀 LONG ENTRY @ {price:.2f} | SL: {stop:.2f} | TP: {self.target_price:.2f} | RSI: {rsi:.1f} | Size: {size}")

        # Short entry
        elif (price < range_low and not np.isnan(bull_fr) and price < bull_fr
                and rsi < self.rsi_short and vol_ok):
            stop_fractal = bear_fr if not np.isnan(bear_fr) and bear_fr > price else price + atr
            stop = min(stop_fractal, price + atr)
            risk = stop - price
            if risk > 0:
                size = int(round((self.equity * self.risk_pct) / risk))
                if size > 0:
                    self.sell(size=size)
                    self.entry_bar = len(self.data) - 1
                    self.entry_price = price
                    self.stop_price = stop
                    self.target_price = price - 2 * range_width
                    self.trail_active = False
                    self.trail_stop = None
                    self.lowest_low = low
                    print(f"🚀 SHORT ENTRY @ {price:.2f} | SL: {stop:.2f} | TP: {self.target_price:.2f} | RSI: {rsi:.1f} | Size: {size}")

    def _reset(self):
        self.entry_bar = None
        self.entry_price = None
        self.stop_price = None
        self.target_price = None
        self.trail_active = False
        self.trail_stop = None
        self.highest_high = None
        self.lowest_low = None


bt = Backtest(data, FractalRangeBreaker, cash=1000000, commission=0.001, exclusive_orders=True)
stats = bt.run()
print(stats)
print(stats._strategy)