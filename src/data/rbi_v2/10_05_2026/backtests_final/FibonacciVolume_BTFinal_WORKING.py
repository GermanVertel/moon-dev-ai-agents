import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# 🌙 Moon Dev's FibonacciVolume Backtest
print("🌙✨ Moon Dev AI initializing FibonacciVolume strategy... 🚀")

data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Map to proper case
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

if 'datetime' in data.columns:
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')

print(f"🌙 Data loaded: {len(data)} bars from {data.index[0]} to {data.index[-1]} ✨")


class FibonacciVolume(Strategy):
    # Parameters
    ema_fast_period = 50
    ema_slow_period = 200
    vol_sma_period = 20
    rsi_period = 14
    atr_period = 14
    swing_lookback = 50
    vol_reduction = 0.25  # 25% below vol SMA
    risk_pct = 0.02
    rr_target = 2.0
    time_stop_bars = 30
    fib_low = 0.382
    fib_high = 0.618
    fib_invalidation = 0.786

    def init(self):
        print("🌙 Initializing indicators... ✨")
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        # 🌙 Convert to double arrays for talib compatibility
        close_arr = np.asarray(close, dtype=np.float64)
        high_arr = np.asarray(high, dtype=np.float64)
        low_arr = np.asarray(low, dtype=np.float64)
        volume_arr = np.asarray(volume, dtype=np.float64)

        self.ema_fast = self.I(talib.EMA, close_arr, timeperiod=self.ema_fast_period)
        self.ema_slow = self.I(talib.EMA, close_arr, timeperiod=self.ema_slow_period)
        self.vol_sma = self.I(talib.SMA, volume_arr, timeperiod=self.vol_sma_period)
        self.rsi = self.I(talib.RSI, close_arr, timeperiod=self.rsi_period)
        self.atr = self.I(talib.ATR, high_arr, low_arr, close_arr, timeperiod=self.atr_period)

        # 🌙 MACD via talib — talib.MACD returns (macd, signal, hist) tuple
        # Wrap with a helper that returns a single output for self.I compatibility
        def _macd_line(close_arr, fastperiod=12, slowperiod=26, signalperiod=9):
            macd, signal, hist = talib.MACD(close_arr, fastperiod=fastperiod,
                                             slowperiod=slowperiod, signalperiod=signalperiod)
            return macd

        def _macd_signal(close_arr, fastperiod=12, slowperiod=26, signalperiod=9):
            macd, signal, hist = talib.MACD(close_arr, fastperiod=fastperiod,
                                             slowperiod=slowperiod, signalperiod=signalperiod)
            return signal

        self.macd = self.I(_macd_line, close_arr, fastperiod=12, slowperiod=26, signalperiod=9, name='MACD')
        self.macd_signal = self.I(_macd_signal, close_arr, fastperiod=12, slowperiod=26, signalperiod=9, name='MACD_signal')

        self.swing_high = self.I(talib.MAX, high_arr, timeperiod=self.swing_lookback)
        self.swing_low = self.I(talib.MIN, low_arr, timeperiod=self.swing_lookback)

        # Track state
        self.entry_bar = None
        self.entry_price = None
        self.stop_price = None
        self.target_price = None
        self.trade_dir = None
        self.prev_rsi_high = None
        self.prev_rsi_low = None
        print("🌙 Indicators ready! 🚀")

    def next(self):
        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        vol = self.data.Volume[-1]

        ema_f = self.ema_fast[-1]
        ema_s = self.ema_slow[-1]
        vol_sma = self.vol_sma[-1]
        rsi = self.rsi[-1]
        atr = self.atr[-1]
        macd = self.macd[-1]
        macd_sig = self.macd_signal[-1]

        if np.isnan(ema_s) or np.isnan(vol_sma) or np.isnan(rsi) or np.isnan(atr):
            return

        # ----- Manage open position -----
        if self.position:
            bars_held = len(self.data) - 1 - self.entry_bar

            # Time stop
            if bars_held >= self.time_stop_bars:
                print(f"⏰ Moon Dev time stop hit after {bars_held} bars — closing 🌙")
                self.position.close()
                self._reset()
                return

            if self.trade_dir == 'long':
                # MACD bearish crossover exit
                if self.macd[-2] >= self.macd_signal[-2] and macd < macd_sig:
                    print(f"🌙 Bearish MACD cross — exiting long at {price} ✨")
                    self.position.close()
                    self._reset()
                    return
                # RSI bearish divergence (price higher high, RSI lower high)
                if self.prev_rsi_high is not None and high > self._last_price_high() and rsi < self.prev_rsi_high:
                    print(f"🌙 Bearish RSI divergence — exiting long ✨")
                    self.position.close()
                    self._reset()
                    return
                if self.prev_rsi_high is not None and high > self._last_price_high():
                    self.prev_rsi_high = rsi
                # Invalidation
                if price < self.stop_price:
                    print(f"🛑 Stop hit on long at {price} 🌙")
                    self.position.close()
                    self._reset()
                    return
                # Target
                if high >= self.target_price:
                    print(f"🎯 Target hit on long at {price} 🚀")
                    self.position.close()
                    self._reset()
                    return

            elif self.trade_dir == 'short':
                # MACD bullish crossover exit
                if self.macd[-2] <= self.macd_signal[-2] and macd > macd_sig:
                    print(f"🌙 Bullish MACD cross — exiting short at {price} ✨")
                    self.position.close()
                    self._reset()
                    return
                if self.prev_rsi_low is not None and low < self._last_price_low() and rsi > self.prev_rsi_low:
                    print(f"🌙 Bullish RSI divergence — exiting short ✨")
                    self.position.close()
                    self._reset()
                    return
                if self.prev_rsi_low is not None and low < self._last_price_low():
                    self.prev_rsi_low = rsi
                if price > self.stop_price:
                    print(f"🛑 Stop hit on short at {price} 🌙")
                    self.position.close()
                    self._reset()
                    return
                if low <= self.target_price:
                    print(f"🎯 Target hit on short at {price} 🚀")
                    self.position.close()
                    self._reset()
                    return
            return

        # ----- Entry logic -----
        if np.isnan(self.swing_high[-1]) or np.isnan(self.swing_low[-1]):
            return

        sh = self.swing_high[-1]
        sl = self.swing_low[-1]
        rng = sh - sl
        if rng <= 0:
            return

        # Fib levels (retracement from swing high down in uptrend)
        fib_382 = sh - rng * 0.382
        fib_500 = sh - rng * 0.500
        fib_618 = sh - rng * 0.618
        fib_786 = sh - rng * 0.786

        uptrend = price > ema_s and ema_f > ema_s
        downtrend = price < ema_s and ema_f < ema_s

        # Reduced volume condition
        vol_ok = vol < vol_sma * (1 - self.vol_reduction)

        # ----- Long setup -----
        if uptrend and vol_ok:
            in_zone = fib_618 <= price <= fib_382
            if in_zone:
                entry = price
                stop = min(fib_786, entry - 1.5 * atr)
                target = sh
                risk = entry - stop
                reward = target - entry
                if risk > 0 and reward / risk >= self.rr_target:
                    size = int(round((self.equity * self.risk_pct) / risk))
                    if size > 0:
                        print(f"🌙🚀 LONG entry at {entry} | stop {stop} | target {target} | size {size} ✨")
                        self.buy(size=size)
                        self.entry_bar = len(self.data) - 1
                        self.entry_price = entry
                        self.stop_price = stop
                        self.target_price = target
                        self.trade_dir = 'long'
                        self.prev_rsi_high = rsi
                        return

        # ----- Short setup -----
        if downtrend and vol_ok:
            fib_382_s = sl + rng * 0.382
            fib_500_s = sl + rng * 0.500
            fib_618_s = sl + rng * 0.618
            fib_786_s = sl + rng * 0.786
            in_zone = fib_382_s <= price <= fib_618_s
            if in_zone:
                entry = price
                stop = max(fib_786_s, entry + 1.5 * atr)
                target = sl
                risk = stop - entry
                reward = entry - target
                if risk > 0 and reward / risk >= self.rr_target:
                    size = int(round((self.equity * self.risk_pct) / risk))
                    if size > 0:
                        print(f"🌙🚀 SHORT entry at {entry} | stop {stop} | target {target} | size {size} ✨")
                        self.sell(size=size)
                        self.entry_bar = len(self.data) - 1
                        self.entry_price = entry
                        self.stop_price = stop
                        self.target_price = target
                        self.trade_dir = 'short'
                        self.prev_rsi_low = rsi
                        return

    def _reset(self):
        self.entry_bar = None
        self.entry_price = None
        self.stop_price = None
        self.target_price = None
        self.trade_dir = None
        self.prev_rsi_high = None
        self.prev_rsi_low = None

    def _last_price_high(self):
        return self.data.High[-2] if len(self.data) > 1 else self.data.High[-1]

    def _last_price_low(self):
        return self.data.Low[-2] if len(self.data) > 1 else self.data.Low[-1]


bt = Backtest(data, FibonacciVolume, cash=1_000_000, commission=0.002)
stats = bt.run()
print(stats)
print(stats._strategy)