import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Load data
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')

# Clean columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper column mapping
data.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)

# Ensure datetime index
if 'datetime' in data.columns:
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')
    data.index.name = 'Datetime'

data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

# Ensure float64 dtype for talib compatibility
data['Open'] = data['Open'].astype(np.float64)
data['High'] = data['High'].astype(np.float64)
data['Low'] = data['Low'].astype(np.float64)
data['Close'] = data['Close'].astype(np.float64)
data['Volume'] = data['Volume'].astype(np.float64)

print("🌙 Moon Dev ExhaustionFade Backtest Loading... ✨")
print(f"📊 Data shape: {data.shape}")
print(f"🚀 Date range: {data.index[0]} to {data.index[-1]}")


class ExhaustionFade(Strategy):
    adx_period = 14
    adx_threshold = 20
    adx_exit_threshold = 25
    chaikin_fast = 3
    chaikin_slow = 10
    atr_period = 14
    atr_mult = 1.5
    ema_period = 20
    rsi_period = 14
    rsi_threshold = 65
    spike_std_mult = 2.0
    time_stop_bars = 18
    risk_pct = 0.01

    def init(self):
        print("🌙 Initializing Moon Dev indicators... ✨")
        high = self.data.High
        low = self.data.Low
        close = self.data.Close
        volume = self.data.Volume

        # ADX regime filter
        self.adx = self.I(talib.ADX, high, low, close, timeperiod=self.adx_period)

        # Chaikin Oscillator = EMA(ADL, 3) - EMA(ADL, 10)
        self.adl = self.I(talib.AD, high, low, close, volume)
        self.ema_fast = self.I(talib.EMA, self.adl, timeperiod=self.chaikin_fast)
        self.ema_slow = self.I(talib.EMA, self.adl, timeperiod=self.chaikin_slow)
        self.chaikin = self.I(lambda f, s: f - s, self.ema_fast, self.ema_slow, name='Chaikin')

        # ATR for stops / sizing
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        # EMA for spike detection
        self.ema20 = self.I(talib.EMA, close, timeperiod=self.ema_period)

        # RSI
        self.rsi = self.I(talib.RSI, close, timeperiod=self.rsi_period)

        # Rolling std of close for spike detection
        self.std20 = self.I(talib.STDDEV, close, timeperiod=20)

        # Track state
        self.entry_bar = None
        self.highest_since_entry = None
        self.stop_price = None
        self.target_price = None
        self.spike_high = None

    def next(self):
        price = self.data.Close[-1]

        # Manage open position
        if self.position:
            bars_held = len(self.data) - self.entry_bar

            # Trailing stop (short position)
            if self.position.is_short:
                self.highest_since_entry = min(self.highest_since_entry, self.data.Low[-1])
                trail = self.highest_since_entry + self.atr_mult * self.atr[-1]
                if trail < self.stop_price:
                    self.stop_price = trail

                # Check stop
                if self.data.High[-1] >= self.stop_price:
                    print(f"🛑 Moon Dev STOP hit at {self.stop_price:.2f} 🌙")
                    self.position.close()
                    self._reset()
                    return

                # Check target (POC proxy = EMA20)
                if self.data.Low[-1] <= self.target_price:
                    print(f"🎯 Moon Dev TARGET hit at {self.target_price:.2f} ✨")
                    self.position.close()
                    self._reset()
                    return

                # Invalidation: close back above spike high
                if self.data.Close[-1] > self.spike_high:
                    print(f"❌ Moon Dev INVALIDATION — close above spike high 🚫")
                    self.position.close()
                    self._reset()
                    return

                # Time stop or ADX trend resumption
                if bars_held >= self.time_stop_bars:
                    print(f"⏰ Moon Dev TIME STOP after {bars_held} bars 🌙")
                    self.position.close()
                    self._reset()
                    return

                if self.adx[-1] > self.adx_exit_threshold:
                    print(f"📈 Moon Dev ADX resumed trend ({self.adx[-1]:.1f}) — exit 🚀")
                    self.position.close()
                    self._reset()
                    return

            # Long side mirror
            elif self.position.is_long:
                self.highest_since_entry = max(self.highest_since_entry, self.data.High[-1])
                trail = self.highest_since_entry - self.atr_mult * self.atr[-1]
                if trail > self.stop_price:
                    self.stop_price = trail

                if self.data.Low[-1] <= self.stop_price:
                    print(f"🛑 Moon Dev STOP hit at {self.stop_price:.2f} 🌙")
                    self.position.close()
                    self._reset()
                    return

                if self.data.High[-1] >= self.target_price:
                    print(f"🎯 Moon Dev TARGET hit at {self.target_price:.2f} ✨")
                    self.position.close()
                    self._reset()
                    return

                if self.data.Close[-1] < self.spike_high:
                    print(f"❌ Moon Dev INVALIDATION — close below spike low 🚫")
                    self.position.close()
                    self._reset()
                    return

                if bars_held >= self.time_stop_bars:
                    print(f"⏰ Moon Dev TIME STOP after {bars_held} bars 🌙")
                    self.position.close()
                    self._reset()
                    return

                if self.adx[-1] > self.adx_exit_threshold:
                    print(f"📈 Moon Dev ADX resumed trend ({self.adx[-1]:.1f}) — exit 🚀")
                    self.position.close()
                    self._reset()
                    return
            return

        # Need enough bars
        if len(self.data) < 30:
            return

        # Guard against NaN indicators
        if (np.isnan(self.adx[-1]) or np.isnan(self.chaikin[-1]) or np.isnan(self.chaikin[-2])
                or np.isnan(self.atr[-1]) or np.isnan(self.rsi[-1]) or np.isnan(self.rsi[-2])
                or np.isnan(self.ema20[-1]) or np.isnan(self.std20[-1])):
            return

        # Regime filter: ADX < 20
        if self.adx[-1] >= self.adx_threshold:
            return

        chaikin_now = self.chaikin[-1]
        chaikin_prev = self.chaikin[-2]

        # Spike detection: close > EMA20 + 2*std (overextended)
        spike_up = self.data.Close[-1] > (self.ema20[-1] + self.spike_std_mult * self.std20[-1])
        spike_down = self.data.Close[-1] < (self.ema20[-1] - self.spike_std_mult * self.std20[-1])

        # SHORT entry: Chaikin crosses below 0, price spiked up, RSI > 65 rolling over
        if (chaikin_prev >= 0 and chaikin_now < 0 and spike_up
                and self.rsi[-1] > self.rsi_threshold and self.rsi[-1] < self.rsi[-2]):
            atr_val = self.atr[-1]
            if atr_val <= 0 or np.isnan(atr_val):
                return
            stop_dist = self.atr_mult * atr_val
            if stop_dist <= 0:
                return
            # Convert to fraction of equity for backtesting.py sizing
            risk_amount = self.equity * self.risk_pct
            position_size = risk_amount / stop_dist
            position_frac = (position_size * price) / self.equity
            # Clamp to valid fraction range for backtesting.py
            if position_frac <= 0:
                return
            if position_frac >= 1:
                position_frac = 0.99
            position_frac = round(position_frac, 4)
            if position_frac <= 0:
                return

            self.spike_high = self.data.High[-1]
            self.stop_price = self.spike_high + stop_dist
            self.target_price = self.ema20[-1]
            self.entry_bar = len(self.data)
            self.highest_since_entry = self.data.Low[-1]

            print(f"🌙 SHORT ExhaustionFade @ {price:.2f} | ADX={self.adx[-1]:.1f} | "
                  f"Chaikin={chaikin_now:.2f} | RSI={self.rsi[-1]:.1f} | size={position_frac:.4f} 🚀")
            self.sell(size=position_frac)
            return

        # LONG entry (mirror): Chaikin crosses above 0, price spiked down, RSI < 35 rolling up
        if (chaikin_prev <= 0 and chaikin_now > 0 and spike_down
                and self.rsi[-1] < 35 and self.rsi[-1] > self.rsi[-2]):
            atr_val = self.atr[-1]
            if atr_val <= 0 or np.isnan(atr_val):
                return
            stop_dist = self.atr_mult * atr_val
            if stop_dist <= 0:
                return
            risk_amount = self.equity * self.risk_pct
            position_size = risk_amount / stop_dist
            position_frac = (position_size * price) / self.equity
            # Clamp to valid fraction range for backtesting.py
            if position_frac <= 0:
                return
            if position_frac >= 1:
                position_frac = 0.99
            position_frac = round(position_frac, 4)
            if position_frac <= 0:
                return

            self.spike_high = self.data.Low[-1]
            self.stop_price = self.spike_high - stop_dist
            self.target_price = self.ema20[-1]
            self.entry_bar = len(self.data)
            self.highest_since_entry = self.data.High[-1]

            print(f"🌙 LONG ExhaustionFade @ {price:.2f} | ADX={self.adx[-1]:.1f} | "
                  f"Chaikin={chaikin_now:.2f} | RSI={self.rsi[-1]:.1f} | size={position_frac:.4f} 🚀")
            self.buy(size=position_frac)
            return

    def _reset(self):
        self.entry_bar = None
        self.highest_since_entry = None
        self.stop_price = None
        self.target_price = None
        self.spike_high = None


print("🌙✨ Launching Moon Dev ExhaustionFade Backtest 🚀🌙")
bt = Backtest(data, ExhaustionFade, cash=1_000_000, commission=0.0002)
stats = bt.run()
print(stats)
print(stats._strategy)