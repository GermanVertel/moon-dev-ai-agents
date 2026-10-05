import pandas as pd
import numpy as np
import talib
from backtesting import Backtest, Strategy

# Load and clean data
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

# 🌙 Ensure numeric dtypes for talib (fixes "input array type is not double")
for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    data[col] = pd.to_numeric(data[col], errors='coerce')
data = data.dropna()

print("🌙✨ Moon Dev MomentumPulse Backtest Initializing... 🚀")
print(f"📊 Data loaded: {len(data)} bars from {data.index[0]} to {data.index[-1]}")


class MomentumPulse(Strategy):
    rsi_period = 4
    ema_fast = 5
    ema_slow = 10
    ema_trend = 20
    atr_period = 14
    vol_ma_period = 5
    risk_pct = 0.02
    max_hold_bars = 15
    swing_lookback = 10

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        # 🌙 Cast to float64 arrays for talib compatibility
        close_arr = np.asarray(close, dtype=np.float64)
        high_arr = np.asarray(high, dtype=np.float64)
        low_arr = np.asarray(low, dtype=np.float64)
        vol_arr = np.asarray(volume, dtype=np.float64)

        self.rsi = self.I(talib.RSI, close_arr, timeperiod=self.rsi_period)
        self.ema5 = self.I(talib.EMA, close_arr, timeperiod=self.ema_fast)
        self.ema10 = self.I(talib.EMA, close_arr, timeperiod=self.ema_slow)
        self.ema20 = self.I(talib.EMA, close_arr, timeperiod=self.ema_trend)
        self.atr = self.I(talib.ATR, high_arr, low_arr, close_arr, timeperiod=self.atr_period)
        self.vol_ma = self.I(talib.SMA, vol_arr, timeperiod=self.vol_ma_period)
        self.swing_low = self.I(talib.MIN, low_arr, timeperiod=self.swing_lookback)

        self.entry_bar = None
        self.entry_price = None
        self.stop_price = None
        self.trail_active = False

        print("🌙 Indicators warmed up: RSI, EMA5/10/20, ATR, VolMA, SwingLow ✨")

    def next(self):
        price = self.data.Close[-1]

        if self.position:
            bars_held = len(self.data) - self.entry_bar

            # Trailing stop via EMA5 after 1.5R
            if self.entry_price and self.stop_price:
                risk = self.entry_price - self.stop_price
                if risk > 0 and price >= self.entry_price + 1.5 * risk:
                    if not self.trail_active:
                        self.trail_active = True
                        print(f"🚀 Moon Dev: Trailing stop ACTIVATED at {price:.2f} ✨")

            if self.trail_active:
                new_stop = self.ema5[-1]
                if new_stop > self.stop_price:
                    self.stop_price = new_stop

            # Hard stop
            if price <= self.stop_price:
                self.position.close()
                print(f"🛑 Moon Dev: STOP LOSS hit at {price:.2f} 💥")
                self._reset()
                return

            # EMA cross exit with declining volume confirmation
            cross_down = self.ema5[-2] >= self.ema10[-2] and self.ema5[-1] < self.ema10[-1]
            vol_declining = self.data.Volume[-1] < self.vol_ma[-1]

            if cross_down and vol_declining:
                self.position.close()
                print(f"🌙 Moon Dev: EMA5<EMA10 + declining vol → EXIT at {price:.2f} 🎯")
                self._reset()
                return

            # Max hold
            if bars_held >= self.max_hold_bars:
                self.position.close()
                print(f"⏰ Moon Dev: MAX HOLD reached ({bars_held} bars) → EXIT at {price:.2f}")
                self._reset()
                return

        else:
            # Entry: RSI cross above 70 + price above EMA20
            rsi_cross = self.rsi[-2] <= 70 and self.rsi[-1] > 70
            trend_ok = price > self.ema20[-1]

            if rsi_cross and trend_ok:
                atr_val = self.atr[-1]
                swing = self.swing_low[-1]
                stop = min(swing, price - 2 * atr_val)
                risk_per_unit = price - stop
                if risk_per_unit <= 0:
                    return

                risk_amount = self.equity * self.risk_pct
                size = int(round(risk_amount / risk_per_unit))
                if size < 1:
                    size = 1

                self.buy(size=size)
                self.entry_bar = len(self.data)
                self.entry_price = price
                self.stop_price = stop
                self.trail_active = False
                print(f"🌙✨ Moon Dev ENTRY: RSI={self.rsi[-1]:.1f} crossed 70 | price={price:.2f} | size={size} | stop={stop:.2f} 🚀")

    def _reset(self):
        self.entry_bar = None
        self.entry_price = None
        self.stop_price = None
        self.trail_active = False


bt = Backtest(data, MomentumPulse, cash=1_000_000, commission=0.002, exclusive_orders=True)
stats = bt.run()
print(stats)
print(stats._strategy)