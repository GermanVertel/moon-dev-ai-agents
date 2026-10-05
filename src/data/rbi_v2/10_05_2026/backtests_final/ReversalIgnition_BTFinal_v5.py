import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Load and clean data
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume',
})
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
# 🌙 Ensure all OHLCV columns are float64 for talib compatibility
for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    data[col] = data[col].astype(float)
print("🌙✨ Moon Dev Data Loaded! Shape:", data.shape)
print(data.head())


class ReversalIgnition(Strategy):
    rsi_period = 14
    vol_sma_period = 20
    hh_period = 20
    atr_period = 14
    ema_trend_period = 200
    ema_trail_period = 10
    rsi_bull_threshold = 60
    vol_mult = 1.2
    risk_pct = 0.01
    rr_target = 2.0
    time_stop_bars = 10
    max_ema_extension = 0.03

    def init(self):
        # 🌙 Cast to float64 arrays for talib
        close = self.data.Close.astype(float)
        high = self.data.High.astype(float)
        low = self.data.Low.astype(float)
        volume = self.data.Volume.astype(float)

        self.rsi = self.I(talib.RSI, close, timeperiod=self.rsi_period)
        self.vol_sma = self.I(talib.SMA, volume, timeperiod=self.vol_sma_period)
        self.hh = self.I(talib.MAX, high, timeperiod=self.hh_period)
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)
        self.ema_trend = self.I(talib.EMA, close, timeperiod=self.ema_trend_period)
        self.ema_trail = self.I(talib.EMA, close, timeperiod=self.ema_trail_period)
        self.entry_bar = 0
        print("🚀 Moon Dev Indicators Initialized!")

    def next(self):
        price = self.data.Close[-1]

        # Skip if indicators not ready
        if (np.isnan(self.rsi[-1]) or np.isnan(self.vol_sma[-1]) or np.isnan(self.hh[-2])
                or np.isnan(self.atr[-1]) or np.isnan(self.ema_trend[-1])
                or np.isnan(self.ema_trail[-1])):
            return

        # Manage open position
        if self.position:
            bars_held = len(self.data) - self.entry_bar
            # Trailing exit: close below 10 EMA
            if self.data.Close[-1] < self.ema_trail[-1]:
                print(f"🌙 Trail exit at {price:.2f} (below 10 EMA {self.ema_trail[-1]:.2f})")
                self.position.close()
                return
            # Time stop
            if bars_held >= self.time_stop_bars:
                print(f"⏰ Time stop exit at {price:.2f} after {bars_held} bars")
                self.position.close()
                return
            return

        # Entry conditions
        breakout = self.data.Close[-1] > self.hh[-2]
        bullish_candle = self.data.Close[-1] > self.data.Open[-1]
        rsi_turning_down = self.rsi[-1] < self.rsi[-2]
        rsi_was_bullish = self.rsi[-2] >= self.rsi_bull_threshold
        vol_expansion = self.data.Volume[-1] > self.vol_mult * self.vol_sma[-1]
        vol_rising = self.data.Volume[-1] > self.data.Volume[-2]
        trend_ok = self.data.Close[-1] > self.ema_trend[-1]
        not_extended = self.data.Close[-1] <= self.ema_trail[-1] * (1 + self.max_ema_extension)

        # ATR volatility filter
        atr_window = self.atr[-100:]
        atr_window = atr_window[~np.isnan(atr_window)]
        if len(atr_window) < 20:
            return
        atr_min = np.min(atr_window)
        atr_max = np.max(atr_window)
        atr_range = atr_max - atr_min if atr_max > atr_min else 1e-9
        atr_pct = (self.atr[-1] - atr_min) / atr_range
        atr_ok = 0.10 <= atr_pct <= 0.95

        if (breakout and bullish_candle and rsi_turning_down and rsi_was_bullish
                and vol_expansion and vol_rising and trend_ok and not_extended and atr_ok):

            entry = price
            stop = entry - 1.5 * self.atr[-1]
            # tighten stop to breakout candle low if it gives >= 1 ATR room
            candle_low = self.data.Low[-1]
            if entry - candle_low >= self.atr[-1] and candle_low > stop:
                stop = candle_low

            risk = entry - stop
            if risk <= 0:
                return

            # 🌙 Position sizing: use fraction of equity so it always executes
            # Risk-based fraction (risk_pct of equity per unit risk)
            size_frac = self.risk_pct * entry / risk
            # Clamp to valid fraction (0, 1)
            size_frac = min(max(size_frac, 0.01), 0.99)

            tp = entry + self.rr_target * risk

            print(f"🚀🌙 REVERSAL IGNITION! Entry={entry:.2f} Stop={stop:.2f} TP={tp:.2f} SizeFrac={size_frac:.4f}")
            self.buy(size=size_frac, sl=stop, tp=tp)
            self.entry_bar = len(self.data)


bt = Backtest(data, ReversalIgnition, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)