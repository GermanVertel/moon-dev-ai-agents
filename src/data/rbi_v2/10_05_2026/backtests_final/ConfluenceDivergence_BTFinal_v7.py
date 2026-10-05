import numpy as np
import pandas as pd
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
for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    data[col] = data[col].astype(np.float64)
print("🌙✨ Moon Dev: Data loaded and cleaned! Rows:", len(data))


class ConfluenceDivergence(Strategy):
    rsi_period = 14
    rsi_oversold = 30
    rsi_lookback = 10
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    vol_sma_period = 20
    vol_mult = 1.2
    bb_period = 20
    bb_std = 2.0
    swing_lookback = 20
    stop_pct = 0.02
    tp_pct = 0.02
    time_stop = 20
    ema_trend_period = 200

    def init(self):
        print("🌙 Moon Dev: Initializing ConfluenceDivergence strategy...")
        close = np.asarray(self.data.Close, dtype=np.float64)
        high = np.asarray(self.data.High, dtype=np.float64)
        low = np.asarray(self.data.Low, dtype=np.float64)
        volume = np.asarray(self.data.Volume, dtype=np.float64)

        def _rsi(arr):
            return talib.RSI(np.asarray(arr, dtype=np.float64), timeperiod=self.rsi_period)

        def _macd(arr):
            a = np.asarray(arr, dtype=np.float64)
            macd, signal, hist = talib.MACD(
                a,
                fastperiod=self.macd_fast,
                slowperiod=self.macd_slow,
                signalperiod=self.macd_signal,
            )
            return macd

        def _macd_signal(arr):
            a = np.asarray(arr, dtype=np.float64)
            macd, signal, hist = talib.MACD(
                a,
                fastperiod=self.macd_fast,
                slowperiod=self.macd_slow,
                signalperiod=self.macd_signal,
            )
            return signal

        def _macd_hist(arr):
            a = np.asarray(arr, dtype=np.float64).ravel()
            macd, signal, hist = talib.MACD(
                a,
                fastperiod=self.macd_fast,
                slowperiod=self.macd_slow,
                signalperiod=self.macd_signal,
            )
            return hist

        def _sma_vol(arr):
            return talib.SMA(np.asarray(arr, dtype=np.float64).ravel(), timeperiod=self.vol_sma_period)

        def _bb_upper(arr):
            a = np.asarray(arr, dtype=np.float64).ravel()
            upper, mid, lower = talib.BBANDS(
                a,
                timeperiod=self.bb_period,
                nbdevup=self.bb_std,
                nbdevdn=self.bb_std,
            )
            return upper

        def _bb_mid(arr):
            a = np.asarray(arr, dtype=np.float64).ravel()
            upper, mid, lower = talib.BBANDS(
                a,
                timeperiod=self.bb_period,
                nbdevup=self.bb_std,
                nbdevdn=self.bb_std,
            )
            return mid

        def _bb_lower(arr):
            a = np.asarray(arr, dtype=np.float64).ravel()
            upper, mid, lower = talib.BBANDS(
                a,
                timeperiod=self.bb_period,
                nbdevup=self.bb_std,
                nbdevdn=self.bb_std,
            )
            return lower

        def _min_low(arr):
            return talib.MIN(np.asarray(arr, dtype=np.float64).ravel(), timeperiod=self.swing_lookback)

        def _ema_trend(arr):
            return talib.EMA(np.asarray(arr, dtype=np.float64).ravel(), timeperiod=self.ema_trend_period)

        self.rsi = self.I(_rsi, close)
        self.macd = self.I(_macd, close)
        self.macd_signal = self.I(_macd_signal, close)
        self.macd_hist = self.I(_macd_hist, close)
        self.vol_sma = self.I(_sma_vol, volume)
        self.bb_upper = self.I(_bb_upper, close)
        self.bb_mid = self.I(_bb_mid, close)
        self.bb_lower = self.I(_bb_lower, close)
        self.swing_low = self.I(_min_low, low)
        self.ema_trend = self.I(_ema_trend, close)

        print("🌙✨ Moon Dev: All indicators initialized! 🚀")

    def next(self):
        price = self.data.Close[-1]

        # Manage existing position
        if self.position:
            try:
                entry = self.trades[-1].entry_price
            except Exception:
                entry = price

            if price >= entry * (1 + self.tp_pct):
                print(f"🌙💰 Moon Dev: TP HIT! Closing at {price:.2f} (+2% from {entry:.2f})")
                self.position.close()
                return
            if not np.isnan(self.bb_upper[-1]) and price >= self.bb_upper[-1]:
                print(f"🌙🎯 Moon Dev: BB UPPER BAND HIT! Closing at {price:.2f} (band {self.bb_upper[-1]:.2f})")
                self.position.close()
                return
            if len(self.trades) > 0 and (len(self.data) - self.trades[-1].entry_bar) >= self.time_stop:
                print(f"🌙⏰ Moon Dev: TIME STOP! Closing at {price:.2f}")
                self.position.close()
                return
            return

        # Need enough bars
        min_bars = max(self.bb_period, self.swing_lookback, self.ema_trend_period) + 5
        if len(self.data) < min_bars:
            return

        # Guard against NaN indicators
        if (np.isnan(self.rsi[-1]) or np.isnan(self.macd_hist[-1]) or
                np.isnan(self.vol_sma[-1]) or np.isnan(self.ema_trend[-1]) or
                np.isnan(self.swing_low[-1]) or np.isnan(self.bb_upper[-1])):
            return

        # Trend filter: skip if strongly bearish (price well below EMA)
        if price < self.ema_trend[-1] * 0.95:
            return

        # Oversold context: RSI was below 30 in last N bars
        recent_rsi = self.rsi[-self.rsi_lookback:]
        recent_rsi = recent_rsi[~np.isnan(recent_rsi)]
        if len(recent_rsi) == 0 or not np.any(recent_rsi < self.rsi_oversold):
            return

        # Bullish RSI divergence: price lower low, RSI higher low
        curr_low = self.data.Low[-1]
        prev_low = self.data.Low[-self.swing_lookback]
        curr_rsi = self.rsi[-1]
        prev_rsi = self.rsi[-self.swing_lookback]

        if np.isnan(prev_rsi):
            return

        rsi_div = (curr_low <= prev_low) and (curr_rsi > prev_rsi) and (curr_rsi < 50)

        curr_macd = self.macd_hist[-1]
        prev_macd = self.macd_hist[-self.swing_lookback]

        if np.isnan(prev_macd):
            return

        macd_div = (curr_low <= prev_low) and (curr_macd > prev_macd)

        vol_ok = self.data.Volume[-1] > self.vol_sma[-1] * self.vol_mult

        if rsi_div and macd_div and vol_ok:
            swing_stop = self.swing_low[-1]
            fixed_stop = price * (1 - self.stop_pct)
            stop_price = max(swing_stop, fixed_stop)

            risk = price - stop_price
            if risk <= 0:
                return

            risk_amount = self.equity * 0.01
            position_size_units = risk_amount / risk
            position_size_frac = position_size_units * price / self.equity
            position_size_frac = min(position_size_frac, 0.95)
            if position_size_frac <= 0 or position_size_frac >= 1:
                return

            print(f"🌙🚀 Moon Dev: CONFLUENCE SIGNAL! RSI_div={rsi_div} MACD_div={macd_div} Vol_ok={vol_ok}")
            print(f"🌙✨ Entry={price:.2f} Stop={stop_price:.2f} Size={position_size_frac:.4f}")
            self.buy(size=position_size_frac, sl=stop_price)


bt = Backtest(data, ConfluenceDivergence, cash=1_000_000, commission=0.001)
print("🌙🚀 Moon Dev: Running backtest...")
stats = bt.run()
print(stats)
print(stats._strategy)