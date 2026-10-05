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
# Ensure all OHLCV columns are float64 (talib requires double arrays)
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
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        # Wrap talib calls to enforce float64 double arrays (talib requires double)
        def _rsi(arr, timeperiod=14):
            return talib.RSI(np.asarray(arr, dtype=np.float64), timeperiod=timeperiod)

        def _macd(arr, fastperiod=12, slowperiod=26, signalperiod=9):
            return talib.MACD(
                np.asarray(arr, dtype=np.float64),
                fastperiod=fastperiod,
                slowperiod=slowperiod,
                signalperiod=signalperiod,
            )

        def _sma(arr, timeperiod=20):
            return talib.SMA(np.asarray(arr, dtype=np.float64), timeperiod=timeperiod)

        def _bbands(arr, timeperiod=20, nbdevup=2.0, nbdevdn=2.0):
            return talib.BBANDS(
                np.asarray(arr, dtype=np.float64),
                timeperiod=timeperiod,
                nbdevup=nbdevup,
                nbdevdn=nbdevdn,
            )

        def _min(arr, timeperiod=20):
            return talib.MIN(np.asarray(arr, dtype=np.float64), timeperiod=timeperiod)

        def _ema(arr, timeperiod=200):
            return talib.EMA(np.asarray(arr, dtype=np.float64), timeperiod=timeperiod)

        # RSI
        self.rsi = self.I(_rsi, close, timeperiod=self.rsi_period)

        # MACD
        self.macd, self.macd_signal, self.macd_hist = self.I(
            _macd, close,
            fastperiod=self.macd_fast,
            slowperiod=self.macd_slow,
            signalperiod=self.macd_signal
        )

        # Volume SMA
        self.vol_sma = self.I(_sma, volume, timeperiod=self.vol_sma_period)

        # Bollinger Bands
        self.bb_upper, self.bb_mid, self.bb_lower = self.I(
            _bbands, close,
            timeperiod=self.bb_period,
            nbdevup=self.bb_std,
            nbdevdn=self.bb_std
        )

        # Swing low
        self.swing_low = self.I(_min, low, timeperiod=self.swing_lookback)

        # Trend filter EMA
        self.ema_trend = self.I(_ema, close, timeperiod=self.ema_trend_period)

        print("🌙✨ Moon Dev: All indicators initialized! 🚀")

    def next(self):
        price = self.data.Close[-1]

        # Manage existing position
        if self.position:
            entry = self.trades[-1].entry_price
            # Take profit at +2%
            if price >= entry * (1 + self.tp_pct):
                print(f"🌙💰 Moon Dev: TP HIT! Closing at {price:.2f} (+2% from {entry:.2f})")
                self.position.close()
                return
            # Upper Bollinger Band exit
            if price >= self.bb_upper[-1]:
                print(f"🌙🎯 Moon Dev: BB UPPER BAND HIT! Closing at {price:.2f} (band {self.bb_upper[-1]:.2f})")
                self.position.close()
                return
            # Time stop
            if len(self.trades) > 0 and (len(self.data) - self.trades[-1].entry_bar) >= self.time_stop:
                print(f"🌙⏰ Moon Dev: TIME STOP! Closing at {price:.2f}")
                self.position.close()
                return
            return

        # Need enough bars
        if len(self.data) < max(self.bb_period, self.swing_lookback, self.ema_trend_period) + 5:
            return

        # Trend filter: skip if strongly bearish (price well below EMA)
        if price < self.ema_trend[-1] * 0.95:
            return

        # Oversold context: RSI was below 30 in last N bars
        recent_rsi = self.rsi[-self.rsi_lookback:]
        if not np.any(recent_rsi < self.rsi_oversold):
            return

        # Bullish RSI divergence: price lower low, RSI higher low
        curr_low = self.data.Low[-1]
        prev_low = self.data.Low[-self.swing_lookback]
        curr_rsi = self.rsi[-1]
        prev_rsi = self.rsi[-self.swing_lookback]

        rsi_div = (curr_low <= prev_low) and (curr_rsi > prev_rsi) and (curr_rsi < 50)

        # MACD bullish divergence
        curr_macd = self.macd_hist[-1]
        prev_macd = self.macd_hist[-self.swing_lookback]
        macd_div = (curr_low <= prev_low) and (curr_macd > prev_macd)

        # Volume confirmation
        vol_ok = self.data.Volume[-1] > self.vol_sma[-1] * self.vol_mult

        if rsi_div and macd_div and vol_ok:
            # Stop loss below recent swing low, or fixed 2%, whichever tighter
            swing_stop = self.swing_low[-1]
            fixed_stop = price * (1 - self.stop_pct)
            stop_price = max(swing_stop, fixed_stop)  # tighter = higher stop

            risk = price - stop_price
            if risk <= 0:
                return

            # Position sizing: risk 1% of equity, expressed as fraction of equity
            # (backtesting.py requires size as fraction (0<size<1) or whole units)
            risk_amount = self.equity * 0.01
            position_size_units = risk_amount / risk
            # Convert units to fraction of equity
            position_size_frac = position_size_units * price / self.equity
            # Cap fraction at 0.95 to avoid over-leverage
            position_size_frac = min(position_size_frac, 0.95)
            if position_size_frac <= 0:
                return

            print(f"🌙🚀 Moon Dev: CONFLUENCE SIGNAL! RSI_div={rsi_div} MACD_div={macd_div} Vol_ok={vol_ok}")
            print(f"🌙✨ Entry={price:.2f} Stop={stop_price:.2f} Size={position_size_frac:.4f}")
            self.buy(size=position_size_frac, sl=stop_price)


# Run backtest
bt = Backtest(data, ConfluenceDivergence, cash=1_000_000, commission=0.001)
print("🌙🚀 Moon Dev: Running backtest...")
stats = bt.run()
print(stats)
print(stats._strategy)