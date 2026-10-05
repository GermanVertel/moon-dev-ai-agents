import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

print("🌙 Moon Dev OscillatorSurge Backtest starting... ✨🚀")

data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper column mapping
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

print(f"🌙 Data loaded: {len(data)} bars ✨")
print(f"🚀 Columns: {list(data.columns)}")


class OscillatorSurge(Strategy):
    rsi_period = 14
    rsi_oversold = 20
    atr_period = 14
    atr_tp_mult = 2.0
    atr_sl_mult = 1.5
    hist_multiplier = 1.5
    avg_period = 20
    time_exit_bars = 15
    risk_pct = 0.01
    use_trend_filter = False

    def init(self):
        print("🌙 Initializing OscillatorSurge indicators... ✨")
        close = self.data.Close
        high = self.data.High
        low = self.data.Low

        self.rsi = self.I(talib.RSI, close, timeperiod=self.rsi_period)

        # Compute MACD histogram via talib and wrap arrays properly
        macd, macdsignal, macdhist = talib.MACD(
            np.asarray(close, dtype=float),
            fastperiod=12, slowperiod=26, signalperiod=9
        )
        self.macd_hist = self.I(lambda: macdhist, name='MACD_Hist')

        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        # Precompute histogram delta and rolling average of absolute delta
        hist = pd.Series(macdhist)
        hist_delta = hist.diff()
        abs_delta = hist_delta.abs()
        avg_abs_delta = abs_delta.rolling(self.avg_period).mean()

        self.hist_delta = self.I(lambda: hist_delta.values, name='HistDelta')
        self.avg_abs_delta = self.I(lambda: avg_abs_delta.values, name='AvgAbsDelta')

        self.ema200 = self.I(talib.EMA, close, timeperiod=200)
        print("🚀 Indicators ready!")

    def next(self):
        price = self.data.Close[-1]

        if len(self.data) < 210:
            return

        # Manage open position
        if self.position:
            entry = self.position.entry_price
            atr_val = self.atr[-1]
            tp = entry + self.atr_tp_mult * atr_val
            sl = entry - self.atr_sl_mult * atr_val

            # Time-based exit
            bars_held = len(self.data) - self.position.entry_bar
            if bars_held >= self.time_exit_bars:
                print(f"⏰ Time exit after {bars_held} bars at {price:.2f} 🌙")
                self.position.close()
                return

            # RSI exit
            if self.rsi[-1] > 65:
                print(f"🎯 RSI exit at {price:.2f} (RSI={self.rsi[-1]:.1f}) ✨")
                self.position.close()
                return

            # MACD hist turns negative
            if self.macd_hist[-1] < 0:
                print(f"📉 MACD hist negative exit at {price:.2f} 🌙")
                self.position.close()
                return

            if price >= tp:
                print(f"💰 TP hit at {price:.2f} (target {tp:.2f}) 🚀")
                self.position.close()
                return
            if price <= sl:
                print(f"🛑 SL hit at {price:.2f} (stop {sl:.2f}) 🌙")
                self.position.close()
                return
            return

        # Entry logic
        rsi_val = self.rsi[-1]
        hist_delta = self.hist_delta[-1]
        avg_abs = self.avg_abs_delta[-1]
        atr_val = self.atr[-1]

        if np.isnan(rsi_val) or np.isnan(hist_delta) or np.isnan(avg_abs) or np.isnan(atr_val):
            return
        if avg_abs <= 0:
            return

        oversold = rsi_val < self.rsi_oversold
        accelerating = (self.macd_hist[-1] > self.macd_hist[-2]) and (hist_delta > self.hist_multiplier * avg_abs)
        trend_ok = (not self.use_trend_filter) or (price > self.ema200[-1])

        if oversold and accelerating and trend_ok:
            sl_distance = self.atr_sl_mult * atr_val
            if sl_distance <= 0:
                return
            risk_amount = self.equity * self.risk_pct
            size = int(round(risk_amount / sl_distance))
            if size < 1:
                size = 1
            # Cap by equity
            max_size = int(self.equity / price)
            if size > max_size:
                size = max_size
            if size < 1:
                return

            print(f"🌙🚀 ENTRY at {price:.2f} | RSI={rsi_val:.1f} | HistDelta={hist_delta:.4f} > {self.hist_multiplier}×{avg_abs:.4f} | Size={size}")
            self.buy(size=size)


bt = Backtest(data, OscillatorSurge, cash=1_000_000, commission=0.001)
print("🌙 Running backtest... ✨")
stats = bt.run()
print(stats)
print(stats._strategy)
print("🚀 Moon Dev backtest complete! 🌙✨")