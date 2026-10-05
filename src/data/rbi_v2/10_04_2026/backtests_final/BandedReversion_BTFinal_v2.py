import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# 🌙 Moon Dev's BandedReversion Backtest 🚀

data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"

data = pd.read_csv(data_path)
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
data['Datetime'] = pd.to_datetime(data['Datetime'])
data = data.set_index('Datetime')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

print("🌙✨ Moon Dev data loaded! Shape:", data.shape)
print("🚀 Columns:", list(data.columns))


class BandedReversion(Strategy):
    bb_period = 20
    bb_std = 2.0
    atr_period = 14
    atr_mult = 2.0
    rsi_period = 14
    rsi_threshold = 30
    time_stop = 18
    risk_pct = 0.02

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low

        self.sma = self.I(talib.SMA, close, timeperiod=self.bb_period)
        self.std = self.I(talib.STDDEV, close, timeperiod=self.bb_period, nbdev=1)
        self.upper = self.I(lambda s, st: s + self.bb_std * st, self.sma, self.std)
        self.lower = self.I(lambda s, st: s - self.bb_std * st, self.sma, self.std)
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)
        self.rsi = self.I(talib.RSI, close, timeperiod=self.rsi_period)

        self.entry_bar = None
        self.stop_price = None
        self.target_price = None

        print("🌙✨ BandedReversion indicators initialized! 🚀")

    def next(self):
        price = self.data.Close[-1]
        lower = self.lower[-1]
        sma = self.sma[-1]
        atr = self.atr[-1]
        rsi = self.rsi[-1]

        if np.isnan(lower) or np.isnan(sma) or np.isnan(atr) or np.isnan(rsi):
            return

        # 🌙 Manage open position
        if self.position:
            bars_held = len(self.data) - self.entry_bar

            # Trail stop to breakeven once price crosses SMA
            if price >= sma and self.stop_price < price:
                self.stop_price = max(self.stop_price, price)
                print(f"🌙✨ Trailing stop to breakeven! New stop: {self.stop_price:.2f} 🚀")

            # Primary target: reversion to SMA
            if price >= sma and self.position.is_long:
                print(f"🌙💰 Mean reversion complete! Closing at {price:.2f} (SMA: {sma:.2f}) ✨")
                self.position.close()
                return

            # Stop loss
            if price <= self.stop_price:
                print(f"🌙🛑 STOP LOSS hit at {price:.2f} (stop: {self.stop_price:.2f}) 💔")
                self.position.close()
                return

            # Time stop
            if bars_held >= self.time_stop:
                print(f"🌙⏰ TIME STOP after {bars_held} bars at {price:.2f} ⌛")
                self.position.close()
                return

            return

        # 🌙 Entry logic: Lower BB > SMA (extreme oversold anomaly)
        setup = lower > sma
        if not setup:
            return

        # Confirmation: close back above lower band
        confirmation = price > lower and self.data.Close[-2] <= self.lower[-2]
        if not confirmation:
            return

        # Optional filter: RSI < 30
        if rsi >= self.rsi_threshold:
            return

        # Volatility regime filter: skip if ATR expanding violently
        if len(self.atr) > 5 and atr > 3 * np.nanmean(self.atr[-5:]):
            print(f"🌙⚠️ ATR expanding violently ({atr:.2f}), skipping trade 🚫")
            return

        # Risk management: position size = risk_amount / (2 * ATR)
        risk_amount = self.equity * self.risk_pct
        stop_distance = self.atr_mult * atr
        if stop_distance <= 0:
            return

        position_size = int(round(risk_amount / stop_distance))
        if position_size <= 0:
            position_size = 1

        # Cap size to ensure sufficient cash (price * size <= equity)
        max_size = int(self.equity // price)
        if max_size <= 0:
            return
        if position_size > max_size:
            position_size = max_size

        # Convert to fraction of equity for backtesting.py sizing
        size_fraction = (position_size * price) / self.equity
        if size_fraction <= 0:
            return
        if size_fraction > 0.99:
            size_fraction = 0.99

        self.stop_price = price - stop_distance
        self.target_price = sma
        self.entry_bar = len(self.data)

        print(f"🌙🚀 ENTRY! Price: {price:.2f} | Lower BB: {lower:.2f} | SMA: {sma:.2f} | "
              f"RSI: {rsi:.1f} | ATR: {atr:.2f} | Size: {position_size} | Stop: {self.stop_price:.2f} ✨")

        self.buy(size=size_fraction)


bt = Backtest(data, BandedReversion, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)