import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Load and clean data
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data.columns = [col.capitalize() for col in data.columns]
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
data = data.dropna()

print("🌙✨ Moon Dev KineticRupture backtest initiated! 🚀")
print(f"📊 Data loaded: {len(data)} bars")


class KineticRupture(Strategy):
    # Strategy parameters
    bb_period = 20
    bb_std = 2.0
    atr_period = 14
    atr_ma_period = 20
    vol_ma_period = 20
    ema_period = 200
    rsi_period = 14
    atr_stop_mult = 1.5
    risk_reward = 2.0
    risk_pct = 0.02
    position_size = 1_000_000

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        # Bollinger Bands
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS, close, timeperiod=self.bb_period,
            nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0
        )

        # Bollinger Band Width (volatility squeeze detection)
        self.bb_width = self.I(
            lambda u, l, m: (u - l) / m, self.bb_upper, self.bb_lower, self.bb_middle
        )
        self.bb_width_ma = self.I(talib.SMA, self.bb_width, timeperiod=20)

        # ATR and its MA
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)
        self.atr_ma = self.I(talib.SMA, self.atr, timeperiod=self.atr_ma_period)

        # Volume MA
        self.vol_ma = self.I(talib.SMA, volume, timeperiod=self.vol_ma_period)

        # Trend filter
        self.ema200 = self.I(talib.EMA, close, timeperiod=self.ema_period)

        # Momentum
        self.rsi = self.I(talib.RSI, close, timeperiod=self.rsi_period)

        print("🌙 Indicators initialized: BB, ATR, Volume MA, EMA200, RSI ✨")

    def next(self):
        price = self.data.Close[-1]

        # Skip if indicators not ready
        if (np.isnan(self.bb_upper[-1]) or np.isnan(self.atr_ma[-1]) or
                np.isnan(self.ema200[-1]) or np.isnan(self.bb_width_ma[-1])):
            return

        # Volatility squeeze condition
        squeeze = self.bb_width[-1] < self.bb_width_ma[-1]
        # ATR expansion
        atr_expansion = self.atr[-1] > self.atr_ma[-1]
        # Volume spike
        volume_spike = self.data.Volume[-1] > self.vol_ma[-1] * 1.2

        # Breakout conditions
        long_breakout = price > self.bb_upper[-1]
        short_breakout = price < self.bb_lower[-1]

        # Trend filter
        uptrend = price > self.ema200[-1]
        downtrend = price < self.ema200[-1]

        # Momentum confirmation
        long_momentum = self.rsi[-1] > 50
        short_momentum = self.rsi[-1] < 50

        # Manage existing position
        if self.position:
            # Check stop loss / take profit
            for trade in self.trades:
                if trade.is_long:
                    # Chandelier-style trailing / ATR stop
                    stop = trade.entry_price - self.atr[-1] * self.atr_stop_mult
                    target = trade.entry_price + self.atr[-1] * self.atr_stop_mult * self.risk_reward
                    if price <= stop:
                        print(f"🌙💥 LONG STOP hit at {price:.2f} | entry {trade.entry_price:.2f}")
                        trade.close()
                    elif price >= target:
                        print(f"🚀✨ LONG TP hit at {price:.2f} | entry {trade.entry_price:.2f}")
                        trade.close()
                    elif price < self.bb_middle[-1]:
                        print(f"🌙⚠️ LONG time-exit (re-entered zone) at {price:.2f}")
                        trade.close()
                else:
                    stop = trade.entry_price + self.atr[-1] * self.atr_stop_mult
                    target = trade.entry_price - self.atr[-1] * self.atr_stop_mult * self.risk_reward
                    if price >= stop:
                        print(f"🌙💥 SHORT STOP hit at {price:.2f} | entry {trade.entry_price:.2f}")
                        trade.close()
                    elif price <= target:
                        print(f"🚀✨ SHORT TP hit at {price:.2f} | entry {trade.entry_price:.2f}")
                        trade.close()
                    elif price > self.bb_middle[-1]:
                        print(f"🌙⚠️ SHORT time-exit (re-entered zone) at {price:.2f}")
                        trade.close()
            return

        # Long entry
        if (squeeze and atr_expansion and volume_spike and
                long_breakout and uptrend and long_momentum):
            stop_distance = self.atr[-1] * self.atr_stop_mult
            if stop_distance > 0:
                size = int(round(self.position_size))
                print(f"🌙🚀 LONG ENTRY | price {price:.2f} | ATR {self.atr[-1]:.2f} | size {size}")
                self.buy(size=size)

        # Short entry
        elif (squeeze and atr_expansion and volume_spike and
              short_breakout and downtrend and short_momentum):
            stop_distance = self.atr[-1] * self.atr_stop_mult
            if stop_distance > 0:
                size = int(round(self.position_size))
                print(f"🌙🔻 SHORT ENTRY | price {price:.2f} | ATR {self.atr[-1]:.2f} | size {size}")
                self.sell(size=size)


bt = Backtest(data, KineticRupture, cash=1_000_000, commission=0.001, exclusive_orders=True)
stats = bt.run()
print(stats)
print(stats._strategy)