import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Load data
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper column mapping
data = data.rename(columns={
    'datetime': 'Date',
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

data['Date'] = pd.to_datetime(data['Date'])
data = data.set_index('Date')

print("🌙✨ Moon Dev Data Loaded! Rows:", len(data))
print(data.head())


class DivergentSqueeze(Strategy):
    bb_period = 20
    bb_std = 2
    rsi_period = 14
    rsi_oversold = 30
    atr_period = 14
    atr_mult = 2.0
    tp_mult = 1.5
    risk_pct = 0.02
    max_bars_held = 5
    sma_slope_period = 5

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low

        # Bollinger Bands
        self.bb_mid = self.I(talib.SMA, close, timeperiod=self.bb_period)
        self.bb_stddev = self.I(talib.STDDEV, close, timeperiod=self.bb_period, nbdev=1)
        self.bb_upper = self.I(lambda c, m, s: m + self.bb_std * s,
                               close, self.bb_mid, self.bb_stddev)
        self.bb_lower = self.I(lambda c, m, s: m - self.bb_std * s,
                               close, self.bb_mid, self.bb_stddev)

        # RSI
        self.rsi = self.I(talib.RSI, close, timeperiod=self.rsi_period)

        # ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        # 3-bar average return (per-bar pct change averaged over 3 bars)
        close_series = pd.Series(close)
        self.avg_ret3 = self.I(lambda c: pd.Series(c).pct_change().rolling(3).mean().values, close)

        # SMA slope filter
        self.sma_slope = self.I(lambda m: pd.Series(m).diff(self.sma_slope_period).values, self.bb_mid)

        self.bar_count = 0
        self.stop_price = None
        self.tp_price = None

    def next(self):
        price = self.data.Close[-1]

        # Manage open position
        if self.position:
            self.bar_count += 1

            # Time-based exit
            if self.bar_count >= self.max_bars_held:
                print(f"⏰ Moon Dev Time Exit at {price:.2f}")
                self.position.close()
                self.bar_count = 0
                return

            # Exit on negative AvgRet3
            if not np.isnan(self.avg_ret3[-1]) and self.avg_ret3[-1] < 0:
                print(f"📉 Moon Dev AvgRet3 Negative Exit at {price:.2f}")
                self.position.close()
                self.bar_count = 0
                return

            # Stop loss / take profit
            if self.stop_price and self.data.Low[-1] <= self.stop_price:
                print(f"🛑 Moon Dev Stop Loss Hit at {self.stop_price:.2f}")
                self.position.close()
                self.bar_count = 0
                return

            if self.tp_price and self.data.High[-1] >= self.tp_price:
                print(f"🎯 Moon Dev Take Profit Hit at {self.tp_price:.2f}")
                self.position.close()
                self.bar_count = 0
                return
            return

        # Entry logic
        if (not np.isnan(self.bb_upper[-1]) and
                not np.isnan(self.rsi[-1]) and
                not np.isnan(self.atr[-1]) and
                not np.isnan(self.sma_slope[-1])):

            upper_breach = price > self.bb_upper[-1]
            rsi_oversold = self.rsi[-1] < self.rsi_oversold
            trend_up = self.sma_slope[-1] > 0

            if upper_breach and rsi_oversold and trend_up:
                atr_val = self.atr[-1]
                stop_dist = self.atr_mult * atr_val
                if stop_dist <= 0:
                    return

                # Fractional position sizing (percentage of equity)
                risk_amount = self.equity * self.risk_pct
                units = risk_amount / stop_dist
                size_fraction = units * price / self.equity

                if size_fraction <= 0:
                    return
                if size_fraction >= 1:
                    size_fraction = 0.99

                self.stop_price = price - stop_dist
                self.tp_price = price + self.tp_mult * stop_dist
                self.bar_count = 0
                print(f"🚀🌙 Moon Dev LONG Entry! Price={price:.2f} RSI={self.rsi[-1]:.2f} "
                      f"UpperBB={self.bb_upper[-1]:.2f} Size={size_fraction:.4f} "
                      f"Stop={self.stop_price:.2f} TP={self.tp_price:.2f}")
                self.buy(size=size_fraction)


bt = Backtest(data, DivergentSqueeze, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)