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
    'datetime': 'Datetime',
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

data['Datetime'] = pd.to_datetime(data['Datetime'])
data = data.set_index('Datetime')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

print("🌙 CompressionTrigger Backtest Initializing... ✨")
print(f"📊 Data loaded: {len(data)} bars")
print(f"🚀 Date range: {data.index[0]} to {data.index[-1]}")


class CompressionTrigger(Strategy):
    bb_period = 20
    bb_std = 2.0
    squeeze_lookback = 20
    atr_period = 14
    atr_mult = 2.0
    breakout_window = 3
    risk_pct = 0.01
    time_stop_bars = 20

    def init(self):
        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)

        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(
            close.values, timeperiod=self.bb_period,
            nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0
        )
        self.bb_upper = self.I(lambda: upper, name='BB_Upper')
        self.bb_middle = self.I(lambda: middle, name='BB_Middle')
        self.bb_lower = self.I(lambda: lower, name='BB_Lower')

        # Bollinger Band Width
        bbw = (upper - lower) / middle
        self.bbw = self.I(lambda: bbw, name='BBW')

        # Min BBW over lookback
        bbw_series = pd.Series(bbw)
        min_bbw = bbw_series.rolling(self.squeeze_lookback).min().values
        self.min_bbw = self.I(lambda: min_bbw, name='Min_BBW')

        # ATR
        atr = talib.ATR(high.values, low.values, close.values, timeperiod=self.atr_period)
        self.atr = self.I(lambda: atr, name='ATR')

        # Track squeeze bars
        self.squeeze_bar = np.zeros(len(close), dtype=bool)
        for i in range(len(close)):
            if not np.isnan(bbw[i]) and not np.isnan(min_bbw[i]):
                if bbw[i] <= min_bbw[i]:
                    self.squeeze_bar[i] = True

        self.highest_close = 0.0
        self.trailing_stop = 0.0
        self.entry_bar = 0
        self.entry_price = 0.0

        print("🌙 Indicators initialized: BB, BBW, ATR ✨")

    def next(self):
        i = len(self.data) - 1

        if np.isnan(self.bb_upper[i]) or np.isnan(self.atr[i]) or np.isnan(self.bbw[i]):
            return

        price = self.data.Close[i]
        upper = self.bb_upper[i]
        atr = self.atr[i]

        # Manage open position
        if self.position:
            self.highest_close = max(self.highest_close, price)
            new_stop = self.highest_close - (self.atr_mult * atr)
            if new_stop > self.trailing_stop:
                self.trailing_stop = new_stop
                print(f"🌙 Trailing stop raised to {self.trailing_stop:.2f} ✨")

            bars_held = i - self.entry_bar

            # Exit on trailing stop hit
            if price < self.trailing_stop:
                print(f"🛑 EXIT: Price {price:.2f} < Trailing Stop {self.trailing_stop:.2f} 🌙")
                self.position.close()
                return

            # Time stop
            if bars_held >= self.time_stop_bars and price <= self.entry_price:
                print(f"⏰ TIME STOP: {bars_held} bars, no favorable move 🚀")
                self.position.close()
                return

            return

        # Check for squeeze within last N bars
        squeeze_recent = False
        for j in range(max(0, i - self.breakout_window), i + 1):
            if self.squeeze_bar[j]:
                squeeze_recent = True
                break

        if not squeeze_recent:
            return

        # Breakout confirmation
        if price > upper:
            # Risk-based position sizing
            stop_price = price - (self.atr_mult * atr)
            risk_per_unit = price - stop_price
            if risk_per_unit <= 0:
                return

            risk_amount = self.equity * self.risk_pct
            position_size = risk_amount / risk_per_unit
            position_size = int(round(position_size))

            if position_size < 1:
                position_size = 1

            self.highest_close = price
            self.trailing_stop = stop_price
            self.entry_bar = i
            self.entry_price = price

            print(f"🚀 ENTRY: Price {price:.2f} > Upper BB {upper:.2f} | Squeeze breakout! 🌙")
            print(f"✨ Size: {position_size} | Stop: {stop_price:.2f} | ATR: {atr:.2f}")
            self.buy(size=position_size)


bt = Backtest(data, CompressionTrigger, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)