import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

print("🌙✨ Moon Dev's VolatilityContraction Backtest Initializing... 🚀")

# Load data
data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Rename columns properly
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

# Set datetime as index
if 'datetime' in data.columns:
    data = data.set_index(pd.to_datetime(data['datetime']))
    data = data.drop(columns=['datetime'])

# Required columns
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

# Ensure all numeric columns are float64 (talib requires double arrays)
for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    data[col] = data[col].astype(np.float64)

print(f"🌙 Data loaded: {len(data)} bars ✨")
print(f"🚀 Columns: {list(data.columns)}")


class VolatilityContraction(Strategy):
    bb_period = 20
    bb_std = 2.0
    rsi_period = 14
    atr_period = 14
    atr_mult = 1.5
    max_bars = 10
    risk_pct = 0.01

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        vol = self.data.Volume

        # Bollinger Bands
        self.bb_upper, self.bb_mid, self.bb_lower = self.I(
            talib.BBANDS, close, timeperiod=self.bb_period,
            nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0
        )
        # RSI
        self.rsi = self.I(talib.RSI, close, timeperiod=self.rsi_period)
        # ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)
        # Volume SMA for reference
        self.vol_sma = self.I(talib.SMA, vol, timeperiod=10)

        self.entry_bar = None
        self.entry_price = None
        self.stop_price = None
        self.target_price = None
        self.be_moved = False

        print("🌙✨ Indicators ready! 🚀")

    def next(self):
        price = self.data.Close[-1]
        vol = self.data.Volume[-1]
        prev_vol = self.data.Volume[-2]

        upper = self.bb_upper[-1]
        mid = self.bb_mid[-1]
        rsi = self.rsi[-1]
        atr = self.atr[-1]

        # === Manage open position ===
        if self.position:
            bars_held = len(self.data) - 1 - self.entry_bar

            # Move stop to breakeven once 1x ATR profit reached
            if not self.be_moved and self.entry_price and atr:
                if price >= self.entry_price + atr:
                    self.stop_price = max(self.stop_price, self.entry_price)
                    self.be_moved = True
                    print(f"🌙 Breakeven stop moved! Entry={self.entry_price:.2f} 🚀")

            # Profit target hit
            if self.target_price and price >= self.target_price:
                print(f"✨ Profit target hit @ {price:.2f} (target {self.target_price:.2f}) 🌙")
                self.position.close()
                return

            # RSI exit (bearish crossover of 70 level)
            if self.rsi[-2] >= 70 and rsi < 70:
                print(f"🌙 RSI momentum fade ({rsi:.2f}) — exiting @ {price:.2f} ✨")
                self.position.close()
                return

            # Fixed bars reassess
            if bars_held >= self.max_bars:
                print(f"🚀 Max bars ({self.max_bars}) reached — exiting @ {price:.2f}")
                self.position.close()
                return

            return

        # === Entry logic ===
        if len(self.data) < self.bb_period + 2:
            return

        # Breakout above upper band
        breakout = price > upper
        # Declining volume
        declining_vol = vol < prev_vol
        # RSI rising but not overbought
        rsi_rising = rsi > self.rsi[-2]
        rsi_room = rsi < 70

        if breakout and declining_vol and rsi_rising and rsi_room:
            # Risk-based position size using 1x ATR stop distance
            if atr <= 0:
                return

            entry = price
            stop = min(mid, entry - atr)
            risk_per_unit = entry - stop

            if risk_per_unit <= 0:
                return

            # Fraction-based sizing (percentage of equity) per backtesting rules
            size = 0.95

            self.entry_bar = len(self.data) - 1
            self.entry_price = entry
            self.stop_price = stop
            self.target_price = entry + self.atr_mult * atr
            self.be_moved = False

            print(f"🌙✨ QUIET BREAKOUT! Entry={entry:.2f} | Stop={stop:.2f} | Target={self.target_price:.2f} | RSI={rsi:.2f} | ATR={atr:.2f} 🚀")
            self.buy(size=size, sl=stop, tp=self.target_price)


# Run backtest
bt = Backtest(
    data,
    VolatilityContraction,
    cash=1_000_000,
    commission=0.002
)

stats = bt.run()
print(stats)
print(stats._strategy)