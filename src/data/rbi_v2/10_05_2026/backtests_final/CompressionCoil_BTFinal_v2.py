import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

print("🌙✨ CompressionCoil Strategy Loading... Let's ride the volatility expansion! 🚀")

# Load data
data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean column names
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

# Ensure datetime index
if 'datetime' in data.columns:
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')
    data.index.name = 'Datetime'

# Keep only required columns
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

print(f"🌙 Data loaded: {len(data)} bars ✨")
print(f"🚀 Date range: {data.index[0]} to {data.index[-1]}")


class CompressionCoil(Strategy):
    bb_period = 20
    bb_std = 2.0
    rsi_period = 14
    atr_period = 14
    squeeze_lookback = 30
    rsi_threshold = 40
    atr_stop_mult = 1.5
    risk_pct = 0.02
    max_hold_bars = 20

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low

        # Bollinger Bands (talib - no backtesting.lib)
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS, close, timeperiod=self.bb_period,
            nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0
        )
        print("🌙✨ Bollinger Bands initialized 🚀")

        # Band Width
        self.band_width = self.I(
            lambda u, m, l: (u - l) / m,
            self.bb_upper, self.bb_middle, self.bb_lower
        )

        # Rolling min of band width (squeeze detection)
        self.bw_min = self.I(
            lambda bw: pd.Series(bw).rolling(self.squeeze_lookback).min().values,
            self.band_width
        )
        print("🌙✨ Band Width + Squeeze detector initialized 🚀")

        # RSI (talib - no backtesting.lib)
        self.rsi = self.I(talib.RSI, close, timeperiod=self.rsi_period)
        print("🌙✨ RSI initialized 🚀")

        # ATR (talib - no backtesting.lib)
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)
        print("🌙✨ ATR initialized 🚀")

        self.entry_bar = None
        self.stop_price = None

    def next(self):
        price = self.data.Close[-1]

        # Manage open position
        if self.position:
            bars_held = len(self.data) - self.entry_bar if self.entry_bar is not None else 0

            # Profit target: close above upper BB
            if price > self.bb_upper[-1]:
                print(f"🌙✨ PROFIT TARGET HIT! Close {price:.2f} > Upper BB {self.bb_upper[-1]:.2f} 🚀")
                self.position.close()
                self.entry_bar = None
                return

            # Stop loss
            if self.stop_price is not None and price <= self.stop_price:
                print(f"🌙💥 STOP LOSS HIT! Price {price:.2f} <= Stop {self.stop_price:.2f}")
                self.position.close()
                self.entry_bar = None
                return

            # Max holding period
            if bars_held >= self.max_hold_bars:
                print(f"🌙⏰ MAX HOLD REACHED ({bars_held} bars). Closing at {price:.2f}")
                self.position.close()
                self.entry_bar = None
                return

            return

        # Entry logic - pure numpy/pandas boolean conditions (no backtesting.lib)
        bw_ok = not np.isnan(self.bw_min[-1]) and self.band_width[-1] <= self.bw_min[-1]
        rsi_ok = not np.isnan(self.rsi[-1]) and self.rsi[-1] < self.rsi_threshold
        bullish = self.data.Close[-1] > self.data.Open[-1]

        if bw_ok and rsi_ok and bullish:
            atr_val = self.atr[-1]
            if np.isnan(atr_val) or atr_val <= 0:
                return

            stop_distance = self.atr_stop_mult * atr_val
            risk_amount = self.equity * self.risk_pct
            position_size = int(round(risk_amount / stop_distance))
            if position_size < 1:
                position_size = 1

            # Cap position size to available equity to avoid insufficient cash
            max_units = int(self.equity / price)
            if position_size > max_units:
                position_size = max_units
            if position_size < 1:
                return

            self.stop_price = price - stop_distance
            self.entry_bar = len(self.data)

            print(f"🌙🚀 COMPRESSION COIL ENTRY! Price {price:.2f} | RSI {self.rsi[-1]:.2f} | BW {self.band_width[-1]:.4f} | Stop {self.stop_price:.2f} | Size {position_size}")
            self.buy(size=position_size)


bt = Backtest(data, CompressionCoil, cash=1_000_000, commission=0.001)

print("🌙✨ Running CompressionCoil Backtest... 🚀")
stats = bt.run()
print(stats)
print(stats._strategy)