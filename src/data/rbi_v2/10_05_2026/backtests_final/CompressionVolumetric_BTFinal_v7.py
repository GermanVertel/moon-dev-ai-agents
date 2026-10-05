import pandas as pd
import numpy as np
import talib
from backtesting import Backtest, Strategy

# Load data
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Set datetime index BEFORE renaming (datetime column is lowercase here)
if 'datetime' in data.columns:
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')

# Map columns to backtesting requirements
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

# Ensure numeric dtypes for OHLCV (talib requires double / float64)
for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    data[col] = pd.to_numeric(data[col], errors='coerce').astype('float64')

data = data.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])

print("🌙 Moon Dev data loaded! Shape:", data.shape)
print("🚀 First few rows:\n", data.head())


class CompressionVolumetric(Strategy):
    bb_period = 20
    bb_std = 2.0
    bbw_lookback = 30
    vol_period = 20
    vol_multiplier = 1.5
    atr_period = 14
    atr_mult = 2.0
    risk_pct = 0.02
    time_stop_bars = 10

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        # Bollinger Bands
        self.upper, self.middle, self.lower = self.I(
            talib.BBANDS, close, timeperiod=self.bb_period,
            nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0
        )

        # Bollinger Band Width
        self.bbw = self.I(lambda u, l, m: (u - l) / m, self.upper, self.lower, self.middle)

        # Rolling min BBW over lookback
        self.bbw_min = self.I(
            lambda x: pd.Series(x).rolling(self.bbw_lookback).min().values,
            self.bbw
        )

        # Volume SMA - cast to float64 to satisfy talib
        self.vol_sma = self.I(
            lambda v: talib.SMA(np.asarray(v, dtype=np.float64), timeperiod=self.vol_period),
            volume
        )

        # ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        print("🌙✨ CompressionVolumetric indicators initialized! 🚀")

    def next(self):
        price = self.data.Close[-1]
        vol = self.data.Volume[-1]

        # Skip if indicators not ready
        if len(self.data) < self.bbw_lookback + 2:
            return
        if np.isnan(self.bbw_min[-1]) or np.isnan(self.vol_sma[-1]) or np.isnan(self.atr[-1]):
            return
        if np.isnan(self.upper[-1]) or np.isnan(self.lower[-1]) or np.isnan(self.middle[-1]):
            return

        # Compression condition: BBW at its lowest in lookback
        compression = self.bbw[-1] <= self.bbw_min[-1] * 1.001

        # Volume confirmation
        vol_confirm = vol > self.vol_multiplier * self.vol_sma[-1]

        # Manage existing positions
        if self.position:
            # Time stop
            if len(self.trades) > 0:
                trade = self.trades[-1]
                bars_held = len(self.data) - trade.entry_bar
                if bars_held >= self.time_stop_bars:
                    print(f"⏰ Moon Dev TIME STOP hit at {price:.2f} 🌙")
                    self.position.close()
                    return

            # Volume fade exit
            if vol < self.vol_sma[-1]:
                print(f"💨 Moon Dev Volume FADE exit at {price:.2f} 🌙")
                self.position.close()
                return

        # Entry logic
        if not self.position and compression:
            # Long entry
            if price > self.upper[-1] and vol_confirm:
                sl = self.middle[-1]
                risk = price - sl
                if risk > 0:
                    # Position size as fraction of equity (0 < size < 1)
                    size = min(0.95, (self.equity * self.risk_pct) / (risk * price))
                    if size > 0:
                        print(f"🚀🌙 Moon Dev LONG breakout! Price={price:.2f} Upper={self.upper[-1]:.2f} Vol={vol:.2f} VolSMA={self.vol_sma[-1]:.2f} Size={size:.4f}")
                        self.buy(size=size, sl=sl)

            # Short entry
            elif price < self.lower[-1] and vol_confirm:
                sl = self.middle[-1]
                risk = sl - price
                if risk > 0:
                    size = min(0.95, (self.equity * self.risk_pct) / (risk * price))
                    if size > 0:
                        print(f"🔻🌙 Moon Dev SHORT breakout! Price={price:.2f} Lower={self.lower[-1]:.2f} Vol={vol:.2f} VolSMA={self.vol_sma[-1]:.2f} Size={size:.4f}")
                        self.sell(size=size, sl=sl)


bt = Backtest(data, CompressionVolumetric, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)