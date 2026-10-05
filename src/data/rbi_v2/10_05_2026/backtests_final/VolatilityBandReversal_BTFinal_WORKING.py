import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# 🌙 Moon Dev's VolatilityBand Reversal Strategy ✨

class VolatilityBandReversal(Strategy):
    # Strategy parameters
    bb_period = 20
    bb_std_mult = 2.0
    volume_ma_period = 20
    volume_high_mult = 1.5
    volume_low_mult = 0.75
    stop_loss_pct = 0.05  # 5% adverse move
    time_stop_bars = 20   # ~5 days on 15m bars
    risk_pct = 0.02

    def init(self):
        # 🌙 Moon Dev: Cast to float64 to satisfy talib's double requirement
        close = self.data.Close.astype(np.float64)
        volume = self.data.Volume.astype(np.float64)

        # 🌙 Moon Dev: Bollinger Bands on close
        self.bb_mid = self.I(talib.SMA, close, timeperiod=self.bb_period, name='bb_mid')
        bb_stddev = self.I(talib.STDDEV, close, timeperiod=self.bb_period, nbdev=1, name='bb_stddev')

        # 🌙 Moon Dev: Compute upper/lower bands via lambda wrapping arrays
        self.bb_upper = self.I(
            lambda m, s: m + s * self.bb_std_mult,
            self.bb_mid, bb_stddev,
            name='bb_upper'
        )
        self.bb_lower = self.I(
            lambda m, s: m - s * self.bb_std_mult,
            self.bb_mid, bb_stddev,
            name='bb_lower'
        )

        # 🌙 Moon Dev: Volume moving average for confirmation
        self.vol_ma = self.I(talib.SMA, volume, timeperiod=self.volume_ma_period, name='vol_ma')

        # 🌙 Moon Dev: Track entry bar for time stop
        self.entry_bar = None
        self.entry_price = None
        self.trade_direction = None

    def next(self):
        # 🌙 Moon Dev: Need enough data
        if len(self.data) < self.bb_period + 5:
            return

        price = self.data.Close[-1]
        ratio = price
        upper = self.bb_upper[-1]
        lower = self.bb_lower[-1]
        mid = self.bb_mid[-1]
        vol = self.data.Volume[-1]
        vol_avg = self.vol_ma[-1]

        # 🌙 Moon Dev: Time stop check
        if self.position and self.entry_bar is not None:
            bars_held = len(self.data) - self.entry_bar
            if bars_held >= self.time_stop_bars:
                print(f"⏰ Moon Dev TIME STOP hit after {bars_held} bars! Closing position.")
                self.position.close()
                self.entry_bar = None
                self.trade_direction = None
                return

        # 🌙 Moon Dev: Exit on mean reversion
        if self.position:
            if self.trade_direction == 'short' and ratio <= mid:
                print(f"🎯 Moon Dev PROFIT TARGET hit! Ratio {ratio:.2f} reverted to SMA {mid:.2f}")
                self.position.close()
                self.entry_bar = None
                self.trade_direction = None
                return
            elif self.trade_direction == 'long' and ratio >= mid:
                print(f"🎯 Moon Dev PROFIT TARGET hit! Ratio {ratio:.2f} reverted to SMA {mid:.2f}")
                self.position.close()
                self.entry_bar = None
                self.trade_direction = None
                return

        # 🌙 Moon Dev: Skip if already in position
        if self.position:
            return

        # 🌙 Moon Dev: Entry logic
        if ratio > upper and vol > vol_avg * self.volume_high_mult:
            print(f"🚀 Moon Dev SHORT VOLATILITY signal! Ratio {ratio:.2f} > Upper {upper:.2f}, "
                  f"Vol {vol:.0f} > {vol_avg * self.volume_high_mult:.0f}")
            self.sell(size=0.99)
            self.entry_bar = len(self.data)
            self.entry_price = price
            self.trade_direction = 'short'

        elif ratio < lower and vol < vol_avg * self.volume_low_mult:
            print(f"🚀 Moon Dev LONG VOLATILITY signal! Ratio {ratio:.2f} < Lower {lower:.2f}, "
                  f"Vol {vol:.0f} < {vol_avg * self.volume_low_mult:.0f}")
            self.buy(size=0.99)
            self.entry_bar = len(self.data)
            self.entry_price = price
            self.trade_direction = 'long'


# 🌙 Moon Dev: Data loading and preparation
print("🌙 Moon Dev's VolatilityBand Reversal Backtest Starting... ✨")
print("=" * 60)

data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# 🌙 Moon Dev: Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# 🌙 Moon Dev: Rename columns properly
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

# 🌙 Moon Dev: Parse datetime and set as index
if 'datetime' in data.columns:
    data = data.set_index(pd.to_datetime(data['datetime']))

# 🌙 Moon Dev: Ensure OHLCV are float64 for talib compatibility
for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    if col in data.columns:
        data[col] = data[col].astype(np.float64)

print(f"🌙 Data loaded: {len(data)} bars")
print(f"🌙 Columns: {list(data.columns)}")
print(f"🌙 Date range: {data.index[0]} to {data.index[-1]}")

# 🌙 Moon Dev: Run backtest
bt = Backtest(
    data,
    VolatilityBandReversal,
    cash=1000000,
    commission=0.002
)

stats = bt.run()
print("\n" + "=" * 60)
print("🌙 Moon Dev's VolatilityBand Reversal - Full Stats ✨")
print("=" * 60)
print(stats)
print("\n" + "=" * 60)
print("🌙 Strategy Details:")
print("=" * 60)
print(stats._strategy)
print("\n🌙 Moon Dev Backtest Complete! 🚀✨")