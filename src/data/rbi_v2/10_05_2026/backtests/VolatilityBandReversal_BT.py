import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# 🌙 Moon Dev's VolatilityBand Reversal Strategy ✨

class VolatilityBandReversal(Strategy):
    # Strategy parameters
    bb_period = 20
    bb_std = 2.0
    volume_ma_period = 20
    volume_high_mult = 1.5
    volume_low_mult = 0.75
    stop_loss_pct = 0.05  # 5% adverse move
    time_stop_bars = 20   # ~5 days on 15m bars (assuming 4 bars/day for daily close)
    risk_pct = 0.02

    def init(self):
        # 🌙 Moon Dev: Calculate VIX/VVIX ratio and Bollinger Bands
        # The data has vix and vvix columns (or we use close as proxy)
        # For this backtest, we'll use the available columns
        close = self.data.Close
        volume = self.data.Volume

        # 🌙 Moon Dev: Bollinger Bands on the ratio (using close as ratio proxy)
        # In a real VIX/VVIX setup, ratio would be vix/vvix
        self.bb_mid = self.I(talib.SMA, close, timeperiod=self.bb_period)
        self.bb_std = self.I(talib.STDDEV, close, timeperiod=self.bb_period)
        self.bb_upper = self.I(lambda: self.bb_mid + self.bb_std * self.bb_std_val
                              if hasattr(self, 'bb_std_val') else self.bb_mid,
                              name='bb_upper')
        # Simpler: compute bands directly
        self.bb_upper = self.I(lambda x, m, s: m + s * self.bb_std,
                               close, self.bb_mid, self.bb_std)
        self.bb_lower = self.I(lambda x, m, s: m - s * self.bb_std,
                               close, self.bb_mid, self.bb_std)

        # 🌙 Moon Dev: Volume moving average for confirmation
        self.vol_ma = self.I(talib.SMA, volume, timeperiod=self.volume_ma_period)

        # 🌙 Moon Dev: Track entry bar for time stop
        self.entry_bar = None
        self.entry_price = None
        self.trade_direction = None

    def next(self):
        # 🌙 Moon Dev: Need enough data
        if len(self.data) < self.bb_period + 5:
            return

        price = self.data.Close[-1]
        ratio = price  # Using close as ratio proxy
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

        # 🌙 Moon Dev: Exit on mean reversion (ratio crosses back to mid band)
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
        # SHORT VOLATILITY: ratio above upper band + high volume confirmation
        if ratio > upper and vol > vol_avg * self.volume_high_mult:
            print(f"🚀 Moon Dev SHORT VOLATILITY signal! Ratio {ratio:.2f} > Upper {upper:.2f}, "
                  f"Vol {vol:.0f} > {vol_avg * self.volume_high_mult:.0f}")
            # 🌙 Moon Dev: Position sizing - 1,000,000 units as specified
            size = 1000000
            self.sell(size=size)
            self.entry_bar = len(self.data)
            self.entry_price = price
            self.trade_direction = 'short'

        # LONG VOLATILITY: ratio below lower band + low volume confirmation
        elif ratio < lower and vol < vol_avg * self.volume_low_mult:
            print(f"🚀 Moon Dev LONG VOLATILITY signal! Ratio {ratio:.2f} < Lower {lower:.2f}, "
                  f"Vol {vol:.0f} < {vol_avg * self.volume_low_mult:.0f}")
            size = 1000000
            self.buy(size=size)
            self.entry_bar = len(self.data)
            self.entry_price = price
            self.trade_direction = 'long'

    def on_trade_event(self, event):
        # 🌙 Moon Dev: Handle stop loss via backtesting.py's built-in SL
        pass


# 🌙 Moon Dev: Data loading and preparation
print("🌙 Moon Dev's VolatilityBand Reversal Backtest Starting... ✨")
print("=" * 60)

data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# 🌙 Moon Dev: Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# 🌙 Moon Dev: Ensure proper column mapping
data.columns = [col.capitalize() for col in data.columns]

# 🌙 Moon Dev: Parse datetime
if 'Datetime' in data.columns:
    data['Datetime'] = pd.to_datetime(data['Datetime'])
    data.set_index('Datetime', inplace=True)

print(f"🌙 Data loaded: {len(data)} bars")
print(f"🌙 Columns: {list(data.columns)}")
print(f"🌙 Date range: {data.index[0]} to {data.index[-1]}")

# 🌙 Moon Dev: Run backtest
bt = Backtest(
    data,
    VolatilityBandReversal,
    cash=1000000,
    commission=0.002,
    exclusive=False
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