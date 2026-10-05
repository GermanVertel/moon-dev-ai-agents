import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

print("🌙✨ Moon Dev's SqueezeSurge Backtest Initializing... 🚀")

data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper column mapping
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

# Set datetime index
if 'datetime' in data.columns:
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')

# Ensure numeric dtypes for talib
for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    data[col] = pd.to_numeric(data[col], errors='coerce').astype(np.float64)

data = data.dropna()

print(f"🌙 Data loaded: {len(data)} rows ✨")
print(f"🚀 Columns: {list(data.columns)}")


class SqueezeSurge(Strategy):
    bb_period = 20
    bb_std = 2.0
    bbw_lookback = 20
    vol_sma_period = 50
    atr_period = 14
    atr_mult = 2.0
    risk_pct = 0.02
    time_exit_bars = 20

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

        # Bollinger Band Width
        self.bbw = self.I(
            lambda u, l: u - l, self.bb_upper, self.bb_lower, name="BBW"
        )

        # Lowest BBW over lookback
        self.bbw_min = self.I(
            talib.MIN, self.bbw, timeperiod=self.bbw_lookback, name="BBW_MIN"
        )

        # Volume SMA
        vol_arr = np.asarray(volume, dtype=np.float64)
        self.vol_sma = self.I(
            talib.SMA, vol_arr, timeperiod=self.vol_sma_period, name="VOL_SMA"
        )

        # ATR
        self.atr = self.I(
            talib.ATR, high, low, close, timeperiod=self.atr_period, name="ATR"
        )

        self.highest_high = 0.0
        self.entry_bar = 0
        self.trailing_stop = 0.0

        print("🌙✨ Indicators initialized! 🚀")

    def next(self):
        price = self.data.Close[-1]

        # Guard against NaN indicators
        if (np.isnan(self.bbw[-1]) or np.isnan(self.bbw_min[-1]) or
                np.isnan(self.vol_sma[-1]) or np.isnan(self.bb_upper[-1]) or
                np.isnan(self.atr[-1])):
            return

        if not self.position:
            # Entry conditions
            bbw_squeeze = self.bbw[-1] <= self.bbw_min[-1]
            vol_surge = self.data.Volume[-1] > self.vol_sma[-1]
            breakout = self.data.Close[-1] > self.bb_upper[-1]

            if bbw_squeeze and vol_surge and breakout:
                atr_val = self.atr[-1]
                if atr_val > 0:
                    risk_amount = self.equity * self.risk_pct
                    stop_distance = self.atr_mult * atr_val
                    # Position sizing: risk-based fraction of equity
                    position_size = risk_amount / stop_distance
                    size_fraction = position_size / price
                    # Clamp between 0.01 and 0.99
                    if size_fraction <= 0:
                        size_fraction = 0.01
                    if size_fraction > 0.99:
                        size_fraction = 0.99

                    self.trailing_stop = price - stop_distance
                    self.highest_high = price
                    self.entry_bar = len(self.data)

                    print(f"🌙🚀 LONG ENTRY! Price: {price:.2f} | Size Frac: {size_fraction:.4f} | ATR: {atr_val:.2f} | Stop: {self.trailing_stop:.2f} ✨")
                    self.buy(size=size_fraction)
        else:
            # Update highest high
            if self.data.High[-1] > self.highest_high:
                self.highest_high = self.data.High[-1]

            # Update trailing stop
            new_stop = self.highest_high - (self.atr_mult * self.atr[-1])
            if new_stop > self.trailing_stop:
                self.trailing_stop = new_stop

            # Exit conditions
            bars_held = len(self.data) - self.entry_bar
            price_below_stop = self.data.Close[-1] < self.trailing_stop
            time_exit = bars_held >= self.time_exit_bars

            if price_below_stop:
                print(f"🌙🛑 TRAILING STOP HIT! Price: {price:.2f} | Stop: {self.trailing_stop:.2f} | Bars: {bars_held} ✨")
                self.position.close()
            elif time_exit:
                print(f"🌙⏰ TIME EXIT! Price: {price:.2f} | Bars: {bars_held} ✨")
                self.position.close()


bt = Backtest(
    data, SqueezeSurge,
    cash=1000000, commission=0.001
)

stats = bt.run()
print(stats)
print(stats._strategy)