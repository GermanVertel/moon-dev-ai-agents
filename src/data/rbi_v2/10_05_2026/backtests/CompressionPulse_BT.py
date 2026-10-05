import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Load data
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Ensure proper column mapping
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')

print("🌙✨ CompressionPulse Backtest Initializing... 🚀")
print(f"📊 Data loaded: {len(data)} bars")
print(f"📈 Date range: {data.index[0]} to {data.index[-1]}")


class CompressionPulse(Strategy):
    # Parameters
    bb_period = 20
    bb_std = 2.0
    bbw_avg_period = 20
    roc_period = 3
    vol_median_period = 20
    vol_multiplier = 1.5
    time_stop_bars = 10
    risk_pct = 0.01  # 1% risk per trade
    atr_period = 14

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
        def calc_bbw(upper, middle, lower):
            return (upper - lower) / middle

        self.bbw = self.I(calc_bbw, self.bb_upper, self.bb_middle, self.bb_lower)

        # BBW 20-period average
        self.bbw_avg = self.I(talib.SMA, self.bbw, timeperiod=self.bbw_avg_period)

        # ROC 3-period
        self.roc = self.I(talib.ROC, close, timeperiod=self.roc_period)

        # Volume median 20-period
        self.vol_median = self.I(talib.SMA, volume, timeperiod=self.vol_median_period)

        # ATR for stop loss calculation
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        # Track entry bar for time stop
        self.entry_bar = None

        print("🌙 Indicators initialized: BB, BBW, ROC, VolMedian, ATR ✨")

    def next(self):
        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        volume = self.data.Volume[-1]

        # Skip if indicators not ready
        if (np.isnan(self.bbw[-1]) or np.isnan(self.bbw_avg[-1]) or
                np.isnan(self.roc[-1]) or np.isnan(self.roc[-2]) or
                np.isnan(self.vol_median[-1]) or np.isnan(self.bb_upper[-1]) or
                np.isnan(self.bb_lower[-1]) or np.isnan(self.bb_middle[-1]) or
                np.isnan(self.atr[-1])):
            return

        # Manage open position
        if self.position:
            # Primary exit: price touches or closes above upper band
            if price >= self.bb_upper[-1]:
                print(f"🎯 Moon Dev EXIT - Upper Band Touch! Price: {price:.2f}, Upper: {self.bb_upper[-1]:.2f} 🌙")
                self.position.close()
                self.entry_bar = None
                return

            # Time stop: N bars and ROC rolls back below zero
            bars_held = len(self.data) - self.entry_bar if self.entry_bar else 0
            if bars_held >= self.time_stop_bars and self.roc[-1] < 0:
                print(f"⏰ Moon Dev TIME STOP - {bars_held} bars, ROC: {self.roc[-1]:.2f} 🌙")
                self.position.close()
                self.entry_bar = None
                return

            return

        # Entry conditions
        # 1. Squeeze: BBW below its 20-period average
        squeeze = self.bbw[-1] < self.bbw_avg[-1]

        # 2. Momentum flip: ROC crosses from <= 0 to > 0
        roc_cross = self.roc[-2] <= 0 and self.roc[-1] > 0

        # 3. Volume confirmation: current volume >= 1.5x 20-period median
        vol_confirm = volume >= self.vol_multiplier * self.vol_median[-1]

        # 4. Directional bias: price at or above middle band
        direction_bias = price >= self.bb_middle[-1]

        if squeeze and roc_cross and vol_confirm and direction_bias:
            # Calculate stop loss: below lower band or swing low, whichever is tighter
            lower_band_stop = self.bb_lower[-1]
            atr_stop = price - (1.5 * self.atr[-1])
            stop_price = max(lower_band_stop, atr_stop)  # tighter = higher of the two

            risk_per_unit = price - stop_price
            if risk_per_unit <= 0:
                return

            # Position sizing: risk fixed % of equity
            risk_amount = self.equity * self.risk_pct
            position_size = risk_amount / risk_per_unit
            position_size = int(round(position_size))

            if position_size < 1:
                return

            # Cap position size to avoid over-leverage
            max_size = int(self.equity / price) if price > 0 else 0
            position_size = min(position_size, max_size)

            if position_size < 1:
                return

            print(f"🚀 Moon Dev LONG Entry! Price: {price:.2f}, Size: {position_size}, "
                  f"Stop: {stop_price:.2f}, BBW: {self.bbw[-1]:.4f} < Avg: {self.bbw_avg[-1]:.4f}, "
                  f"ROC: {self.roc[-1]:.2f}, Vol: {volume:.2f} >= {self.vol_multiplier * self.vol_median[-1]:.2f} 🌙")

            self.buy(size=position_size, sl=stop_price)
            self.entry_bar = len(self.data)


bt = Backtest(data, CompressionPulse, cash=1_000_000, commission=0.002)

stats = bt.run()
print(stats)
print(stats._strategy)