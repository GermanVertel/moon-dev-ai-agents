import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# 🌙 Moon Dev CompressionIgnition Backtest 🚀

data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"

print("🌙 Loading Moon Dev data from:", data_path)
data = pd.read_csv(data_path)

# Clean column names
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

data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

print("✨ Data loaded! Shape:", data.shape)
print("🚀 Starting CompressionIgnition backtest...")


class CompressionIgnition(Strategy):
    bb_period = 20
    bb_std = 2.0
    squeeze_lookback = 30
    atr_period = 14
    atr_multiplier = 0.5
    stop_atr_mult = 1.5
    risk_pct = 0.01

    def init(self):
        print("🌙 Initializing CompressionIgnition indicators...")
        close = self.data.Close
        high = self.data.High
        low = self.data.Low

        # Bollinger Bands
        def bb_func(close):
            upper, middle, lower = talib.BBANDS(
                close, timeperiod=self.bb_period,
                nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0
            )
            return upper, middle, lower
        self.upper, self.middle, self.lower = self.I(bb_func, close)

        # Bollinger Band Width
        def bbw_func(upper, lower, middle):
            return (upper - lower) / middle
        self.bbw = self.I(bbw_func, self.upper, self.lower, self.middle)

        # Rolling minimum of BBW (30-bar lowest)
        def bbw_min_func(bbw):
            return talib.MIN(bbw, timeperiod=self.squeeze_lookback)
        self.bbw_min = self.I(bbw_min_func, self.bbw)

        # ATR
        def atr_func(high, low, close):
            return talib.ATR(high, low, close, timeperiod=self.atr_period)
        self.atr = self.I(atr_func, high, low, close)

        print("✨ Indicators ready! BBW, BBW_MIN, ATR, BB initialized.")

    def next(self):
        price = self.data.Close[-1]
        upper = self.upper[-1]
        middle = self.middle[-1]
        lower = self.lower[-1]
        bbw = self.bbw[-1]
        bbw_min = self.bbw_min[-1]
        atr = self.atr[-1]

        if np.isnan(bbw) or np.isnan(bbw_min) or np.isnan(atr) or np.isnan(upper) or np.isnan(lower) or np.isnan(middle):
            return

        # 🌙 Squeeze detection (with small tolerance so equality can trigger)
        squeeze = bbw <= bbw_min * 1.05

        # 🚀 Breakout confirmation
        breakout_long = price > (upper + (self.atr_multiplier * atr))
        breakout_short = price < (lower - (self.atr_multiplier * atr))

        if not self.position:
            if squeeze and breakout_long:
                stop_price = max(price - (self.stop_atr_mult * atr), middle)
                stop_distance = price - stop_price
                if stop_distance <= 0:
                    stop_distance = self.stop_atr_mult * atr

                risk_amount = self.equity * self.risk_pct
                position_size = risk_amount / stop_distance
                position_size = int(round(position_size))

                # Convert units to fraction of equity for backtesting.py
                if position_size > 0:
                    size_fraction = (position_size * price) / self.equity
                    size_fraction = min(max(size_fraction, 0.001), 0.99)
                    print(f"🚀🌙 LONG SIGNAL! Price={price:.2f} Upper={upper:.2f} ATR={atr:.2f} Size={size_fraction:.4f}")
                    self.buy(size=size_fraction, sl=stop_price)

            elif squeeze and breakout_short:
                stop_price = min(price + (self.stop_atr_mult * atr), middle)
                stop_distance = stop_price - price
                if stop_distance <= 0:
                    stop_distance = self.stop_atr_mult * atr

                risk_amount = self.equity * self.risk_pct
                position_size = risk_amount / stop_distance
                position_size = int(round(position_size))

                if position_size > 0:
                    size_fraction = (position_size * price) / self.equity
                    size_fraction = min(max(size_fraction, 0.001), 0.99)
                    print(f"🔻🌙 SHORT SIGNAL! Price={price:.2f} Lower={lower:.2f} ATR={atr:.2f} Size={size_fraction:.4f}")
                    self.sell(size=size_fraction, sl=stop_price)

        else:
            # Exit logic
            if self.position.is_long:
                if price < middle:
                    print(f"🌙 Exit LONG at {price:.2f} (momentum failure)")
                    self.position.close()
            elif self.position.is_short:
                if price > middle:
                    print(f"🌙 Exit SHORT at {price:.2f} (momentum failure)")
                    self.position.close()


bt = Backtest(data, CompressionIgnition, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)