import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

print("🌙✨ Moon Dev Backtest AI Initializing... ✨🌙")
print("🚀 Loading SqueezeVolumeTrigger Strategy...")

data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"

data = pd.read_csv(data_path)
print(f"📊 Data loaded: {len(data)} rows")

data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

if 'datetime' in data.columns:
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')

data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
# Ensure all OHLCV columns are float64 (talib requires double arrays)
for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    data[col] = data[col].astype(np.float64)

print(f"✅ Data cleaned. Columns: {list(data.columns)}")
print(f"📅 Date range: {data.index[0]} to {data.index[-1]}")


class SqueezeVolumeTrigger(Strategy):
    bb_period = 20
    bb_std = 2.0
    squeeze_lookback = 20
    volume_ma_period = 50
    atr_period = 14
    atr_mult = 2.0
    risk_pct = 0.02
    time_stop_bars = 10
    volume_mult = 1.5

    def init(self):
        print("🌙 Initializing indicators...")
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        vol = self.data.Volume

        # Ensure arrays are float64 for talib
        close_arr = np.asarray(close, dtype=np.float64)
        high_arr = np.asarray(high, dtype=np.float64)
        low_arr = np.asarray(low, dtype=np.float64)
        vol_arr = np.asarray(vol, dtype=np.float64)

        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS, close_arr, timeperiod=self.bb_period,
            nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0
        )

        # Compute BB width via a custom function using talib outputs
        def _bb_width(upper, middle, lower):
            upper = np.asarray(upper, dtype=np.float64)
            middle = np.asarray(middle, dtype=np.float64)
            lower = np.asarray(lower, dtype=np.float64)
            with np.errstate(divide='ignore', invalid='ignore'):
                return (upper - lower) / middle

        self.bb_width = self.I(
            _bb_width, self.bb_upper, self.bb_middle, self.bb_lower, name="BB_Width"
        )

        self.bb_width_min = self.I(
            talib.MIN, np.asarray(self.bb_width, dtype=np.float64),
            timeperiod=self.squeeze_lookback, name="BB_Width_Min"
        )
        self.bb_width_avg = self.I(
            talib.SMA, np.asarray(self.bb_width, dtype=np.float64),
            timeperiod=self.squeeze_lookback, name="BB_Width_Avg"
        )
        self.bb_width_avg50 = self.I(
            talib.SMA, np.asarray(self.bb_width, dtype=np.float64),
            timeperiod=50, name="BB_Width_Avg50"
        )

        self.vol_ma = self.I(
            talib.SMA, vol_arr, timeperiod=self.volume_ma_period, name="Vol_MA"
        )
        self.atr = self.I(
            talib.ATR, high_arr, low_arr, close_arr,
            timeperiod=self.atr_period, name="ATR"
        )

        self.entry_bar = None
        self.entry_price = None
        self.stop_price = None
        self.direction = None
        print("✨ Indicators ready!")

    def next(self):
        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        vol = self.data.Volume[-1]

        bb_w = self.bb_width[-1]
        bb_w_min = self.bb_width_min[-1]
        bb_w_avg = self.bb_width_avg[-1]
        bb_w_avg50 = self.bb_width_avg50[-1]
        vol_ma = self.vol_ma[-1]
        atr = self.atr[-1]
        upper = self.bb_upper[-1]
        lower = self.bb_lower[-1]

        if np.isnan(bb_w) or np.isnan(bb_w_min) or np.isnan(vol_ma) or np.isnan(atr):
            return

        # Manage open position
        if self.position:
            bars_held = len(self.data) - 1 - self.entry_bar if self.entry_bar else 0

            if self.direction == 'long':
                # Trailing stop
                new_stop = high - self.atr_mult * atr
                if new_stop > self.stop_price:
                    self.stop_price = new_stop
                if low <= self.stop_price:
                    print(f"🛑 Long stop hit @ {self.stop_price:.2f}")
                    self.position.close()
                    self.entry_bar = None
                    return
                # Channel expansion exit
                if bb_w > bb_w_avg:
                    print(f"🎯 Long exit: BB width expanded ({bb_w:.4f} > {bb_w_avg:.4f})")
                    self.position.close()
                    self.entry_bar = None
                    return
            else:
                new_stop = low + self.atr_mult * atr
                if new_stop < self.stop_price:
                    self.stop_price = new_stop
                if high >= self.stop_price:
                    print(f"🛑 Short stop hit @ {self.stop_price:.2f}")
                    self.position.close()
                    self.entry_bar = None
                    return
                if bb_w > bb_w_avg:
                    print(f"🎯 Short exit: BB width expanded ({bb_w:.4f} > {bb_w_avg:.4f})")
                    self.position.close()
                    self.entry_bar = None
                    return

            # Time stop
            if bars_held >= self.time_stop_bars:
                print(f"⏰ Time stop hit after {bars_held} bars")
                self.position.close()
                self.entry_bar = None
                return
            return

        # Entry logic
        squeeze = bb_w <= bb_w_min
        no_choppy = bb_w < bb_w_avg50 if not np.isnan(bb_w_avg50) else True
        vol_trigger = vol > vol_ma * self.volume_mult

        if squeeze and no_choppy and vol_trigger:
            risk_amount = self.equity * self.risk_pct
            stop_dist = self.atr_mult * atr
            if stop_dist <= 0:
                return
            position_size = int(round(risk_amount / stop_dist))
            if position_size < 1:
                position_size = 1

            if price > upper:
                self.stop_price = price - stop_dist
                self.direction = 'long'
                self.entry_bar = len(self.data) - 1
                self.entry_price = price
                print(f"🚀🌙 LONG ENTRY @ {price:.2f} | BB_W={bb_w:.4f} | Vol={vol:.0f} vs MA={vol_ma:.0f} | Stop={self.stop_price:.2f} | Size={position_size}")
                self.buy(size=position_size)
            elif price < lower:
                self.stop_price = price + stop_dist
                self.direction = 'short'
                self.entry_bar = len(self.data) - 1
                self.entry_price = price
                print(f"🔻🌙 SHORT ENTRY @ {price:.2f} | BB_W={bb_w:.4f} | Vol={vol:.0f} vs MA={vol_ma:.0f} | Stop={self.stop_price:.2f} | Size={position_size}")
                self.sell(size=position_size)


print("🌙 Starting backtest...")
bt = Backtest(data, SqueezeVolumeTrigger, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)