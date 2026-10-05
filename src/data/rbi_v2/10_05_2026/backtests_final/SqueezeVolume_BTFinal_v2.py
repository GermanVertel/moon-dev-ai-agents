import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Load and clean data
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Rename columns to match backtesting requirements
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

# Ensure numeric dtypes (fixes talib "input array type is not double")
for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    data[col] = pd.to_numeric(data[col], errors='coerce').astype(np.float64)

data = data.dropna()

print("🌙✨ Moon Dev Data Loading Complete! ✨🌙")
print(f"📊 Data shape: {data.shape}")
print(f"📈 Columns: {list(data.columns)}")


class SqueezeVolume(Strategy):
    # Strategy parameters
    bb_period = 20
    bb_std = 2.0
    bbw_lookback = 126  # 6-month low of BBW
    squeeze_window = 10  # bars after BBW low that breakout is valid
    vol_multiplier = 1.5
    vol_period = 20
    atr_period = 14
    atr_stop_mult = 1.5

    def init(self):
        print("🌙🚀 Initializing SqueezeVolume Strategy...")

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

        # 6-month low of BBW
        self.bbw_min = self.I(talib.MIN, self.bbw, timeperiod=self.bbw_lookback)

        # Volume SMA
        self.vol_sma = self.I(talib.SMA, volume, timeperiod=self.vol_period)

        # ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        # Track squeeze state
        self.bars_since_squeeze = 999
        self.squeeze_active = False
        self.stop_price = None

        print("🌙✅ Indicators initialized! ✨")

    def next(self):
        # Skip if not enough data
        if len(self.data) < self.bbw_lookback + 5:
            return

        current_bbw = self.bbw[-1]
        current_bbw_min = self.bbw_min[-1]

        # Detect squeeze: BBW at 6-month low
        if not np.isnan(current_bbw) and not np.isnan(current_bbw_min):
            if current_bbw <= current_bbw_min * 1.001:  # tiny tolerance
                self.squeeze_active = True
                self.bars_since_squeeze = 0
                print(f"🌙🔍 SQUEEZE DETECTED! BBW={current_bbw:.6f} at 6-month low 🎯")
            else:
                self.bars_since_squeeze += 1

        # Check if squeeze window expired
        if self.bars_since_squeeze > self.squeeze_window:
            self.squeeze_active = False

        # Skip if no active squeeze
        if not self.squeeze_active:
            return

        # Get current values
        price = self.data.Close[-1]
        upper = self.bb_upper[-1]
        lower = self.bb_lower[-1]
        volume = self.data.Volume[-1]
        vol_avg = self.vol_sma[-1]
        atr = self.atr[-1]

        if np.isnan(upper) or np.isnan(lower) or np.isnan(vol_avg) or np.isnan(atr):
            return

        # Volume confirmation
        vol_confirmed = volume >= self.vol_multiplier * vol_avg

        # No position - look for entry
        if not self.position:
            # Long entry
            if price > upper and vol_confirmed:
                stop_price = price - self.atr_stop_mult * atr
                risk = price - stop_price
                if risk > 0:
                    # Size = 0.99 fraction of equity (valid backtesting size)
                    self.buy(size=0.99)
                    self.stop_price = stop_price
                    print(f"🌙🚀 LONG ENTRY! Price={price:.2f} > Upper={upper:.2f}, Vol={volume:.2f} > {self.vol_multiplier}x{vol_avg:.2f} ✨")
                    print(f"   🛑 Stop: {stop_price:.2f}, ATR: {atr:.2f}")
                    self.squeeze_active = False

            # Short entry
            elif price < lower and vol_confirmed:
                stop_price = price + self.atr_stop_mult * atr
                risk = stop_price - price
                if risk > 0:
                    self.sell(size=0.99)
                    self.stop_price = stop_price
                    print(f"🌙🔻 SHORT ENTRY! Price={price:.2f} < Lower={lower:.2f}, Vol={volume:.2f} > {self.vol_multiplier}x{vol_avg:.2f} ✨")
                    print(f"   🛑 Stop: {stop_price:.2f}, ATR: {atr:.2f}")
                    self.squeeze_active = False

        # In position - check exits
        else:
            # Long position
            if self.position.is_long:
                # Primary exit: close back inside bands
                if price < upper:
                    self.position.close()
                    print(f"🌙💰 LONG EXIT (band re-entry)! Price={price:.2f} < Upper={upper:.2f} 🎯")
                    self.squeeze_active = False
                # Stop loss
                elif self.stop_price is not None and price <= self.stop_price:
                    self.position.close()
                    print(f"🌙🛑 LONG STOP OUT! Price={price:.2f} <= Stop={self.stop_price:.2f}")
                    self.squeeze_active = False

            # Short position
            elif self.position.is_short:
                # Primary exit: close back inside bands
                if price > lower:
                    self.position.close()
                    print(f"🌙💰 SHORT EXIT (band re-entry)! Price={price:.2f} > Lower={lower:.2f} 🎯")
                    self.squeeze_active = False
                # Stop loss
                elif self.stop_price is not None and price >= self.stop_price:
                    self.position.close()
                    print(f"🌙🛑 SHORT STOP OUT! Price={price:.2f} >= Stop={self.stop_price:.2f}")
                    self.squeeze_active = False


print("🌙✨ Starting Moon Dev Backtest! ✨🌙")

bt = Backtest(
    data,
    SqueezeVolume,
    cash=100000000,
    commission=0.002,
    exclusive_orders=True
)

stats = bt.run()
print(stats)
print(stats._strategy)