import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# ============================================================
# 🌙 Moon Dev's CompressionTrigger Backtest 🌙
# ============================================================

print("🌙✨ Moon Dev Backtest Engine Starting... ✨🚀")

# Load data
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

# Parse datetime
if 'datetime' in data.columns:
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')

# 🌙 Ensure numeric dtypes for talib (fix "input array type is not double")
for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    data[col] = pd.to_numeric(data[col], errors='coerce').astype(np.float64)

data = data.dropna()

print(f"🌙 Data loaded: {len(data)} bars 📊")
print(f"🚀 Date range: {data.index[0]} to {data.index[-1]}")


class CompressionTrigger(Strategy):
    """
    🌙 CompressionTrigger Strategy
    Volatility-contraction breakout system using dual volatility filters.
    """

    # Strategy parameters
    atr_period = 10
    atr_sma_long = 50
    atr_sma_short = 20
    range_period = 5
    vol_sma_period = 20
    compression_bars = 3
    time_stop_bars = 12
    risk_pct = 0.01
    atr_stop_mult = 1.5
    use_volume_filter = True

    def init(self):
        print("🌙 Initializing CompressionTrigger indicators... ✨")

        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        # ATR indicators
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)
        self.atr_sma50 = self.I(talib.SMA, self.atr, timeperiod=self.atr_sma_long)
        self.atr_sma20 = self.I(talib.SMA, self.atr, timeperiod=self.atr_sma_short)

        # Rolling high/low (prior N bars, excluding current)
        self.roll_high = self.I(talib.MAX, high, timeperiod=self.range_period)
        self.roll_low = self.I(talib.MIN, low, timeperiod=self.range_period)

        # Volume SMA
        self.vol_sma = self.I(talib.SMA, volume, timeperiod=self.vol_sma_period)

        # Track trade state
        self.entry_bar = None
        self.entry_price = None
        self.stop_price = None

        print("🚀 Indicators ready! Let's hunt some breakouts! 🌙")

    def next(self):
        # Need enough bars
        if len(self.data) < self.atr_sma_long + 5:
            return

        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        vol = self.data.Volume[-1]

        atr = self.atr[-1]
        atr_sma50 = self.atr_sma50[-1]
        atr_sma20 = self.atr_sma20[-1]

        if np.isnan(atr) or np.isnan(atr_sma50) or np.isnan(atr_sma20):
            return

        # Compression: ATR < 50-SMA ATR
        compression = atr < atr_sma50

        # Sustained compression for N bars
        sustained = True
        for i in range(1, self.compression_bars + 1):
            if len(self.atr) > i and len(self.atr_sma50) > i:
                if not (self.atr[-i] < self.atr_sma50[-i]):
                    sustained = False
                    break

        # Prior 5-bar high/low (exclude current bar)
        prior_high = self.roll_high[-2] if len(self.roll_high) > 1 else np.nan
        prior_low = self.roll_low[-2] if len(self.roll_low) > 1 else np.nan

        if np.isnan(prior_high) or np.isnan(prior_low):
            return

        # Volume confirmation
        vol_ok = True
        if self.use_volume_filter:
            vol_sma = self.vol_sma[-1]
            if not np.isnan(vol_sma):
                vol_ok = vol > vol_sma

        # ============ MANAGE OPEN POSITION ============
        if self.position:
            bars_held = len(self.data) - self.entry_bar if self.entry_bar else 0

            # Primary exit: ATR expansion (ATR > 20-SMA ATR)
            if atr > atr_sma20:
                print(f"🌙✨ VOL EXPANSION EXIT @ {price:.2f} | ATR={atr:.2f} > SMA20={atr_sma20:.2f} 🚀")
                self.position.close()
                self.entry_bar = None
                return

            # Time stop
            if bars_held >= self.time_stop_bars:
                print(f"⏰ Time stop hit after {bars_held} bars @ {price:.2f} 🌙")
                self.position.close()
                self.entry_bar = None
                return

            # Stop loss check
            if self.stop_price is not None:
                if self.position.is_long and low <= self.stop_price:
                    print(f"🛑 Long STOP @ {self.stop_price:.2f} 💥")
                    self.position.close()
                    self.entry_bar = None
                    return
                if self.position.is_short and high >= self.stop_price:
                    print(f"🛑 Short STOP @ {self.stop_price:.2f} 💥")
                    self.position.close()
                    self.entry_bar = None
                    return

            return  # Don't open new trades while in position

        # ============ ENTRY LOGIC ============
        if not compression or not sustained:
            return

        # Long breakout: close above prior 5-bar high
        if price > prior_high and vol_ok:
            # Stop: below prior 5-bar low or 1.5 ATR below entry
            stop_atr = price - self.atr_stop_mult * atr
            stop_consol = prior_low
            stop = max(stop_atr, stop_consol)

            risk_per_unit = price - stop
            if risk_per_unit <= 0:
                return

            risk_amount = self.equity * self.risk_pct
            size = int(round(risk_amount / risk_per_unit))
            if size <= 0:
                size = 1

            print(f"🚀🌙 LONG BREAKOUT @ {price:.2f} | PriorHigh={prior_high:.2f} | Stop={stop:.2f} | Size={size} ✨")
            self.buy(size=size)
            self.entry_bar = len(self.data)
            self.entry_price = price
            self.stop_price = stop

        # Short breakout: close below prior 5-bar low
        elif price < prior_low and vol_ok:
            stop_atr = price + self.atr_stop_mult * atr
            stop_consol = prior_high
            stop = min(stop_atr, stop_consol)

            risk_per_unit = stop - price
            if risk_per_unit <= 0:
                return

            risk_amount = self.equity * self.risk_pct
            size = int(round(risk_amount / risk_per_unit))
            if size <= 0:
                size = 1

            print(f"🌙🚀 SHORT BREAKDOWN @ {price:.2f} | PriorLow={prior_low:.2f} | Stop={stop:.2f} | Size={size} ✨")
            self.sell(size=size)
            self.entry_bar = len(self.data)
            self.entry_price = price
            self.stop_price = stop


# ============================================================
# 🌙 RUN BACKTEST 🚀
# ============================================================
print("🌙✨ Launching CompressionTrigger Backtest... 🚀🚀")

bt = Backtest(
    data,
    CompressionTrigger,
    cash=1_000_000,
    commission=0.002,
    exclusive_orders=True
)

stats = bt.run()
print(stats)
print(stats._strategy)

print("🌙✨ Backtest complete! Moon Dev out! 🚀💫")