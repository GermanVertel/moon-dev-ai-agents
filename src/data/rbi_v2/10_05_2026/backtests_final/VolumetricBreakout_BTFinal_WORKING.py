import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

print("🌙✨ Moon Dev Backtest AI initializing... VolumetricBreakout loading! 🚀")

# Load data
data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper mapping
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

print(f"🌙 Data loaded: {len(data)} bars ✨")
print(f"🚀 Columns: {list(data.columns)}")


class VolumetricBreakout(Strategy):
    # Strategy parameters
    bb_period = 20
    bb_std = 2.0
    vol_period = 20
    vol_mult = 2.0
    rsi_period = 14
    rsi_exit = 70
    atr_period = 14
    max_ext_pct = 0.03  # skip if price >3% above upper band
    risk_pct = 0.02     # 2% risk per trade
    time_stop = 20      # bars

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        # Bollinger Bands
        self.bb_mid = self.I(talib.SMA, close, timeperiod=self.bb_period, name="BB_Mid")
        self.bb_stddev = self.I(talib.STDDEV, close, timeperiod=self.bb_period, nbdev=1, name="BB_StdDev")
        self.bb_upper = self.I(lambda m, s: m + self.bb_std * s, self.bb_mid, self.bb_stddev, name="BB_Upper")
        self.bb_lower = self.I(lambda m, s: m - self.bb_std * s, self.bb_mid, self.bb_stddev, name="BB_Lower")

        # Volume SMA
        self.vol_sma = self.I(talib.SMA, volume, timeperiod=self.vol_period, name="Vol_SMA")

        # RSI
        self.rsi = self.I(talib.RSI, close, timeperiod=self.rsi_period, name="RSI")

        # ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period, name="ATR")

        print("🌙✨ Indicators initialized! BB, Vol, RSI, ATR ready 🚀")

    def next(self):
        price = self.data.Close[-1]

        # ==== ENTRY ====
        if not self.position:
            upper = self.bb_upper[-1]
            mid = self.bb_mid[-1]
            vol = self.data.Volume[-1]
            vol_avg = self.vol_sma[-1]
            atr = self.atr[-1]

            if np.isnan(upper) or np.isnan(vol_avg) or np.isnan(atr):
                return

            breakout = price > upper
            vol_spike = vol > self.vol_mult * vol_avg

            # Avoid chasing extended breakouts
            extended = price > upper * (1 + self.max_ext_pct)

            if breakout and vol_spike and not extended:
                # Risk sizing: 2% equity, stop at tighter of mid-band or 1xATR below entry
                stop_mid = mid
                stop_atr = price - atr
                stop_price = max(stop_mid, stop_atr)  # tighter = higher stop for long
                risk_per_unit = price - stop_price

                if risk_per_unit <= 0:
                    return

                equity = self.equity
                risk_amount = equity * self.risk_pct
                size = int(round(risk_amount / risk_per_unit))

                if size < 1:
                    size = 1

                self.buy(size=size)
                self.entry_bar = len(self.data)
                self.stop_price = stop_price
                print(f"🌙🚀 ENTRY VolumetricBreakout! Price={price:.2f} Upper={upper:.2f} Vol={vol:.2f} vs Avg={vol_avg:.2f} | Size={size} Stop={stop_price:.2f} ✨")

        # ==== EXIT ====
        else:
            rsi_now = self.rsi[-1]
            rsi_prev = self.rsi[-2] if len(self.rsi) > 1 else rsi_now
            mid = self.bb_mid[-1]

            # Momentum exit: RSI crosses below 70
            rsi_cross_down = rsi_prev >= self.rsi_exit and rsi_now < self.rsi_exit

            # Trailing stop along middle band
            trail_stop = mid

            # Time stop
            bars_held = len(self.data) - self.entry_bar
            time_hit = bars_held >= self.time_stop

            # Hard stop check
            low_now = self.data.Low[-1]
            stop_hit = low_now <= self.stop_price

            if stop_hit:
                self.position.close()
                print(f"🛑🌙 STOP HIT at {self.stop_price:.2f}! Price={price:.2f} 🚀")
            elif rsi_cross_down:
                self.position.close()
                print(f"🌙✨ RSI EXIT! RSI {rsi_prev:.1f} -> {rsi_now:.1f} | Price={price:.2f} 🚀")
            elif time_hit:
                self.position.close()
                print(f"⏰🌙 TIME STOP after {bars_held} bars! Price={price:.2f} 🚀")
            else:
                # Update trailing stop to tighter of current stop and mid-band
                if trail_stop > self.stop_price:
                    self.stop_price = trail_stop


print("🌙✨ Running backtest... 🚀")
bt = Backtest(data, VolumetricBreakout, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙✨ Backtest complete! 🚀")