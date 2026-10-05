import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Load data
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')

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

# Set datetime index
if 'datetime' in data.columns:
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')
elif 'date' in data.columns:
    data['date'] = pd.to_datetime(data['date'])
    data = data.set_index('date')

# Ensure index is datetime
if not isinstance(data.index, pd.DatetimeIndex):
    data.index = pd.to_datetime(data.index)

# Ensure required columns exist
required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
for col in required_cols:
    if col not in data.columns:
        raise ValueError(f"Missing required column: {col}")

# Drop any remaining NaN rows
data = data.dropna()

print("🌙✨ NocturnalSectorPulse Backtest Initializing... 🚀")
print(f"📊 Data loaded: {len(data)} rows")
print(f"📅 Date range: {data.index[0]} to {data.index[-1]}")


class NocturnalSectorPulse(Strategy):
    """
    🌙 NocturnalSectorPulse Strategy 🌙
    """

    high_period = 4
    atr_period = 14
    ma_period = 50
    erod_window = 20
    risk_pct = 0.01
    max_stop_pct = 0.05
    atr_mult = 1.5
    rr_ratio = 3.0

    def init(self):
        print("🌙 Initializing indicators...")

        self.rolling_high = self.I(talib.MAX, self.data.High, timeperiod=self.high_period)
        self.rolling_low_30 = self.I(talib.MIN, self.data.Low, timeperiod=2)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period)
        self.ma50 = self.I(talib.SMA, self.data.Close, timeperiod=self.ma_period)

        short_w = self.erod_window
        long_w = self.erod_window * 2

        def _erod(x):
            arr = np.asarray(x, dtype=float)
            n = len(arr)
            short_ret = np.full(n, np.nan, dtype=float)
            long_ret = np.full(n, np.nan, dtype=float)
            if n > short_w:
                short_ret[short_w:] = (arr[short_w:] - arr[:-short_w]) / arr[:-short_w]
            if n > long_w:
                long_ret[long_w:] = (arr[long_w:] - arr[:-long_w]) / arr[:-long_w]
            erod = short_ret - long_ret
            erod = np.where(np.isnan(erod), 0.0, erod)
            return erod

        self.erod = self.I(_erod, self.data.Close, name="EROD")

        print("✨ Indicators ready: rolling_high, atr, ma50, erod 🌙")

    def next(self):
        price = self.data.Close[-1]
        high_60 = self.rolling_high[-1]
        atr_val = self.atr[-1]
        ma50_val = self.ma50[-1]
        erod_val = self.erod[-1]

        if np.isnan(high_60) or np.isnan(atr_val) or np.isnan(ma50_val):
            return

        try:
            hour = self.data.index[-1].hour
        except Exception:
            hour = 0

        is_overnight = (hour >= 18) or (hour < 9)

        # ---- ENTRY LOGIC ----
        if not self.position and is_overnight:
            breakout = price > high_60
            erod_positive = erod_val > 0
            sector_uptrend = price > ma50_val

            if breakout and erod_positive and sector_uptrend:
                atr_stop_dist = self.atr_mult * atr_val
                max_stop_dist = self.max_stop_pct * price
                stop_dist = min(atr_stop_dist, max_stop_dist)

                stop_price = price - stop_dist
                tp_price = price + (stop_dist * self.rr_ratio)

                # Position sizing: risk_pct as fraction of equity
                # size = risk_amount / stop_dist / price  -> fraction of equity
                if stop_dist > 0 and price > 0:
                    size = self.risk_pct * price / stop_dist / price
                else:
                    size = 0.10

                # Clamp to valid fraction (0 < size < 1)
                if size <= 0.01:
                    size = 0.01
                if size >= 1:
                    size = 0.95

                print(f"🚀🌙 MOON DEV ENTRY SIGNAL! 🚀")
                print(f"   Price: {price:.2f} | 60m High: {high_60:.2f}")
                print(f"   EROD: {erod_val:.4f} | MA50: {ma50_val:.2f}")
                print(f"   Stop: {stop_price:.2f} | TP: {tp_price:.2f}")
                print(f"   Size (fraction): {size:.4f}")

                self.buy(size=size, sl=stop_price, tp=tp_price)

        # ---- EXIT LOGIC ----
        elif self.position:
            if not is_overnight:
                print(f"🌅 Session ended - closing position at {price:.2f}")
                self.position.close()
                return

            low_30 = self.rolling_low_30[-1]
            if not np.isnan(low_30) and price < low_30:
                print(f"📉 30-min low breach - trailing exit at {price:.2f}")
                self.position.close()


print("🌙✨ Launching NocturnalSectorPulse backtest... 🚀")
bt = Backtest(
    data,
    NocturnalSectorPulse,
    cash=1_000_000,
    commission=0.002
)

stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙 Moon Dev Backtest Complete! ✨")