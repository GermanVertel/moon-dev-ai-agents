import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Load data
data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
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

# Ensure all OHLCV columns are float64 (talib requires double)
for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    data[col] = data[col].astype(np.float64)

print("🌙 Moon Dev LiquidationVolatility Backtest Loading...")
print(f"✨ Data shape: {data.shape}")
print(f"🚀 Date range: {data.index[0]} to {data.index[-1]}")


class LiquidationVolatility(Strategy):
    # Strategy parameters
    atr_period = 14
    high_period = 20
    low_period = 10
    rsi_period = 14
    atr_mult_entry = 2.0
    atr_mult_crash = 3.0
    atr_mult_stop = 1.5
    atr_mult_target = 1.0
    rsi_oversold = 30
    volume_mult = 1.5
    atr_avg_period = 20
    risk_pct = 0.02
    time_stop_bars = 8
    trail_trigger = 0.5
    trail_distance = 0.5

    def init(self):
        print("🌙 Initializing Moon Dev indicators...")
        # Wrap price/volume arrays as float64 to satisfy talib's double requirement
        high = np.asarray(self.data.High, dtype=np.float64)
        low = np.asarray(self.data.Low, dtype=np.float64)
        close = np.asarray(self.data.Close, dtype=np.float64)
        volume = np.asarray(self.data.Volume, dtype=np.float64)

        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)
        self.atr_avg = self.I(talib.SMA, self.atr, timeperiod=self.atr_avg_period)
        self.high20 = self.I(talib.MAX, high, timeperiod=self.high_period)
        self.low10 = self.I(talib.MIN, low, timeperiod=self.low_period)
        self.rsi = self.I(talib.RSI, close, timeperiod=self.rsi_period)
        self.vol_avg = self.I(talib.SMA, volume, timeperiod=20)
        self.sma200 = self.I(talib.SMA, close, timeperiod=200)
        print("✨ Indicators ready! 🚀")

    def next(self):
        price = self.data.Close[-1]

        # Skip if indicators not ready
        if (np.isnan(self.atr[-1]) or np.isnan(self.atr_avg[-1]) or
                np.isnan(self.high20[-1]) or np.isnan(self.low10[-1]) or
                np.isnan(self.rsi[-1]) or np.isnan(self.vol_avg[-1]) or
                np.isnan(self.sma200[-1])):
            return

        atr = self.atr[-1]
        high20 = self.high20[-1]
        low10 = self.low10[-1]

        # Manage open positions
        if self.position:
            entry = self.trades[-1].entry_price
            bars_held = len(self.data) - self.trades[-1].entry_bar

            # Trailing stop logic
            if price >= entry + self.trail_trigger * atr:
                new_stop = price - self.trail_distance * atr
                if self.trades[-1].sl is None or new_stop > self.trades[-1].sl:
                    self.trades[-1].sl = new_stop
                    print(f"🌙 Trailing stop moved to {new_stop:.2f} 🚀")

            # Primary target exit
            target = low10 + self.atr_mult_target * atr
            if price >= target:
                print(f"✨ TARGET HIT! Exit at {price:.2f} (target {target:.2f}) 🌙")
                self.position.close()
                return

            # Time stop
            if bars_held >= self.time_stop_bars:
                print(f"⏰ Time stop hit after {bars_held} bars at {price:.2f} 🌙")
                self.position.close()
                return

            return

        # Entry logic
        entry_trigger = high20 - self.atr_mult_entry * atr
        crash_floor = high20 - self.atr_mult_crash * atr

        # Regime filter: high volatility + not in bear trend
        vol_regime = atr > self.atr_avg[-1]
        bull_regime = price > self.sma200[-1]

        # Confirmation filters
        rsi_ok = self.rsi[-1] < self.rsi_oversold
        vol_spike = self.data.Volume[-1] > self.volume_mult * self.vol_avg[-1]

        # Avoid falling knives
        not_crash = price > crash_floor

        if (price <= entry_trigger and vol_regime and bull_regime and
                rsi_ok and vol_spike and not_crash):
            stop = price - self.atr_mult_stop * atr
            risk_per_unit = price - stop
            if risk_per_unit <= 0:
                return

            risk_amount = self.equity * self.risk_pct
            size = int(round(risk_amount / risk_per_unit))
            size = min(size, int(self.equity / price))
            if size <= 0:
                return

            print(f"🚀 LIQUIDATION BOUNCE ENTRY! Price={price:.2f} | "
                  f"Trigger={entry_trigger:.2f} | RSI={self.rsi[-1]:.1f} | "
                  f"VolSpike={self.data.Volume[-1]:.2f} | Size={size} 🌙")
            self.buy(size=size, sl=stop)


bt = Backtest(data, LiquidationVolatility, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)