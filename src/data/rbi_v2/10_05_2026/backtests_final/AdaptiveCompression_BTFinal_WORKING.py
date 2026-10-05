import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# 🌙 Moon Dev's AdaptiveCompression Strategy 🚀

class AdaptiveCompression(Strategy):
    # Strategy parameters
    atr_period = 20
    atr_percentile_lookback = 252
    atr_percentile_threshold = 10  # 10th percentile
    sma_period = 50
    atr_stop_mult = 2.0
    time_exit_days = 5
    risk_per_trade = 0.01  # 1% of equity

    def init(self):
        print("🌙 Moon Dev initializing AdaptiveCompression strategy...")

        # ATR(20)
        self.atr = self.I(talib.ATR,
                          self.data.High, self.data.Low, self.data.Close,
                          timeperiod=self.atr_period)
        print("✨ ATR(20) calculated")

        # SMA(50)
        self.sma = self.I(talib.SMA, self.data.Close, timeperiod=self.sma_period)
        print("✨ SMA(50) calculated")

        # Rolling 10th percentile of ATR over lookback
        atr_series = pd.Series(self.atr)
        self.atr_pct = self.I(
            lambda x: pd.Series(x).rolling(self.atr_percentile_lookback).quantile(
                self.atr_percentile_threshold / 100.0).values,
            atr_series
        )
        print("✨ ATR 10th percentile calculated")

        # SMA slope (rising/flat filter)
        self.sma_slope = self.I(
            lambda x: pd.Series(x).diff(5).values,
            self.sma
        )
        print("✨ SMA slope calculated")

        # Track entry bar for time exit
        self.entry_bar = None
        self.entry_price = None
        self.stop_price = None

    def next(self):
        # Skip if indicators not ready
        if (np.isnan(self.atr[-1]) or np.isnan(self.sma[-1]) or
                np.isnan(self.atr_pct[-1]) or np.isnan(self.sma_slope[-1])):
            return

        price = self.data.Close[-1]

        # ============ EXIT LOGIC ============
        if self.position:
            # Stop-loss exit
            if self.data.Low[-1] <= self.stop_price:
                print(f"🛑 Moon Dev STOP-LOSS hit at {self.stop_price:.2f} | Price: {price:.2f}")
                self.position.close()
                self.entry_bar = None
                return

            # Time-based exit
            bars_held = len(self.data) - 1 - self.entry_bar
            if bars_held >= self.time_exit_days:
                print(f"⏰ Moon Dev TIME EXIT after {bars_held} bars | Price: {price:.2f}")
                self.position.close()
                self.entry_bar = None
                return

        # ============ ENTRY LOGIC ============
        if not self.position:
            # Condition A: Compression regime (ATR below 10th percentile)
            compression = self.atr[-1] < self.atr_pct[-1]

            # Condition B: Price crosses above SMA(50) - Moon Dev manual crossover 🌙
            cross_above = (self.data.Close[-2] < self.sma[-2] and
                           self.data.Close[-1] > self.sma[-1])

            # Optional filter: SMA slope flat or rising
            slope_ok = self.sma_slope[-1] >= 0

            if compression and cross_above and slope_ok:
                # Risk-based position sizing
                equity = self.equity
                risk_amount = equity * self.risk_per_trade
                stop_distance = self.atr_stop_mult * self.atr[-1]

                if stop_distance <= 0:
                    return

                position_size = int(round(risk_amount / stop_distance))

                if position_size <= 0:
                    return

                # Cap position size to avoid absurd values
                max_size = int(equity / price)
                position_size = min(position_size, max_size)

                if position_size <= 0:
                    return

                self.stop_price = price - stop_distance
                self.entry_bar = len(self.data) - 1
                self.entry_price = price

                print(f"🚀 Moon Dev ENTRY SIGNAL! 🌙")
                print(f"   Price: {price:.2f} | SMA: {self.sma[-1]:.2f}")
                print(f"   ATR: {self.atr[-1]:.2f} | ATR 10th pct: {self.atr_pct[-1]:.2f}")
                print(f"   Stop: {self.stop_price:.2f} | Size: {position_size}")
                print(f"   Compression: {compression} | Cross: {cross_above} | Slope: {slope_ok}")

                self.buy(size=position_size)


# ============ DATA LOADING & BACKTEST ============
print("🌙 Moon Dev loading data...")
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

print(f"✨ Data loaded: {len(data)} bars")
print(f"✨ Columns: {list(data.columns)}")

# Run backtest
print("🚀 Moon Dev launching backtest...")
bt = Backtest(data, AdaptiveCompression, cash=1_000_000, commission=0.001)
stats = bt.run()

print("\n" + "=" * 60)
print("🌙 MOON DEV ADAPTIVECOMPRESSION BACKTEST RESULTS 🚀")
print("=" * 60)
print(stats)
print("\n" + "=" * 60)
print("📊 STRATEGY DETAILS")
print("=" * 60)
print(stats._strategy)