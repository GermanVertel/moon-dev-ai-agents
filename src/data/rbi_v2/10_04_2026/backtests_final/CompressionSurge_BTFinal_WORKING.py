import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Load and prepare data
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

# Ensure datetime index
if 'datetime' in data.columns:
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')

print("🌙 Moon Dev Data Loaded! Shape:", data.shape)
print("✨ Columns:", list(data.columns))
print("🚀 First few rows:\n", data.head())


class CompressionSurge(Strategy):
    # Strategy parameters
    bb_period = 50
    bb_std = 2.0
    rsi_period = 20
    atr_period = 14
    sma_regime_period = 200
    bw_lookback = 100
    bw_percentile = 20
    exit_ratio = 1.5
    partial_exit_ratio = 1.3
    atr_stop_mult = 2.0
    time_stop_bars = 20
    risk_pct = 0.01

    def init(self):
        print("🌙 Initializing CompressionSurge strategy...")

        close = self.data.Close
        high = self.data.High
        low = self.data.Low

        # Bollinger Bands (50-period, 2 std)
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS, close, timeperiod=self.bb_period,
            nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0
        )
        print("✨ Bollinger Bands calculated (50-period, 2 std)")

        # BandWidth
        self.bandwidth = self.I(
            lambda u, m, l: (u - l) / m, self.bb_upper, self.bb_middle, self.bb_lower
        )
        print("🚀 BandWidth calculated")

        # BandWidth rolling percentile threshold
        self.bw_threshold = self.I(
            lambda bw: pd.Series(bw).rolling(self.bw_lookback).quantile(self.bw_percentile / 100.0).values,
            self.bandwidth
        )
        print("🌙 BandWidth percentile threshold calculated")

        # RSI (20-period)
        self.rsi = self.I(talib.RSI, close, timeperiod=self.rsi_period)
        print("✨ RSI(20) calculated")

        # ATR (14-period)
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)
        print("🚀 ATR(14) calculated")

        # 200-period SMA regime filter
        self.sma_regime = self.I(talib.SMA, close, timeperiod=self.sma_regime_period)
        print("🌙 200-period SMA regime filter calculated")

        # RSI / UpperBand ratio
        self.rsi_ratio = self.I(
            lambda r, u: r / u, self.rsi, self.bb_upper
        )
        print("✨ RSI/UpperBand ratio calculated")

        # Track entry bar for time stop
        self.entry_bar = None
        self.partial_taken = False

    def next(self):
        price = self.data.Close[-1]

        # Skip if indicators not ready
        if (np.isnan(self.bw_threshold[-1]) or np.isnan(self.sma_regime[-1])
                or np.isnan(self.rsi_ratio[-1]) or np.isnan(self.atr[-1])):
            return

        # Manage open position
        if self.position:
            bars_held = len(self.data) - 1 - self.entry_bar

            # Primary exit: RSI/UpperBand ratio > 1.5
            if self.rsi_ratio[-1] > self.exit_ratio:
                print(f"🌙 EXIT (ratio {self.rsi_ratio[-1]:.3f} > {self.exit_ratio}) at {price:.2f} 🚀")
                self.position.close()
                self.entry_bar = None
                self.partial_taken = False
                return

            # Partial exit at ratio 1.3
            if not self.partial_taken and self.rsi_ratio[-1] > self.partial_exit_ratio:
                print(f"✨ PARTIAL EXIT (ratio {self.rsi_ratio[-1]:.3f} > {self.partial_exit_ratio}) at {price:.2f} 🌙")
                self.position.close(portion=0.5)
                self.partial_taken = True

            # Secondary exit: close below middle band (trend failure)
            if price < self.bb_middle[-1]:
                print(f"🚀 EXIT (close {price:.2f} < middle band {self.bb_middle[-1]:.2f}) trend failure 🌙")
                self.position.close()
                self.entry_bar = None
                self.partial_taken = False
                return

            # Time stop
            if bars_held >= self.time_stop_bars:
                print(f"✨ EXIT (time stop {bars_held} bars) at {price:.2f} 🌙")
                self.position.close()
                self.entry_bar = None
                self.partial_taken = False
                return

            return

        # ENTRY LOGIC (long only)
        # Regime filter: price > 200 SMA
        if price <= self.sma_regime[-1]:
            return

        # Condition A: Compression - BandWidth below threshold
        compression = self.bandwidth[-1] < self.bw_threshold[-1]

        # Condition B: Breakout confirmation - close above middle band
        breakout = price > self.bb_middle[-1]

        # Condition C: RSI > 50
        momentum = self.rsi[-1] > 50

        # Optional: BandWidth expanding
        expanding = self.bandwidth[-1] > self.bandwidth[-2] if len(self.bandwidth) > 1 else False

        if compression and breakout and momentum and expanding:
            # Risk-based position sizing
            stop_distance = self.atr_stop_mult * self.atr[-1]
            if stop_distance <= 0:
                return

            risk_amount = self.equity * self.risk_pct
            position_size = int(round(risk_amount / stop_distance))
            if position_size <= 0:
                position_size = 1

            # Cap size to affordable amount
            max_size = int(self.equity / price)
            position_size = min(position_size, max_size)
            if position_size <= 0:
                return

            stop_price = price - stop_distance
            print(f"🌙 ENTRY SIGNAL! Compression + Breakout + Momentum 🚀")
            print(f"   Price: {price:.2f} | Size: {position_size} | Stop: {stop_price:.2f}")
            print(f"   BandWidth: {self.bandwidth[-1]:.5f} < Threshold: {self.bw_threshold[-1]:.5f}")
            print(f"   RSI: {self.rsi[-1]:.2f} | Ratio: {self.rsi_ratio[-1]:.3f}")

            self.buy(size=position_size, sl=stop_price)
            self.entry_bar = len(self.data) - 1
            self.partial_taken = False


# Run backtest
bt = Backtest(data, CompressionSurge, cash=1_000_000, commission=0.002)
stats = bt.run()
print(stats)
print(stats._strategy)