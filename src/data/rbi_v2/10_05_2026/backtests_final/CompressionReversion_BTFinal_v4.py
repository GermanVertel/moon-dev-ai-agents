import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# ============================================================
# 🌙 MOON DEV - COMPRESSION REVERSION STRATEGY 🌙
# ============================================================

print("🌙 Moon Dev Backtest AI initializing...")
print("🚀 Loading CompressionReversion strategy...")

# Load data
data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
print(f"📂 Loading data from: {data_path}")

data = pd.read_csv(data_path)

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Set datetime index BEFORE renaming
if 'datetime' in data.columns:
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')

# Ensure proper column mapping
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

print(f"✅ Data loaded: {len(data)} bars")
print(f"📊 Columns: {list(data.columns)}")


class CompressionReversion(Strategy):
    """
    🌙 CompressionReversion Strategy 🌙

    Volatility compression → expansion → mean reversion
    """

    # Strategy parameters
    bb_period = 20
    bb_std = 2.0
    atr_period = 14
    squeeze_lookback = 20
    atr_multiplier = 2.0
    stop_atr_mult = 1.5
    time_exit_bars = 12
    risk_pct = 0.01

    def init(self):
        print("🌙 Initializing indicators...")

        close = self.data.Close
        high = self.data.High
        low = self.data.Low

        # Bollinger Bands
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS, close, timeperiod=self.bb_period,
            nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0
        )

        # Band Width = (Upper - Lower) / Middle
        self.band_width = self.I(
            lambda u, l, m: (u - l) / m,
            self.bb_upper, self.bb_lower, self.bb_middle
        )

        # ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        # Squeeze detection: current band width is the lowest in lookback
        self.bw_min = self.I(talib.MIN, self.band_width, timeperiod=self.squeeze_lookback)

        print("✨ Indicators ready!")

    def next(self):
        # Skip if insufficient data
        if len(self.data) < self.bb_period + 5:
            return

        price = self.data.Close[-1]
        upper = self.bb_upper[-1]
        lower = self.bb_lower[-1]
        middle = self.bb_middle[-1]
        atr = self.atr[-1]
        bw = self.band_width[-1]
        bw_min = self.bw_min[-1]

        if np.isnan(atr) or np.isnan(middle) or atr <= 0:
            return
        if np.isnan(upper) or np.isnan(lower) or np.isnan(bw) or np.isnan(bw_min):
            return

        # Check squeeze condition
        is_squeeze = (bw <= bw_min * 1.001)  # near the low

        # Check if band width has been declining over prior bars
        declining = False
        if len(self.band_width) > 6:
            recent_bw = [self.band_width[-i] for i in range(1, 6)]
            if all(not np.isnan(x) for x in recent_bw):
                declining = recent_bw[0] < recent_bw[-1]

        # Manage open position
        if self.position:
            try:
                bars_held = len(self.data) - self.trades[-1].entry_bar
            except Exception:
                bars_held = 0

            if self.position.is_long:
                # Exit at middle band (mean reversion target)
                if price >= middle:
                    print(f"🌙 LONG exit at MEAN: {price:.2f} >= {middle:.2f} | 🎯 Target hit!")
                    self.position.close()
                    return
                # Secondary exit: close back inside opposite band
                if price <= lower:
                    print(f"🚨 LONG exit at OPPOSITE band: {price:.2f}")
                    self.position.close()
                    return
                # Time-based exit
                if bars_held >= self.time_exit_bars:
                    print(f"⏰ LONG time exit after {bars_held} bars")
                    self.position.close()
                    return

            elif self.position.is_short:
                # Exit at middle band
                if price <= middle:
                    print(f"🌙 SHORT exit at MEAN: {price:.2f} <= {middle:.2f} | 🎯 Target hit!")
                    self.position.close()
                    return
                # Secondary exit: close back inside opposite band
                if price >= upper:
                    print(f"🚨 SHORT exit at OPPOSITE band: {price:.2f}")
                    self.position.close()
                    return
                # Time-based exit
                if bars_held >= self.time_exit_bars:
                    print(f"⏰ SHORT time exit after {bars_held} bars")
                    self.position.close()
                    return
            return

        # Entry logic - only if squeeze was active recently
        if not is_squeeze:
            return

        # Long entry: close above upper band with ATR confirmation
        if price > upper:
            distance = price - upper
            candle_range = self.data.High[-1] - self.data.Low[-1]
            if distance > self.atr_multiplier * atr or candle_range > self.atr_multiplier * atr:
                # Position sizing based on risk
                stop_price = upper - self.stop_atr_mult * atr
                risk_per_unit = price - stop_price
                if risk_per_unit <= 0:
                    return
                risk_amount = self.equity * self.risk_pct
                size = int(round(risk_amount / risk_per_unit))
                max_size = int(self.equity / price)
                if max_size < 1:
                    return
                size = min(size, max_size)
                if size < 1:
                    size = 1

                print(f"🚀 LONG ENTRY | Price: {price:.2f} > Upper: {upper:.2f} | ATR: {atr:.2f} | Size: {size}")
                self.buy(size=size, sl=stop_price, tp=middle)

        # Short entry: close below lower band with ATR confirmation
        elif price < lower:
            distance = lower - price
            candle_range = self.data.High[-1] - self.data.Low[-1]
            if distance > self.atr_multiplier * atr or candle_range > self.atr_multiplier * atr:
                stop_price = lower + self.stop_atr_mult * atr
                risk_per_unit = stop_price - price
                if risk_per_unit <= 0:
                    return
                risk_amount = self.equity * self.risk_pct
                size = int(round(risk_amount / risk_per_unit))
                max_size = int(self.equity / price)
                if max_size < 1:
                    return
                size = min(size, max_size)
                if size < 1:
                    size = 1

                print(f"🔻 SHORT ENTRY | Price: {price:.2f} < Lower: {lower:.2f} | ATR: {atr:.2f} | Size: {size}")
                self.sell(size=size, sl=stop_price, tp=middle)


print("🌙 Setting up backtest...")
bt = Backtest(
    data,
    CompressionReversion,
    cash=1_000_000,
    commission=0.002,
    exclusive_orders=True
)

print("🚀 Running backtest...")
stats = bt.run()

print("\n" + "="*60)
print("🌙 MOON DEV - COMPRESSION REVERSION RESULTS 🌙")
print("="*60)
print(stats)
print("\n" + "="*60)
print("📊 STRATEGY DETAILS")
print("="*60)
print(stats._strategy)
print("="*60)
print("✨ Backtest complete! Moon Dev out! 🌙")