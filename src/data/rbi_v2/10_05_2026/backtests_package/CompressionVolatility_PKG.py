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

# Ensure proper column mapping
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

print("🌙 Moon Dev Data Loaded! Shape:", data.shape)
print("✨ Columns:", list(data.columns))


class CompressionVolatility(Strategy):
    """
    🌙 CompressionVolatility Strategy 🚀
    Captures explosive breakouts after volatility squeezes.
    """

    # Strategy parameters
    bb_period = 20
    bb_std = 2.0
    vol_period = 20
    vol_percentile = 20  # 20th percentile threshold
    range_avg_period = 5
    stop_pct = 0.5       # 50% of breakout candle range
    trail_mult = 1.0     # Trail by 1x breakout range
    time_exit_bars = 10
    risk_pct = 0.02      # 2% risk per trade
    volume_mult = 1.0    # Volume filter multiplier

    def init(self):
        print("🌙 Initializing CompressionVolatility indicators...")

        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        volume = pd.Series(self.data.Volume)

        # Bollinger Bands - use talib SMA and STDDEV
        self.bb_mid = self.I(talib.SMA, close, timeperiod=self.bb_period)
        std = self.I(talib.STDDEV, close, timeperiod=self.bb_period, nbdev=1)

        # Manually compute bands from mid and std
        self.bb_upper = self.I(lambda: np.array(self.bb_mid) + self.bb_std * np.array(std))
        self.bb_lower = self.I(lambda: np.array(self.bb_mid) - self.bb_std * np.array(std))

        # Bandwidth
        self.bandwidth = self.I(
            lambda: (np.array(self.bb_upper) - np.array(self.bb_lower)) / np.array(self.bb_mid)
        )

        # Historical volatility (std of returns)
        returns = close.pct_change()
        self.hv = self.I(lambda: returns.rolling(self.vol_period).std().values)

        # Rolling percentile threshold of bandwidth
        def bw_threshold():
            bw = pd.Series(self.bandwidth)
            return bw.rolling(self.vol_period).quantile(self.vol_percentile / 100.0).values

        self.bw_thresh = self.I(bw_threshold)

        # Average candle range
        candle_range = high - low
        self.avg_range = self.I(lambda: candle_range.rolling(self.range_avg_period).mean().values)

        # Average volume
        self.avg_volume = self.I(talib.SMA, volume, timeperiod=self.range_avg_period)

        # ATR for reference
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=14)

        print("✨ Indicators ready! Let's hunt for squeezes... 🎯")

    def next(self):
        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        volume = self.data.Volume[-1]

        upper = self.bb_upper[-1]
        lower = self.bb_lower[-1]
        bw = self.bandwidth[-1]
        bw_thresh = self.bw_thresh[-1]
        avg_range = self.avg_range[-1]
        avg_vol = self.avg_volume[-1]

        # Skip if indicators not ready
        if np.isnan(bw) or np.isnan(bw_thresh) or np.isnan(avg_range) or np.isnan(avg_vol):
            return

        # Manage existing position (trailing stop + time exit)
        if self.position:
            entry = self.position.entry_price
            candle_range = self.data.High[-2] - self.data.Low[-2] if len(self.data) > 1 else 0
            if candle_range <= 0:
                candle_range = avg_range

            trail_dist = self.trail_mult * candle_range

            if self.position.is_long:
                # Move stop to breakeven once price moves 1x range
                new_stop = entry - self.stop_pct * candle_range
                if price >= entry + candle_range:
                    new_stop = max(new_stop, entry)
                if price >= entry + 2 * candle_range:
                    new_stop = max(new_stop, entry + candle_range)

                # Time-based exit
                if len(self.trades) > 0:
                    bars_held = len(self.data) - self.trades[-1].entry_bar
                    if bars_held >= self.time_exit_bars and price < entry + candle_range:
                        print(f"⏰ Moon Dev time exit LONG at {price:.2f}")
                        self.position.close()
                        return

                # Check stop hit
                if low <= new_stop:
                    print(f"🛑 Moon Dev trailing stop LONG hit at {new_stop:.2f}")
                    self.position.close()
                    return

            elif self.position.is_short:
                new_stop = entry + self.stop_pct * candle_range
                if price <= entry - candle_range:
                    new_stop = min(new_stop, entry)
                if price <= entry - 2 * candle_range:
                    new_stop = min(new_stop, entry - candle_range)

                if len(self.trades) > 0:
                    bars_held = len(self.data) - self.trades[-1].entry_bar
                    if bars_held >= self.time_exit_bars and price > entry - candle_range:
                        print(f"⏰ Moon Dev time exit SHORT at {price:.2f}")
                        self.position.close()
                        return

                if high >= new_stop:
                    print(f"🛑 Moon Dev trailing stop SHORT hit at {new_stop:.2f}")
                    self.position.close()
                    return

            return  # already in a trade, don't add more

        # Squeeze detection
        squeeze = bw < bw_thresh
        if not squeeze:
            return

        # Breakout candle
        candle_range = high - low
        expanded_range = candle_range > avg_range
        above_avg_vol = volume > avg_vol * self.volume_mult

        # Long entry
        if price > upper and expanded_range and above_avg_vol:
            stop_price = price - self.stop_pct * candle_range
            risk_per_unit = price - stop_price
            if risk_per_unit <= 0:
                return
            risk_amount = self.equity * self.risk_pct
            size = int(round(risk_amount / risk_per_unit))
            if size <= 0:
                size = 1
            print(f"🚀 Moon Dev LONG BREAKOUT! Price={price:.2f} Upper={upper:.2f} "
                  f"Range={candle_range:.2f} Size={size} 🌙")
            self.buy(size=size)

        # Short entry
        elif price < lower and expanded_range and above_avg_vol:
            stop_price = price + self.stop_pct * candle_range
            risk_per_unit = stop_price - price
            if risk_per_unit <= 0:
                return
            risk_amount = self.equity * self.risk_pct
            size = int(round(risk_amount / risk_per_unit))
            if size <= 0:
                size = 1
            print(f"🔻 Moon Dev SHORT BREAKOUT! Price={price:.2f} Lower={lower:.2f} "
                  f"Range={candle_range:.2f} Size={size} 🌙")
            self.sell(size=size)


# Run backtest
bt = Backtest(data, CompressionVolatility, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)