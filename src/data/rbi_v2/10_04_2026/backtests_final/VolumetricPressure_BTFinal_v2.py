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
    'volume': 'Volume',
})

# Ensure datetime index
if 'datetime' in data.columns:
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')

print("🌙✨ Moon Dev VolumetricPressure Backtest Starting! 🚀🚀")
print(f"📊 Data loaded: {len(data)} bars")
print(f"📅 Range: {data.index[0]} to {data.index[-1]}")


class VolumetricPressure(Strategy):
    # Strategy parameters
    donchian_period = 20
    volume_ma_period = 20
    volume_multiplier = 1.5
    atr_period = 14
    atr_stop_multiplier = 1.5
    atr_trail_multiplier = 2.0
    ema_period = 200
    risk_pct = 0.02  # 2% risk per trade
    time_exit_bars = 10

    def init(self):
        print("🌙 Initializing indicators... ✨")
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        # Donchian Channel (highest high / lowest low of last N bars)
        self.highest_high = self.I(talib.MAX, high, timeperiod=self.donchian_period)
        self.lowest_low = self.I(talib.MIN, low, timeperiod=self.donchian_period)

        # Volume MA
        self.volume_ma = self.I(talib.SMA, volume, timeperiod=self.volume_ma_period)

        # ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        # EMA trend filter
        self.ema = self.I(talib.EMA, close, timeperiod=self.ema_period)

        # Track entry bar for time-based exit
        self.entry_bar = None
        print("🚀 Indicators ready! Let's fly to the moon! 🌙")

    def next(self):
        # Need enough bars for indicators
        if len(self.data) < max(self.ema_period, self.donchian_period, self.atr_period, self.volume_ma_period) + 2:
            return

        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        vol = self.data.Volume[-1]

        # Skip if indicators not ready
        if (np.isnan(self.highest_high[-1]) or np.isnan(self.lowest_low[-1]) or
                np.isnan(self.volume_ma[-1]) or np.isnan(self.atr[-1]) or
                np.isnan(self.ema[-1]) or self.atr[-1] <= 0):
            return

        # Volume confirmation
        volume_surge = vol > (self.volume_multiplier * self.volume_ma[-1])

        # Breakout levels (prior bar's donchian to avoid self-reference)
        prior_high = self.highest_high[-2]
        prior_low = self.lowest_low[-2]

        if np.isnan(prior_high) or np.isnan(prior_low):
            return

        # --- Manage open position ---
        if self.position:
            bars_in_trade = len(self.data) - 1 - self.entry_bar if self.entry_bar is not None else 0

            if self.position.is_long:
                # ATR trailing stop
                trail_stop = price - (self.atr_trail_multiplier * self.atr[-1])
                if low <= trail_stop:
                    print(f"🌙💥 LONG trailing stop hit at {trail_stop:.2f} | Price: {price:.2f}")
                    self.position.close()
                    self.entry_bar = None
                    return

                # Time-based exit
                if bars_in_trade >= self.time_exit_bars:
                    entry_price = self.trades[-1].entry_price if len(self.trades) > 0 else price
                    if price < entry_price:
                        print(f"⏰ LONG time exit after {bars_in_trade} bars | Price: {price:.2f}")
                        self.position.close()
                        self.entry_bar = None
                        return

            elif self.position.is_short:
                trail_stop = price + (self.atr_trail_multiplier * self.atr[-1])
                if high >= trail_stop:
                    print(f"🌙💥 SHORT trailing stop hit at {trail_stop:.2f} | Price: {price:.2f}")
                    self.position.close()
                    self.entry_bar = None
                    return

                if bars_in_trade >= self.time_exit_bars:
                    entry_price = self.trades[-1].entry_price if len(self.trades) > 0 else price
                    if price > entry_price:
                        print(f"⏰ SHORT time exit after {bars_in_trade} bars | Price: {price:.2f}")
                        self.position.close()
                        self.entry_bar = None
                        return

            return  # Don't open new trades while in one

        # --- Entry logic ---
        # Long entry
        long_breakout = price > prior_high
        long_trend = price > self.ema[-1]
        if long_breakout and volume_surge and long_trend:
            sl = price - (self.atr_stop_multiplier * self.atr[-1])
            risk_per_unit = price - sl
            if risk_per_unit > 0:
                risk_amount = self.equity * self.risk_pct
                # Fractional sizing: fraction of equity (must be 0 < size < 1)
                position_size = risk_amount / risk_per_unit
                # Cap at 95% of equity as fraction
                max_size_fraction = 0.95
                position_size = min(position_size / self.equity, max_size_fraction)
                if 0 < position_size < 1:
                    print(f"🌙🚀 LONG BREAKOUT! Price: {price:.2f} | Prior High: {prior_high:.2f} | Vol: {vol:.2f} vs MA: {self.volume_ma[-1]:.2f} | SL: {sl:.2f} | Size: {position_size:.6f}")
                    self.buy(size=position_size, sl=sl)
                    self.entry_bar = len(self.data) - 1
                    return

        # Short entry
        short_breakout = price < prior_low
        short_trend = price < self.ema[-1]
        if short_breakout and volume_surge and short_trend:
            sl = price + (self.atr_stop_multiplier * self.atr[-1])
            risk_per_unit = sl - price
            if risk_per_unit > 0:
                risk_amount = self.equity * self.risk_pct
                position_size = risk_amount / risk_per_unit
                max_size_fraction = 0.95
                position_size = min(position_size / self.equity, max_size_fraction)
                if 0 < position_size < 1:
                    print(f"🌙🔻 SHORT BREAKOUT! Price: {price:.2f} | Prior Low: {prior_low:.2f} | Vol: {vol:.2f} vs MA: {self.volume_ma[-1]:.2f} | SL: {sl:.2f} | Size: {position_size:.6f}")
                    self.sell(size=position_size, sl=sl)
                    self.entry_bar = len(self.data) - 1
                    return


print("🌙 Building backtest engine... 🚀")
bt = Backtest(
    data,
    VolumetricPressure,
    cash=1_000_000,
    commission=0.001,
    exclusive_orders=True,
)

print("✨ Running VolumetricPressure backtest... 🌙")
stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙🚀 Moon Dev backtest complete! To the moon! ✨")