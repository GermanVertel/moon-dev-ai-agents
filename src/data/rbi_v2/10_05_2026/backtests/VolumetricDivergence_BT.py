import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Load data
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Map columns to proper case
data = data.rename(columns={
    'datetime': 'Datetime',
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

data['Datetime'] = pd.to_datetime(data['Datetime'])
data = data.set_index('Datetime')

print("🌙✨ Moon Dev VolumetricDivergence Backtest Loading... 🚀")
print(f"📊 Data shape: {data.shape}")
print(f"📅 Date range: {data.index[0]} to {data.index[-1]}")


class VolumetricDivergence(Strategy):
    # Strategy parameters
    stoch_period = 14
    stoch_smooth_k = 3
    stoch_smooth_d = 3
    volume_ma_period = 20
    volume_multiplier = 1.5
    swing_lookback = 20
    atr_period = 14
    risk_pct = 0.01
    rr_ratio = 2.0
    time_exit_bars = 5

    def init(self):
        print("🌙 Initializing Moon Dev VolumetricDivergence indicators...")

        # Stochastic Oscillator (14, 3, 3)
        self.stoch_k, self.stoch_d = self.I(
            talib.STOCH,
            self.data.High,
            self.data.Low,
            self.data.Close,
            fastk_period=self.stoch_period,
            slowk_period=self.stoch_smooth_k,
            slowk_matype=0,
            slowd_period=self.stoch_smooth_d,
            slowd_matype=0
        )

        # Volume Moving Average (20-period)
        self.vol_ma = self.I(talib.SMA, self.data.Volume, timeperiod=self.volume_ma_period)

        # Swing highs/lows
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=self.swing_lookback)
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=self.swing_lookback)

        # ATR for dynamic stops
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period)

        # Track bars in trade
        self.bars_in_trade = 0
        self.entry_price = 0
        self.stop_price = 0
        self.tp_price = 0
        self.trade_direction = None

        print("✨ Indicators ready! Let's find those divergences... 🚀")

    def next(self):
        # Skip if not enough data
        if len(self.data) < max(self.swing_lookback, self.volume_ma_period, self.stoch_period) + 5:
            return

        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        volume = self.data.Volume[-1]
        vol_avg = self.vol_ma[-1]
        k = self.stoch_k[-1]
        d = self.stoch_d[-1]

        # Volume confirmation
        volume_surge = volume > (vol_avg * self.volume_multiplier)

        # Wick detection - skip if large wick (trap)
        body = abs(self.data.Close[-1] - self.data.Open[-1])
        upper_wick = high - max(self.data.Close[-1], self.data.Open[-1])
        lower_wick = min(self.data.Close[-1], self.data.Open[-1]) - low
        large_wick = (upper_wick > body * 2) or (lower_wick > body * 2)

        # Manage existing position
        if self.position:
            self.bars_in_trade += 1

            if self.trade_direction == 'long':
                # Trailing exit: stochastic overbought while price stalls
                if k > 80 and self.data.Close[-1] <= self.data.Close[-2]:
                    new_stop = max(self.stop_price, self.entry_price)
                    if new_stop > self.stop_price:
                        self.stop_price = new_stop
                        print(f"🌙 Tightening long stop to breakeven: {self.stop_price:.2f}")

                # Time-based exit
                if self.bars_in_trade >= self.time_exit_bars:
                    if price < self.entry_price:
                        print(f"⏰ Time exit - long closed at {price:.2f} (no follow-through)")
                        self.position.close()
                        self._reset_trade()
                        return

                # Stop loss / take profit handled by bracket orders

            elif self.trade_direction == 'short':
                # Trailing exit: stochastic oversold while price stalls
                if k < 20 and self.data.Close[-1] >= self.data.Close[-2]:
                    new_stop = min(self.stop_price, self.entry_price)
                    if new_stop < self.stop_price:
                        self.stop_price = new_stop
                        print(f"🌙 Tightening short stop to breakeven: {self.stop_price:.2f}")

                # Time-based exit
                if self.bars_in_trade >= self.time_exit_bars:
                    if price > self.entry_price:
                        print(f"⏰ Time exit - short closed at {price:.2f} (no follow-through)")
                        self.position.close()
                        self._reset_trade()
                        return

            # Update stop if changed
            if self.trade_direction == 'long':
                self.orders.set_stop(self.stop_price)
            else:
                self.orders.set_stop(self.stop_price)

            return

        # --- Entry Logic ---

        # Bullish divergence: price makes lower low, stochastic makes higher low
        if len(self.data) >= self.swing_lookback + 2:
            recent_low = self.data.Low[-1]
            prev_low = self.data.Low[-self.swing_lookback]
            recent_k = k
            prev_k = self.stoch_k[-self.swing_lookback]

            bullish_div = (recent_low < prev_low) and (recent_k > prev_k) and (recent_k < 30)
            bearish_div = (high > self.data.High[-self.swing_lookback]) and (recent_k < self.stoch_k[-self.swing_lookback]) and (recent_k > 70)

            # Long entry
            if bullish_div and volume_surge and not large_wick:
                range_high = self.swing_high[-2]
                if price > range_high:
                    range_low = self.swing_low[-2]
                    range_height = range_high - range_low
                    stop = low * 0.995
                    risk = price - stop
                    if risk > 0:
                        tp = price + (range_height * 1.0)
                        # Ensure minimum RR 1:2
                        if (tp - price) / risk >= self.rr_ratio:
                            size = int(round(1_000_000 / price))
                            print(f"🚀 MOON DEV LONG SIGNAL! Price: {price:.2f}, Divergence + Volume Surge ({volume/vol_avg:.2f}x)")
                            print(f"   Stop: {stop:.2f}, TP: {tp:.2f}, Size: {size}")
                            self.buy(size=size, sl=stop, tp=tp)
                            self.entry_price = price
                            self.stop_price = stop
                            self.tp_price = tp
                            self.trade_direction = 'long'
                            self.bars_in_trade = 0

            # Short entry
            elif bearish_div and volume_surge and not large_wick:
                range_low = self.swing_low[-2]
                if price < range_low:
                    range_high = self.swing_high[-2]
                    range_height = range_high - range_low
                    stop = high * 1.005
                    risk = stop - price
                    if risk > 0:
                        tp = price - (range_height * 1.0)
                        if (price - tp) / risk >= self.rr_ratio:
                            size = int(round(1_000_000 / price))
                            print(f"🌙 MOON DEV SHORT SIGNAL! Price: {price:.2f}, Divergence + Volume Surge ({volume/vol_avg:.2f}x)")
                            print(f"   Stop: {stop:.2f}, TP: {tp:.2f}, Size: {size}")
                            self.sell(size=size, sl=stop, tp=tp)
                            self.entry_price = price
                            self.stop_price = stop
                            self.tp_price = tp
                            self.trade_direction = 'short'
                            self.bars_in_trade = 0

    def _reset_trade(self):
        self.bars_in_trade = 0
        self.entry_price = 0
        self.stop_price = 0
        self.tp_price = 0
        self.trade_direction = None


# Run backtest
print("🌙🚀 Starting Moon Dev VolumetricDivergence Backtest...")
bt = Backtest(data, VolumetricDivergence, cash=1_000_000, commission=0.002)
stats = bt.run()
print(stats)
print(stats._strategy)
print("✨ Moon Dev Backtest Complete! 🌙")