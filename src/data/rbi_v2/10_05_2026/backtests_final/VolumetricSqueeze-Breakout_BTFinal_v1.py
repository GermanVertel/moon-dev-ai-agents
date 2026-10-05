import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

print("🌙✨ Moon Dev's Volumetric Squeeze-Breakout Backtest Loading... 🚀")

# Load data
data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper column mapping
data = data.rename(columns={
    'datetime': 'Date',
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

data['Date'] = pd.to_datetime(data['Date'])
data = data.set_index('Date')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

print(f"🌙 Data loaded: {len(data)} bars ✨")


class VolumetricSqueezeBreakout(Strategy):
    # Parameters
    bb_period = 20
    bb_std = 2.0
    squeeze_lookback = 50
    squeeze_threshold = 1.2  # bandwidth must be within 1.2x of min
    vol_delta_mult = 3.0
    vol_delta_lookback = 20
    profit_target_mult = 1.5
    stop_buffer = 0.001
    risk_pct = 0.02
    min_rr = 1.5

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        # Bollinger Bands
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS, close, timeperiod=self.bb_period,
            nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0
        )

        # Bandwidth
        self.bandwidth = self.I(
            lambda u, l: u - l, self.bb_upper, self.bb_lower
        )

        # Bandwidth N-period low (squeeze detection)
        self.bandwidth_min = self.I(
            talib.MIN, self.bandwidth, timeperiod=self.squeeze_lookback
        )

        # Volume delta proxy: (close - open) / (high - low) * volume
        def calc_vol_delta(o, h, l, c, v):
            rng = np.where((h - l) == 0, 1e-10, h - l)
            return ((c - o) / rng) * v

        self.vol_delta = self.I(
            calc_vol_delta,
            self.data.Open, high, low, close, volume
        )

        # Volume delta rolling average (absolute value)
        self.vol_delta_avg = self.I(
            lambda vd: pd.Series(np.abs(vd)).rolling(self.vol_delta_lookback).mean().values,
            self.vol_delta
        )

        # ATR for context
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=14)

        print("🌙✨ Indicators initialized: BB, Bandwidth, Volume Delta, ATR 🚀")

    def next(self):
        # Need enough history
        if len(self.data) < max(self.squeeze_lookback, self.vol_delta_lookback) + 5:
            return

        # Skip if already in position
        if self.position:
            self._manage_position()
            return

        price = self.data.Close[-1]
        open_ = self.data.Open[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]

        upper = self.bb_upper[-1]
        lower = self.bb_lower[-1]
        bw = self.bandwidth[-1]
        bw_min = self.bandwidth_min[-1]

        vd = self.vol_delta[-1]
        vd_avg = self.vol_delta_avg[-1]

        # Guard against NaN
        if any(np.isnan(x) for x in [upper, lower, bw, bw_min, vd, vd_avg]):
            return

        if vd_avg <= 0:
            return

        # Squeeze condition
        is_squeeze = bw <= bw_min * self.squeeze_threshold

        # Volume delta surge
        vd_surge_long = vd >= self.vol_delta_mult * vd_avg
        vd_surge_short = vd <= -self.vol_delta_mult * vd_avg

        # Engulfing detection (current vs prior candle)
        prev_open = self.data.Open[-2]
        prev_close = self.data.Close[-2]
        prev_high = self.data.High[-2]
        prev_low = self.data.Low[-2]

        bull_engulf = (open_ <= prev_close and price >= prev_open and
                       price > open_ and price > prev_high)
        bear_engulf = (open_ >= prev_close and price <= prev_open and
                       price < open_ and price < prev_low)

        # Breakout conditions
        long_breakout = price > upper
        short_breakout = price < lower

        # Squeeze range for target
        squeeze_range = bw

        # Long entry
        if is_squeeze and long_breakout and vd_surge_long and bull_engulf:
            stop = low * (1 - self.stop_buffer)
            risk = price - stop
            if risk <= 0:
                return
            target = price + self.profit_target_mult * squeeze_range
            rr = (target - price) / risk

            if rr < self.min_rr:
                print(f"🌙 ⚠️ Long skipped: R:R {rr:.2f} < {self.min_rr}")
                return

            risk_amount = self.equity * self.risk_pct
            size = int(round(risk_amount / risk))
            if size <= 0:
                return

            print(f"🚀🌙 LONG: price={price:.2f} stop={stop:.2f} target={target:.2f} "
                  f"R:R={rr:.2f} size={size} vd_mult={vd/vd_avg:.2f}x")
            self.buy(size=size, sl=stop, tp=target)

        # Short entry
        elif is_squeeze and short_breakout and vd_surge_short and bear_engulf:
            stop = high * (1 + self.stop_buffer)
            risk = stop - price
            if risk <= 0:
                return
            target = price - self.profit_target_mult * squeeze_range
            rr = (price - target) / risk

            if rr < self.min_rr:
                print(f"🌙 ⚠️ Short skipped: R:R {rr:.2f} < {self.min_rr}")
                return

            risk_amount = self.equity * self.risk_pct
            size = int(round(risk_amount / risk))
            if size <= 0:
                return

            print(f"🚀🌙 SHORT: price={price:.2f} stop={stop:.2f} target={target:.2f} "
                  f"R:R={rr:.2f} size={size} vd_mult={vd/vd_avg:.2f}x")
            self.sell(size=size, sl=stop, tp=target)

    def _manage_position(self):
        # Optional trailing stop: trail below/above prior candle's low/high once
        # price has moved 1x squeeze range in favor
        if not self.position:
            return

        price = self.data.Close[-1]
        low = self.data.Low[-1]
        high = self.data.High[-1]
        bw = self.bandwidth[-1]
        if np.isnan(bw):
            return

        # Get entry price from last trade (Position object has no .entry_price)
        if not self.trades:
            return
        entry = self.trades[-1].entry_price

        if self.position.is_long:
            if price - entry >= bw:
                new_sl = low * (1 - self.stop_buffer)
                if new_sl > self.trades[-1].sl:
                    self.trades[-1].sl = new_sl
        else:
            if entry - price >= bw:
                new_sl = high * (1 + self.stop_buffer)
                if new_sl < self.trades[-1].sl:
                    self.trades[-1].sl = new_sl


print("🌙✨ Starting Moon Dev's Volumetric Squeeze-Breakout Backtest... 🚀")
bt = Backtest(
    data,
    VolumetricSqueezeBreakout,
    cash=1_000_000,
    commission=0.001
)
stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙✨ Backtest complete! Moon Dev out 🚀")