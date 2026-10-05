import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Load and clean data
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

# Ensure all OHLCV are float64 for talib compatibility
for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    data[col] = data[col].astype(np.float64)

print("🌙✨ Moon Dev VolatilityChaikin Backtest Loading... 🚀")
print(f"📊 Data shape: {data.shape}")


class VolatilityChaikin(Strategy):
    bb_period = 20
    bb_std = 2.0
    squeeze_lookback = 100
    squeeze_percentile = 0.10
    chaikin_fast = 3
    chaikin_slow = 10
    atr_period = 14
    vol_sma_period = 20
    risk_pct = 0.01
    time_stop_bars = 15

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        # Bollinger Bands
        self.bb_mid = self.I(talib.SMA, close, timeperiod=self.bb_period)
        self.bb_stddev = self.I(talib.STDDEV, close, timeperiod=self.bb_period, nbdev=1)
        self.bb_upper = self.I(lambda m, s: m + self.bb_std * s, self.bb_mid, self.bb_stddev)
        self.bb_lower = self.I(lambda m, s: m - self.bb_std * s, self.bb_mid, self.bb_stddev)

        # Bandwidth
        self.bandwidth = self.I(lambda u, l, m: (u - l) / m, self.bb_upper, self.bb_lower, self.bb_mid)

        # Chaikin Oscillator via ADL — cast to float64 explicitly
        self.adl = self.I(
            lambda h, l, c, v: talib.AD(
                np.asarray(h, dtype=np.float64),
                np.asarray(l, dtype=np.float64),
                np.asarray(c, dtype=np.float64),
                np.asarray(v, dtype=np.float64),
            ),
            high, low, close, volume
        )
        self.chaikin = self.I(
            lambda adl: talib.EMA(np.asarray(adl, dtype=np.float64), timeperiod=self.chaikin_fast)
            - talib.EMA(np.asarray(adl, dtype=np.float64), timeperiod=self.chaikin_slow),
            self.adl
        )

        # ATR
        self.atr = self.I(
            lambda h, l, c: talib.ATR(
                np.asarray(h, dtype=np.float64),
                np.asarray(l, dtype=np.float64),
                np.asarray(c, dtype=np.float64),
                timeperiod=self.atr_period,
            ),
            high, low, close
        )

        # Volume SMA
        self.vol_sma = self.I(
            lambda v: talib.SMA(np.asarray(v, dtype=np.float64), timeperiod=self.vol_sma_period),
            volume
        )

        # Squeeze threshold: rolling percentile
        self.squeeze_thresh = self.I(
            lambda bw: pd.Series(bw).rolling(self.squeeze_lookback).quantile(self.squeeze_percentile).values,
            self.bandwidth
        )

        # Track squeeze range
        self.squeeze_high = None
        self.squeeze_low = None
        self.squeeze_start = None

        # Position management
        self.entry_price = None
        self.initial_size = 0
        self.stop_price = None
        self.target1 = None
        self.target2 = None
        self.target3 = None
        self.t1_hit = False
        self.t2_hit = False
        self.bars_in_trade = 0

    def next(self):
        if len(self.data) < self.squeeze_lookback + 5:
            return

        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        vol = self.data.Volume[-1]

        bw = self.bandwidth[-1]
        thresh = self.squeeze_thresh[-1]
        chaikin = self.chaikin[-1]
        chaikin_prev = self.chaikin[-2]
        bb_up = self.bb_upper[-1]
        bb_lo = self.bb_lower[-1]
        atr = self.atr[-1]
        vol_sma = self.vol_sma[-1]

        if np.isnan(bw) or np.isnan(thresh) or np.isnan(chaikin) or np.isnan(atr):
            return

        # Manage existing position
        if self.position:
            self.bars_in_trade += 1
            entry = self.entry_price

            # Check stops
            if self.stop_price and low <= self.stop_price:
                print(f"🛑 Moon Dev STOP HIT at {self.stop_price:.2f} 🌙")
                self.position.close()
                self._reset_trade()
                return

            # Target 1
            if not self.t1_hit and high >= self.target1:
                print(f"🎯 Moon Dev Target 1 hit @ {self.target1:.2f} — scaling 50% ✨")
                close_size = int(round(self.position.size * 0.5))
                if close_size > 0:
                    self.sell(size=close_size)
                self.t1_hit = True
                self.stop_price = entry  # breakeven

            # Target 2
            if self.t1_hit and not self.t2_hit and high >= self.target2:
                print(f"🎯 Moon Dev Target 2 hit @ {self.target2:.2f} — scaling 30% 🚀")
                close_size = int(round(self.position.size * (0.30 / 0.50))) if self.position.size > 0 else 0
                if close_size > 0:
                    self.sell(size=close_size)
                self.t2_hit = True

            # Target 3
            if self.t2_hit and high >= self.target3:
                print(f"🏆 Moon Dev Target 3 hit @ {self.target3:.2f} — closing runner 🌙")
                self.position.close()
                self._reset_trade()
                return

            # Trailing stop after T2
            if self.t2_hit and not np.isnan(atr):
                new_stop = price - 2 * atr
                if new_stop > self.stop_price:
                    self.stop_price = new_stop

            # Momentum exit
            if chaikin < 0 and chaikin_prev >= 0:
                print(f"💫 Moon Dev Chaikin crossed below zero — momentum exit 🌙")
                self.position.close()
                self._reset_trade()
                return

            # Time stop
            if self.bars_in_trade >= self.time_stop_bars and not self.t1_hit:
                print(f"⏰ Moon Dev Time stop reached — exiting at market 🌙")
                self.position.close()
                self._reset_trade()
                return

            return

        # Entry logic
        squeeze_active = bw <= thresh
        chaikin_cross = chaikin_prev < 0 and chaikin >= 0
        breakout = price > bb_up
        vol_confirm = vol > vol_sma

        # Track squeeze range
        if squeeze_active:
            if self.squeeze_start is None:
                self.squeeze_start = len(self.data)
                self.squeeze_high = high
                self.squeeze_low = low
            else:
                self.squeeze_high = max(self.squeeze_high, high)
                self.squeeze_low = min(self.squeeze_low, low)

        if squeeze_active and chaikin_cross and breakout and vol_confirm:
            if self.squeeze_high is None or self.squeeze_low is None:
                return

            rng = self.squeeze_high - self.squeeze_low
            if rng <= 0:
                return

            entry = price
            stop = min(bb_lo, self.squeeze_low)
            risk = entry - stop
            if risk <= 0:
                return

            # Position sizing: risk 1% of equity
            risk_amount = self.equity * self.risk_pct
            size = int(round(risk_amount / risk))
            if size < 1:
                size = 1

            self.target1 = entry + 1.272 * rng
            self.target2 = entry + 1.618 * rng
            self.target3 = entry + 2.618 * rng

            print(f"🌙✨ Moon Dev BREAKOUT! Entry={entry:.2f} Stop={stop:.2f} Size={size} 🚀")
            print(f"   Targets: T1={self.target1:.2f} T2={self.target2:.2f} T3={self.target3:.2f}")

            self.buy(size=size)
            self.entry_price = entry
            self.initial_size = size
            self.stop_price = stop
            self.t1_hit = False
            self.t2_hit = False
            self.bars_in_trade = 0
            self.squeeze_start = None
            self.squeeze_high = None
            self.squeeze_low = None

    def _reset_trade(self):
        self.entry_price = None
        self.initial_size = 0
        self.stop_price = None
        self.target1 = None
        self.target2 = None
        self.target3 = None
        self.t1_hit = False
        self.t2_hit = False
        self.bars_in_trade = 0


bt = Backtest(data, VolatilityChaikin, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)