import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# Load and clean data
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
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
print(f"🌙 Moon Dev Data Loaded: {len(data)} bars ✨")
print(f"🚀 Data range: {data.index[0]} to {data.index[-1]}")


class VolumeDivergentReversion(Strategy):
    bb_period = 20
    bb_std = 2.0
    vol_period = 20
    atr_period = 14
    atr_stop_mult = 1.5
    atr_trail_mult = 1.0
    risk_pct = 0.02
    time_stop_bars = 10
    bb_width_lookback = 100
    bb_width_pct = 0.95

    def init(self):
        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        volume = pd.Series(self.data.Volume)

        # Bollinger Bands
        self.mb = self.I(talib.SMA, close, timeperiod=self.bb_period)
        std = self.I(talib.STDDEV, close, timeperiod=self.bb_period, nbdev=1.0)
        self.ub = self.I(lambda m, s: m + self.bb_std * s, self.mb, std)
        self.lb = self.I(lambda m, s: m - self.bb_std * s, self.mb, std)

        # Volume SMA
        self.vol_sma = self.I(talib.SMA, volume, timeperiod=self.vol_period)

        # ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        # BB width
        self.bb_width = self.I(lambda u, l, m: (u - l) / m, self.ub, self.lb, self.mb)
        self.bb_width_max = self.I(talib.MAX, self.bb_width, timeperiod=self.bb_width_lookback)

        # LB/UB slope proxies (diff)
        self.lb_prev = self.I(lambda x: np.concatenate([[np.nan], x[:-1]]), self.lb)
        self.ub_prev = self.I(lambda x: np.concatenate([[np.nan], x[:-1]]), self.ub)

        self.entry_bar = None
        self.entry_price = None
        self.stop_price = None
        self.target_price = None
        self.highest_close = None
        self.lowest_close = None
        self.trailing_active = False

        print("🌙✨ VolumeDivergent Reversion indicators initialized 🚀")

    def next(self):
        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        vol = self.data.Volume[-1]

        # Manage open position
        if self.position:
            bars_held = len(self.data) - self.entry_bar
            if self.position.is_long:
                # Track highest close
                if self.highest_close is None or price > self.highest_close:
                    self.highest_close = price

                # Activate trailing once price crosses MB
                if not self.trailing_active and price >= self.mb[-1]:
                    self.trailing_active = True
                    print(f"🌙 Long crossed MB — trailing activated at {price:.2f} ✨")

                # Update trailing stop
                if self.trailing_active and self.highest_close is not None:
                    new_stop = self.highest_close - self.atr_trail_mult * self.atr[-1]
                    if new_stop > self.stop_price:
                        self.stop_price = new_stop

                # Primary target: MB
                if price >= self.mb[-1] and not self.trailing_active:
                    print(f"🚀 Long hit MB target at {price:.2f} — exiting 🌙")
                    self.position.close()
                    self._reset()
                    return

                # Stop hit
                if low <= self.stop_price:
                    print(f"🛑 Long stop hit at {self.stop_price:.2f} — exiting 🌙")
                    self.position.close()
                    self._reset()
                    return

                # Time stop
                if bars_held >= self.time_stop_bars and not self.trailing_active:
                    print(f"⏰ Long time stop ({bars_held} bars) — exiting 🌙")
                    self.position.close()
                    self._reset()
                    return

            elif self.position.is_short:
                if self.lowest_close is None or price < self.lowest_close:
                    self.lowest_close = price

                if not self.trailing_active and price <= self.mb[-1]:
                    self.trailing_active = True
                    print(f"🌙 Short crossed MB — trailing activated at {price:.2f} ✨")

                if self.trailing_active and self.lowest_close is not None:
                    new_stop = self.lowest_close + self.atr_trail_mult * self.atr[-1]
                    if new_stop < self.stop_price:
                        self.stop_price = new_stop

                if price <= self.mb[-1] and not self.trailing_active:
                    print(f"🚀 Short hit MB target at {price:.2f} — exiting 🌙")
                    self.position.close()
                    self._reset()
                    return

                if high >= self.stop_price:
                    print(f"🛑 Short stop hit at {self.stop_price:.2f} — exiting 🌙")
                    self.position.close()
                    self._reset()
                    return

                if bars_held >= self.time_stop_bars and not self.trailing_active:
                    print(f"⏰ Short time stop ({bars_held} bars) — exiting 🌙")
                    self.position.close()
                    self._reset()
                    return

        # Entry logic
        if not self.position and len(self.data) > self.bb_width_lookback + 2:
            atr_now = self.atr[-1]
            atr_prev = self.atr[-2] if not np.isnan(self.atr[-2]) else atr_now

            # Skip high volatility regime
            if self.bb_width[-1] >= self.bb_width_max[-1] * self.bb_width_pct:
                return

            # Skip ATR expansion >50%
            if atr_prev > 0 and atr_now > atr_prev * 1.5:
                return

            vol_below = vol < self.vol_sma[-1]
            vol_above = vol > self.vol_sma[-1]

            if np.isnan(self.mb[-1]) or np.isnan(self.ub[-1]) or np.isnan(self.lb[-1]):
                return

            # LONG setup
            touched_lb = low <= self.lb[-1] or self.data.Close[-2] <= self.lb[-2]
            lb_expanding = self.lb[-1] < self.lb_prev[-1]
            ub_flat = self.ub[-1] <= self.ub_prev[-1]
            rejection_long = price > self.lb[-1]

            if touched_lb and lb_expanding and ub_flat and vol_below and rejection_long and not vol_above:
                stop = price - self.atr_stop_mult * atr_now
                risk = price - stop
                if risk <= 0:
                    return
                size = int(round(1000000 * self.risk_pct / risk))
                if size <= 0:
                    return
                print(f"🌙🚀 LONG signal | Price:{price:.2f} LB:{self.lb[-1]:.2f} ATR:{atr_now:.2f} Size:{size} ✨")
                self.buy(size=size)
                self.entry_bar = len(self.data)
                self.entry_price = price
                self.stop_price = stop
                self.target_price = self.mb[-1]
                self.highest_close = price
                self.trailing_active = False

            # SHORT setup
            touched_ub = high >= self.ub[-1] or self.data.Close[-2] >= self.ub[-2]
            ub_expanding = self.ub[-1] > self.ub_prev[-1]
            lb_flat = self.lb[-1] >= self.lb_prev[-1]
            rejection_short = price < self.ub[-1]

            if touched_ub and ub_expanding and lb_flat and vol_below and rejection_short and not vol_above:
                stop = price + self.atr_stop_mult * atr_now
                risk = stop - price
                if risk <= 0:
                    return
                size = int(round(1000000 * self.risk_pct / risk))
                if size <= 0:
                    return
                print(f"🌙🚀 SHORT signal | Price:{price:.2f} UB:{self.ub[-1]:.2f} ATR:{atr_now:.2f} Size:{size} ✨")
                self.sell(size=size)
                self.entry_bar = len(self.data)
                self.entry_price = price
                self.stop_price = stop
                self.target_price = self.mb[-1]
                self.lowest_close = price
                self.trailing_active = False

    def _reset(self):
        self.entry_bar = None
        self.entry_price = None
        self.stop_price = None
        self.target_price = None
        self.highest_close = None
        self.lowest_close = None
        self.trailing_active = False


bt = Backtest(data, VolumeDivergentReversion, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)