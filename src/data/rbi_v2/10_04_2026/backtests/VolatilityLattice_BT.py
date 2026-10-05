import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# 🌙 Moon Dev's VolatilityLattice Backtest 🚀

data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"

data = pd.read_csv(data_path)
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

if 'datetime' in data.columns:
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')

data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()


class VolatilityLattice(Strategy):
    # Lattice parameters
    vwm_lookback = 20
    atr_period = 14
    lattice_mult_1 = 1.0
    lattice_mult_2 = 2.0
    lattice_mult_3 = 3.0

    # Momentum
    roc_period = 10
    roc_threshold = 0.0

    # Volume
    vol_ma_period = 20

    # Regime detection
    vol_short = 5
    vol_long = 20

    # Risk
    risk_pct = 0.02
    max_lattice_width_atr = 6.0
    time_stop_bars = 30

    def init(self):
        print("🌙✨ Initializing VolatilityLattice indicators... 🚀")

        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        # ATR for lattice spacing
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        # Volatility-weighted mean using EMA (proxy) - weight by ATR
        self.ema = self.I(talib.EMA, close, timeperiod=self.vwm_lookback)

        # Momentum
        self.roc = self.I(talib.ROC, close, timeperiod=self.roc_period)

        # Volume filter
        self.vol_ma = self.I(talib.SMA, volume, timeperiod=self.vol_ma_period)

        # Regime detection: short vs long ATR
        self.atr_short = self.I(talib.ATR, high, low, close, timeperiod=self.vol_short)
        self.atr_long = self.I(talib.ATR, high, low, close, timeperiod=self.vol_long)

        # Lattice bands will be computed dynamically in next()

        self.entry_bar = None
        self.entry_price = None
        self.stop_price = None
        self.target_price = None
        self.trade_type = None

        print("🌙 Indicators ready! Let's ride the lattice! 🚀")

    def next(self):
        if len(self.data) < max(self.vwm_lookback, self.atr_period, self.vol_long) + 5:
            return

        price = self.data.Close[-1]
        atr = self.atr[-1]

        if np.isnan(atr) or atr <= 0:
            return

        vwm = self.ema[-1]
        roc = self.roc[-1]
        vol = self.data.Volume[-1]
        vol_ma = self.vol_ma[-1]

        # Lattice levels
        upper1 = vwm + self.lattice_mult_1 * atr
        upper2 = vwm + self.lattice_mult_2 * atr
        upper3 = vwm + self.lattice_mult_3 * atr
        lower1 = vwm - self.lattice_mult_1 * atr
        lower2 = vwm - self.lattice_mult_2 * atr
        lower3 = vwm - self.lattice_mult_3 * atr

        # Lattice width filter
        lattice_width = upper3 - lower3
        if lattice_width > self.max_lattice_width_atr * atr:
            return

        # Regime: expanding vs contracting volatility
        atr_s = self.atr_short[-1]
        atr_l = self.atr_long[-1]
        expanding = atr_s > atr_l if not (np.isnan(atr_s) or np.isnan(atr_l)) else False
        contracting = atr_s < atr_l if not (np.isnan(atr_s) or np.isnan(atr_l)) else False

        prev_price = self.data.Close[-2]
        prev_upper1 = self.ema[-2] + self.lattice_mult_1 * self.atr[-2] if not np.isnan(self.atr[-2]) else upper1
        prev_lower1 = self.ema[-2] - self.lattice_mult_1 * self.atr[-2] if not np.isnan(self.atr[-2]) else lower1

        volume_ok = (not np.isnan(vol_ma)) and vol > vol_ma

        # ---------- MANAGE OPEN POSITION ----------
        if self.position:
            bars_held = len(self.data) - self.entry_bar
            pnl_dir = 1 if self.position.is_long else -1
            current_price = price

            # Volatility stop: if ATR collapses after entry
            if contracting and bars_held > 3 and atr_s < atr_l * 0.7:
                print(f"🌙💨 Volatility collapse detected, exiting! ATR short={atr_s:.2f} long={atr_l:.2f}")
                self.position.close()
                self._reset()
                return

            # Time stop
            if bars_held >= self.time_stop_bars:
                print(f"⏰ Time stop hit after {bars_held} bars. Exiting.")
                self.position.close()
                self._reset()
                return

            # Trailing stop along nearest lattice band
            if self.position.is_long:
                new_stop = lower1
                if new_stop > self.stop_price:
                    self.stop_price = new_stop
                if current_price <= self.stop_price:
                    print(f"🛑 Trailing stop hit LONG at {current_price:.2f} (stop={self.stop_price:.2f})")
                    self.position.close()
                    self._reset()
                    return
                # Profit target at next lattice level
                if current_price >= self.target_price:
                    print(f"🎯 Target hit LONG at {current_price:.2f}")
                    self.position.close()
                    self._reset()
                    return
            else:
                new_stop = upper1
                if new_stop < self.stop_price:
                    self.stop_price = new_stop
                if current_price >= self.stop_price:
                    print(f"🛑 Trailing stop hit SHORT at {current_price:.2f} (stop={self.stop_price:.2f})")
                    self.position.close()
                    self._reset()
                    return
                if current_price <= self.target_price:
                    print(f"🎯 Target hit SHORT at {current_price:.2f}")
                    self.position.close()
                    self._reset()
                    return
            return

        # ---------- ENTRY LOGIC ----------
        # Long breakout: price closes above upper1, momentum positive, expanding vol, volume filter
        if (price > upper1 and prev_price <= prev_upper1 and roc > self.roc_threshold
                and expanding and volume_ok):
            stop = lower1
            risk = price - stop
            if risk > 0:
                size = int(round(1_000_000 / price))
                if size > 0:
                    print(f"🚀🌙 LONG breakout! price={price:.2f} upper1={upper1:.2f} roc={roc:.3f} size={size}")
                    self.buy(size=size)
                    self.entry_bar = len(self.data)
                    self.entry_price = price
                    self.stop_price = stop
                    self.target_price = upper2
                    self.trade_type = 'breakout_long'
            return

        # Short breakout: price closes below lower1, momentum negative, expanding vol, volume filter
        if (price < lower1 and prev_price >= prev_lower1 and roc < -self.roc_threshold
                and expanding and volume_ok):
            stop = upper1
            risk = stop - price
            if risk > 0:
                size = int(round(1_000_000 / price))
                if size > 0:
                    print(f"🔻🌙 SHORT breakout! price={price:.2f} lower1={lower1:.2f} roc={roc:.3f} size={size}")
                    self.sell(size=size)
                    self.entry_bar = len(self.data)
                    self.entry_price = price
                    self.stop_price = stop
                    self.target_price = lower2
                    self.trade_type = 'breakout_short'
            return

        # Fade long: contracting vol, price touches lower2, no negative momentum confirmation
        if (contracting and price <= lower2 and roc > -self.roc_threshold * 0.5
                and price > lower3):
            stop = lower3
            risk = price - stop
            if risk > 0:
                size = int(round(1_000_000 / price))
                if size > 0:
                    print(f"🌊 Fade LONG at lower2={lower2:.2f} price={price:.2f} roc={roc:.3f}")
                    self.buy(size=size)
                    self.entry_bar = len(self.data)
                    self.entry_price = price
                    self.stop_price = stop
                    self.target_price = vwm
                    self.trade_type = 'fade_long'
            return

        # Fade short: contracting vol, price touches upper2, no positive momentum confirmation
        if (contracting and price >= upper2 and roc < self.roc_threshold * 0.5
                and price < upper3):
            stop = upper3
            risk = stop - price
            if risk > 0:
                size = int(round(1_000_000 / price))
                if size > 0:
                    print(f"🌊 Fade SHORT at upper2={upper2:.2f} price={price:.2f} roc={roc:.3f}")
                    self.sell(size=size)
                    self.entry_bar = len(self.data)
                    self.entry_price = price
                    self.stop_price = stop
                    self.target_price = vwm
                    self.trade_type = 'fade_short'
            return

    def _reset(self):
        self.entry_bar = None
        self.entry_price = None
        self.stop_price = None
        self.target_price = None
        self.trade_type = None


print("🌙✨🚀 Starting Moon Dev VolatilityLattice Backtest... 🚀✨🌙")
bt = Backtest(data, VolatilityLattice, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)