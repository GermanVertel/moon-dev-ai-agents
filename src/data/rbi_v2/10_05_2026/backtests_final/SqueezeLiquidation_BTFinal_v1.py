import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

DATA_PATH = '/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv'

# --- Load Data ---
data = pd.read_csv(DATA_PATH)
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open', 'high': 'High', 'low': 'Low',
    'close': 'Close', 'volume': 'Volume'
})
if 'datetime' in data.columns:
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

print("🌙✨ Moon Dev SqueezeLiquidation Backtest Initializing... 🚀")
print(f"📊 Data loaded: {len(data)} bars")
print(f"📅 Range: {data.index[0]} → {data.index[-1]}")


class SqueezeLiquidation(Strategy):
    # Strategy parameters
    bb_period = 20
    bb_std = 2.0
    squeeze_lookback = 100
    squeeze_pct = 20  # lowest 20th percentile
    vol_ma_period = 50
    vol_mult = 1.5
    liq_mult = 3.0
    atr_period = 14
    atr_mult = 1.5
    rsi_period = 14
    risk_pct = 0.02

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
            lambda u, m, l: (u - l) / m, self.bb_upper, self.bb_middle, self.bb_lower
        )
        # Bandwidth percentile threshold (rolling)
        self.bw_threshold = self.I(
            lambda bw: pd.Series(bw).rolling(self.squeeze_lookback).quantile(self.squeeze_pct / 100).values,
            self.bandwidth
        )

        # Volume MA
        self.vol_ma = self.I(talib.SMA, volume, timeperiod=self.vol_ma_period)

        # ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        # RSI
        self.rsi = self.I(talib.RSI, close, timeperiod=self.rsi_period)

        # --- Synthetic liquidation proxy ---
        body = close - self.data.Open
        rng = high - low
        # Normalized move
        norm_move = self.I(lambda c, o, r: (c - o) / np.where(r == 0, 1e-9, r), close, self.data.Open, rng)
        # Liquidation intensity proxy: |norm_move| * volume
        self.liq_intensity = self.I(
            lambda nm, v: np.abs(nm) * v, norm_move, volume
        )
        self.liq_ma = self.I(talib.SMA, self.liq_intensity, timeperiod=self.vol_ma_period)

        # Directional liquidation proxies
        self.short_liq = self.I(
            lambda nm, v: np.where(nm > 0, nm * v, 0.0), norm_move, volume
        )
        self.long_liq = self.I(
            lambda nm, v: np.where(nm < 0, -nm * v, 0.0), norm_move, volume
        )
        self.short_liq_ma = self.I(talib.SMA, self.short_liq, timeperiod=self.vol_ma_period)
        self.long_liq_ma = self.I(talib.SMA, self.long_liq, timeperiod=self.vol_ma_period)

        # --- Spot-Futures Basis proxy ---
        ema_fast = self.I(talib.EMA, close, timeperiod=5)
        ema_slow = self.I(talib.EMA, close, timeperiod=20)
        self.basis = self.I(
            lambda f, s: (f - s) / s, ema_fast, ema_slow
        )
        self.basis_prev = self.I(
            lambda b: pd.Series(b).shift(1).values, self.basis
        )

        # State tracking
        self.entry_price = None
        self.entry_bar = None
        self.stop_price = None
        self.tp_price = None
        self.trail_active = False
        self.trail_stop = None
        self.direction = 0

    def next(self):
        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        vol = self.data.Volume[-1]

        # Handle open position management first
        if self.position:
            self._manage_position()

        # Skip if already in a position (max 1 concurrent for simplicity)
        if self.position:
            return

        # Need enough history
        if len(self.data) < self.squeeze_lookback + 5:
            return

        # Gather indicators
        bw = self.bandwidth[-1]
        bw_thr = self.bw_threshold[-1]
        upper = self.bb_upper[-1]
        lower = self.bb_lower[-1]
        middle = self.bb_middle[-1]
        vol_ma = self.vol_ma[-1]
        atr = self.atr[-1]
        rsi = self.rsi[-1]
        basis = self.basis[-1]
        basis_prev = self.basis_prev[-1]

        if any(np.isnan(x) for x in [bw, bw_thr, upper, lower, middle, vol_ma, atr, basis, basis_prev]):
            return

        # Squeeze condition
        squeeze = bw < bw_thr
        if not squeeze:
            return

        # Volume confirmation
        vol_ok = vol > self.vol_mult * vol_ma
        if not vol_ok:
            return

        # Liquidation surge
        short_liq = self.short_liq[-1]
        long_liq = self.long_liq[-1]
        short_liq_ma = self.short_liq_ma[-1]
        long_liq_ma = self.long_liq_ma[-1]

        short_liq_surge = short_liq > self.liq_mult * short_liq_ma if short_liq_ma > 0 else False
        long_liq_surge = long_liq > self.liq_mult * long_liq_ma if long_liq_ma > 0 else False

        # Basis confirmation
        basis_widening_long = basis > 0 and basis > basis_prev
        basis_widening_short = basis < 0 and basis < basis_prev

        # --- LONG ENTRY ---
        if (price > upper and short_liq_surge and basis_widening_long):
            stop = min(middle, low)
            risk_per_unit = price - stop
            if risk_per_unit <= 0:
                return
            risk_amount = self.equity * self.risk_pct
            size = int(round(risk_amount / risk_per_unit))
            if size < 1:
                size = 1
            max_size = int(self.equity / price)
            size = min(size, max_size)
            if size < 1:
                return
            self.buy(size=size)
            self.entry_price = price
            self.entry_bar = len(self.data)
            self.stop_price = stop
            self.tp_price = price + 2 * (upper - lower)
            self.direction = 1
            self.trail_active = False
            self.trail_stop = None
            print(f"🌙🚀 LONG ENTRY @ {price:.2f} | size={size} | stop={stop:.2f} | tp={self.tp_price:.2f} | basis={basis:.5f}")
            return

        # --- SHORT ENTRY ---
        if (price < lower and long_liq_surge and basis_widening_short):
            stop = max(middle, high)
            risk_per_unit = stop - price
            if risk_per_unit <= 0:
                return
            risk_amount = self.equity * self.risk_pct
            size = int(round(risk_amount / risk_per_unit))
            if size < 1:
                size = 1
            max_size = int(self.equity / price)
            size = min(size, max_size)
            if size < 1:
                return
            self.sell(size=size)
            self.entry_price = price
            self.entry_bar = len(self.data)
            self.stop_price = stop
            self.tp_price = price - 2 * (upper - lower)
            self.direction = -1
            self.trail_active = False
            self.trail_stop = None
            print(f"🌙🔻 SHORT ENTRY @ {price:.2f} | size={size} | stop={stop:.2f} | tp={self.tp_price:.2f} | basis={basis:.5f}")
            return

    def _manage_position(self):
        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        atr = self.atr[-1]
        rsi = self.rsi[-1]
        upper = self.bb_upper[-1]
        lower = self.bb_lower[-1]

        if any(np.isnan(x) for x in [atr, rsi, upper, lower]):
            return

        bars_held = len(self.data) - self.entry_bar
        entry = self.entry_price

        # --- LONG management ---
        if self.direction == 1 and self.position.is_long:
            if low <= self.stop_price:
                self.position.close()
                print(f"🌙💥 LONG STOP @ {self.stop_price:.2f} (bars={bars_held})")
                self._reset()
                return
            if high >= self.tp_price:
                self.position.close()
                print(f"🌙🎯 LONG TP @ {self.tp_price:.2f} (bars={bars_held})")
                self._reset()
                return
            if rsi > 75 and price < self.data.Close[-2]:
                self.position.close()
                print(f"🌙📉 LONG RSI EXIT @ {price:.2f} rsi={rsi:.1f}")
                self._reset()
                return
            if bars_held <= 5 and price < upper:
                self.position.close()
                print(f"🌙⏱️ LONG TIME STOP (re-entered BB) @ {price:.2f}")
                self._reset()
                return
            if not self.trail_active and price >= entry + atr:
                self.trail_active = True
                self.trail_stop = price - self.atr_mult * atr
                print(f"🌙✨ LONG TRAIL ACTIVATED @ {price:.2f} trail={self.trail_stop:.2f}")
            if self.trail_active:
                new_trail = price - self.atr_mult * atr
                if new_trail > self.trail_stop:
                    self.trail_stop = new_trail
                if low <= self.trail_stop:
                    self.position.close()
                    print(f"🌙🛑 LONG TRAIL STOP @ {self.trail_stop:.2f}")
                    self._reset()
                    return

        # --- SHORT management ---
        elif self.direction == -1 and self.position.is_short:
            if high >= self.stop_price:
                self.position.close()
                print(f"🌙💥 SHORT STOP @ {self.stop_price:.2f} (bars={bars_held})")
                self._reset()
                return
            if low <= self.tp_price:
                self.position.close()
                print(f"🌙🎯 SHORT TP @ {self.tp_price:.2f} (bars={bars_held})")
                self._reset()
                return
            if rsi < 25 and price > self.data.Close[-2]:
                self.position.close()
                print(f"🌙📈 SHORT RSI EXIT @ {price:.2f} rsi={rsi:.1f}")
                self._reset()
                return
            if bars_held <= 5 and price > lower:
                self.position.close()
                print(f"🌙⏱️ SHORT TIME STOP (re-entered BB) @ {price:.2f}")
                self._reset()
                return
            if not self.trail_active and price <= entry - atr:
                self.trail_active = True
                self.trail_stop = price + self.atr_mult * atr
                print(f"🌙✨ SHORT TRAIL ACTIVATED @ {price:.2f} trail={self.trail_stop:.2f}")
            if self.trail_active:
                new_trail = price + self.atr_mult * atr
                if new_trail < self.trail_stop:
                    self.trail_stop = new_trail
                if high >= self.trail_stop:
                    self.position.close()
                    print(f"🌙🛑 SHORT TRAIL STOP @ {self.trail_stop:.2f}")
                    self._reset()
                    return

    def _reset(self):
        self.entry_price = None
        self.entry_bar = None
        self.stop_price = None
        self.tp_price = None
        self.trail_active = False
        self.trail_stop = None
        self.direction = 0


print("🌙🚀 Running SqueezeLiquidation Backtest...")
bt = Backtest(
    data, SqueezeLiquidation,
    cash=1_000_000, commission=0.001
)
stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙✨ Backtest Complete! 🚀")