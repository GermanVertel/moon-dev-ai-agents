import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

print("🌙 Moon Dev VortexSurge Backtest Initializing... ✨🚀")

# Load data
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')

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

data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')

print(f"🌙 Data loaded: {len(data)} bars from {data.index[0]} to {data.index[-1]} ✨")


class VortexSurge(Strategy):
    # Strategy parameters
    vi_period = 14
    vol_sma_period = 20
    surge_mult = 1.5
    ema_keltner = 20
    ema_trend = 50
    atr_keltner = 10
    atr_stop = 14
    atr_pct_lookback = 100
    base_kc_mult = 2.0
    risk_pct = 0.02
    time_stop_bars = 20
    vol_guard_pct = 95

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        # Vortex Indicator
        def vortex_plus(h, l, c, n):
            h = np.asarray(h, dtype=float)
            l = np.asarray(l, dtype=float)
            c = np.asarray(c, dtype=float)
            vm_plus = np.abs(h - np.roll(l, 1))
            tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
            vm_plus = pd.Series(vm_plus).rolling(n).sum()
            tr_sum = pd.Series(tr).rolling(n).sum()
            return (vm_plus / tr_sum).values

        def vortex_minus(h, l, c, n):
            h = np.asarray(h, dtype=float)
            l = np.asarray(l, dtype=float)
            c = np.asarray(c, dtype=float)
            vm_minus = np.abs(l - np.roll(h, 1))
            tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
            vm_minus = pd.Series(vm_minus).rolling(n).sum()
            tr_sum = pd.Series(tr).rolling(n).sum()
            return (vm_minus / tr_sum).values

        self.vi_plus = self.I(vortex_plus, high, low, close, self.vi_period, name='VI+')
        self.vi_minus = self.I(vortex_minus, high, low, close, self.vi_period, name='VI-')

        # Volume SMA - cast to float64 for talib
        vol_arr = np.asarray(volume, dtype=np.float64)
        self.vol_sma = self.I(talib.SMA, vol_arr, timeperiod=self.vol_sma_period, name='VolSMA')

        # EMAs
        close_arr = np.asarray(close, dtype=np.float64)
        high_arr = np.asarray(high, dtype=np.float64)
        low_arr = np.asarray(low, dtype=np.float64)

        self.ema20 = self.I(talib.EMA, close_arr, timeperiod=self.ema_keltner, name='EMA20')
        self.ema50 = self.I(talib.EMA, close_arr, timeperiod=self.ema_trend, name='EMA50')

        # ATRs
        self.atr10 = self.I(talib.ATR, high_arr, low_arr, close_arr, timeperiod=self.atr_keltner, name='ATR10')
        self.atr14 = self.I(talib.ATR, high_arr, low_arr, close_arr, timeperiod=self.atr_stop, name='ATR14')

        # ATR Percentile Rank
        def atr_pct_rank(atr, n):
            s = pd.Series(np.asarray(atr, dtype=float))
            return s.rolling(n).apply(lambda x: (x.iloc[-1] > x).mean() * 100, raw=False).values

        self.atr_pct = self.I(atr_pct_rank, self.atr10, self.atr_pct_lookback, name='ATRpct')

        # Dynamic Keltner multiplier
        def dyn_mult(atr_pct, base):
            pct = np.nan_to_num(np.asarray(atr_pct, dtype=float), nan=50)
            return base * (1 + (pct - 50) / 100)

        self.dyn_mult = self.I(dyn_mult, self.atr_pct, self.base_kc_mult, name='DynMult')

        # Keltner Bands
        def kc_upper(ema, atr, mult):
            return np.asarray(ema, dtype=float) + np.asarray(atr, dtype=float) * np.asarray(mult, dtype=float)

        def kc_lower(ema, atr, mult):
            return np.asarray(ema, dtype=float) - np.asarray(atr, dtype=float) * np.asarray(mult, dtype=float)

        self.kc_upper = self.I(kc_upper, self.ema20, self.atr10, self.dyn_mult, name='KC_Upper')
        self.kc_lower = self.I(kc_lower, self.ema20, self.atr10, self.dyn_mult, name='KC_Lower')

        # State tracking
        self.entry_bar = None
        self.stop_price = None
        self.entry_price = None
        self.trade_side = None

        print("🌙 VortexSurge indicators initialized! ✨🚀")

    def next(self):
        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        volume = self.data.Volume[-1]

        # Skip if indicators not ready
        if (np.isnan(self.vi_plus[-1]) or np.isnan(self.vi_minus[-1]) or
                np.isnan(self.vi_plus[-2]) or np.isnan(self.vi_minus[-2]) or
                np.isnan(self.vol_sma[-1]) or np.isnan(self.kc_upper[-1]) or
                np.isnan(self.kc_lower[-1]) or np.isnan(self.atr10[-1]) or
                np.isnan(self.atr_pct[-1])):
            return

        vi_plus_now = self.vi_plus[-1]
        vi_minus_now = self.vi_minus[-1]
        vi_plus_prev = self.vi_plus[-2]
        vi_minus_prev = self.vi_minus[-2]

        bull_cross = vi_plus_prev <= vi_minus_prev and vi_plus_now > vi_minus_now
        bear_cross = vi_minus_prev <= vi_plus_prev and vi_minus_now > vi_plus_now

        vol_surge = volume > self.vol_sma[-1] * self.surge_mult
        atr_pct = self.atr_pct[-1]

        # Volatility guard
        vol_guard = atr_pct > self.vol_guard_pct

        # === EXITS ===
        if self.position:
            bars_held = len(self.data) - 1 - self.entry_bar

            if self.position.is_long:
                # Keltner exit
                if price < self.kc_lower[-1]:
                    print(f"🌙 Long exit: price {price:.2f} < KC lower {self.kc_lower[-1]:.2f} ✨")
                    self.position.close()
                    self.entry_bar = None
                    return
                # Opposite crossover exit
                if bear_cross:
                    print(f"🌙 Long exit: bearish VI crossover ✨")
                    self.position.close()
                    self.entry_bar = None
                    return
                # Trailing stop using lower band
                if self.stop_price is not None:
                    new_stop = self.kc_lower[-1]
                    if new_stop > self.stop_price:
                        self.stop_price = new_stop
                # Time stop
                if bars_held >= self.time_stop_bars:
                    if price <= self.entry_price:
                        print(f"🌙 Long time stop after {bars_held} bars ✨")
                        self.position.close()
                        self.entry_bar = None
                        return

            elif self.position.is_short:
                if price > self.kc_upper[-1]:
                    print(f"🌙 Short exit: price {price:.2f} > KC upper {self.kc_upper[-1]:.2f} ✨")
                    self.position.close()
                    self.entry_bar = None
                    return
                if bull_cross:
                    print(f"🌙 Short exit: bullish VI crossover ✨")
                    self.position.close()
                    self.entry_bar = None
                    return
                if self.stop_price is not None:
                    new_stop = self.kc_upper[-1]
                    if new_stop < self.stop_price:
                        self.stop_price = new_stop
                if bars_held >= self.time_stop_bars:
                    if price >= self.entry_price:
                        print(f"🌙 Short time stop after {bars_held} bars ✨")
                        self.position.close()
                        self.entry_bar = None
                        return

        # === ENTRIES ===
        if not self.position:
            # Long entry
            if bull_cross and vol_surge and price > self.ema50[-1]:
                if vol_guard:
                    print(f"🌙 Volatility guard active (ATR%={atr_pct:.1f}), skipping long 🚀")
                else:
                    stop = min(self.kc_lower[-1], price - 2 * self.atr14[-1])
                    risk = price - stop
                    if risk > 0:
                        size = int(round((self.equity * self.risk_pct) / risk))
                        if size > 0 and size < self.equity / price:
                            self.buy(size=size)
                            self.entry_bar = len(self.data) - 1
                            self.entry_price = price
                            self.stop_price = stop
                            print(f"🚀🌙 LONG entry @ {price:.2f}, size={size}, stop={stop:.2f} ✨")

            # Short entry
            elif bear_cross and vol_surge and price < self.ema50[-1]:
                if vol_guard:
                    print(f"🌙 Volatility guard active (ATR%={atr_pct:.1f}), skipping short 🚀")
                else:
                    stop = max(self.kc_upper[-1], price + 2 * self.atr14[-1])
                    risk = stop - price
                    if risk > 0:
                        size = int(round((self.equity * self.risk_pct) / risk))
                        if size > 0 and size < self.equity / price:
                            self.sell(size=size)
                            self.entry_bar = len(self.data) - 1
                            self.entry_price = price
                            self.stop_price = stop
                            print(f"🚀🌙 SHORT entry @ {price:.2f}, size={size}, stop={stop:.2f} ✨")


print("🌙 Running VortexSurge backtest... ✨🚀")

bt = Backtest(
    data,
    VortexSurge,
    cash=1_000_000,
    commission=0.002,
    exclusive_orders=True
)

stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙 VortexSurge backtest complete! ✨🚀")