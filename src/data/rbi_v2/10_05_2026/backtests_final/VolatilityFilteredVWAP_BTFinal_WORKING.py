import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

print("🌙 Moon Dev's VolatilityFiltered VWAP Backtest Initializing... ✨")

# Load data
data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper column mapping
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

# Set datetime index
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')

print(f"🚀 Data loaded: {len(data)} bars from {data.index[0]} to {data.index[-1]}")


class VolatilityFilteredVWAP(Strategy):
    # Parameters
    atr_period = 14
    atr_percentile_lookback = 100
    atr_percentile_threshold = 30
    vwap_band_mult = 1.0
    fib_level = 0.382
    swing_lookback = 10
    risk_pct = 0.01
    max_stop_atr_mult = 2.0
    time_stop_bars = 20

    def init(self):
        print("🌙 Initializing indicators...")

        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        volume = pd.Series(self.data.Volume)

        # ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        # ATR percentile rank over lookback
        def atr_percentile(atr_vals):
            s = pd.Series(atr_vals)
            return s.rolling(self.atr_percentile_lookback).apply(
                lambda x: (x.iloc[-1] > x).sum() / len(x) * 100 if len(x) > 0 else 50,
                raw=False
            ).values

        self.atr_pct = self.I(atr_percentile, self.atr)

        # Rolling VWAP using typical price
        typical = (high + low + close) / 3
        pv = typical * volume

        def rolling_vwap(pv_vals, vol_vals, window=96):
            pv_s = pd.Series(pv_vals)
            v_s = pd.Series(vol_vals)
            return (pv_s.rolling(window).sum() / v_s.rolling(window).sum()).values

        self.vwap = self.I(rolling_vwap, pv.values, volume.values, 96)

        # Swing high/low
        self.swing_high = self.I(talib.MAX, high, timeperiod=self.swing_lookback)
        self.swing_low = self.I(talib.MIN, low, timeperiod=self.swing_lookback)

        # Volume SMA for confirmation
        self.vol_sma = self.I(talib.SMA, volume, timeperiod=20)

        self.entry_bar = None
        self.entry_price = None
        self.trail_stop = None
        print("✨ Indicators ready!")

    def next(self):
        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        atr = self.atr[-1]
        atr_pct = self.atr_pct[-1]
        vwap = self.vwap[-1]
        vol = self.data.Volume[-1]
        vol_avg = self.vol_sma[-1]

        if np.isnan(atr) or np.isnan(atr_pct) or np.isnan(vwap):
            return

        # VWAP bands
        upper_band = vwap + self.vwap_band_mult * atr
        lower_band = vwap - self.vwap_band_mult * atr

        # Manage open position
        if self.position:
            entry = self.entry_price
            swing_hi = self.swing_high[-1]
            swing_lo = self.swing_low[-1]

            # Recompute Fibonacci trailing stop from swing low to swing high
            if not np.isnan(swing_hi) and not np.isnan(swing_lo) and swing_hi > swing_lo:
                fib_stop = swing_hi - (swing_hi - swing_lo) * self.fib_level
                if self.trail_stop is None or fib_stop > self.trail_stop:
                    self.trail_stop = fib_stop
                    print(f"🌙 Trail stop ratcheted up to {fib_stop:.2f}")

            # Hard exit: close below VWAP
            if price < vwap:
                print(f"🚨 VWAP hard exit at {price:.2f} (VWAP={vwap:.2f})")
                self.position.close()
                self.trail_stop = None
                self.entry_bar = None
                return

            # Trailing stop hit
            if self.trail_stop is not None and low <= self.trail_stop:
                print(f"🛑 Trailing stop hit at {self.trail_stop:.2f}")
                self.position.close()
                self.trail_stop = None
                self.entry_bar = None
                return

            # Time stop
            if self.entry_bar is not None:
                bars_held = len(self.data) - self.entry_bar
                if bars_held >= self.time_stop_bars:
                    profit = price - entry
                    if profit < atr:
                        print(f"⏰ Time stop: no 1xATR profit in {bars_held} bars")
                        self.position.close()
                        self.trail_stop = None
                        self.entry_bar = None
                        return

            # Regime kill-switch
            if atr_pct > 70:
                print(f"⚡ Regime kill-switch: ATR pct={atr_pct:.1f} > 70, flattening")
                self.position.close()
                self.trail_stop = None
                self.entry_bar = None
                return

        # Entry logic
        else:
            # Low volatility regime
            if atr_pct >= self.atr_percentile_threshold:
                return

            # Price closes above upper VWAP band
            if price > upper_band:
                # Volume confirmation
                if vol <= vol_avg:
                    return

                # Position sizing: risk fixed % of equity
                equity = self.equity
                risk_amount = equity * self.risk_pct

                swing_lo = self.swing_low[-1]
                initial_stop = swing_lo if not np.isnan(swing_lo) else price - atr
                # Cap stop distance at 2xATR
                max_stop = price - self.max_stop_atr_mult * atr
                if initial_stop < max_stop:
                    initial_stop = max_stop

                stop_dist = price - initial_stop
                if stop_dist <= 0:
                    return

                size = int(round(risk_amount / stop_dist))
                if size <= 0:
                    return

                print(f"🚀 LONG entry at {price:.2f} | VWAP={vwap:.2f} | ATR%={atr_pct:.1f} | size={size}")
                self.buy(size=size)
                self.entry_price = price
                self.trail_stop = initial_stop
                self.entry_bar = len(self.data)


bt = Backtest(data, VolatilityFilteredVWAP, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)