import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

print("🌙✨ Starting Moon Dev's RelativeSqueeze Backtest! ✨🌙")

# Load data
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Map to proper case
data.columns = data.columns.str.capitalize()
data = data.rename(columns={'Datetime': 'Datetime'})

# Ensure required columns
required = ['Open', 'High', 'Low', 'Close', 'Volume']
for col in required:
    if col not in data.columns:
        raise ValueError(f"🚨 Missing column: {col}")

# Drop rows with NaN
data = data.dropna()

# Set datetime index
if 'Datetime' in data.columns:
    data['Datetime'] = pd.to_datetime(data['Datetime'])
    data = data.set_index('Datetime')

print(f"🌙 Data loaded: {len(data)} bars ✨")
print(f"🚀 Columns: {list(data.columns)}")


class RelativeSqueeze(Strategy):
    bb_period = 20
    bb_std_mult = 2.0
    rs_period = 20
    rs_slope_period = 10
    bw_lookback = 100
    bw_squeeze_pct = 0.20
    bw_exhaust_pct = 0.85
    vol_ma_period = 20
    atr_period = 14
    risk_pct = 0.02
    time_stop_bars = 15

    def init(self):
        print("🌙 Initializing indicators for RelativeSqueeze... ✨")
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        # Bollinger Bands - using talib for all standard indicators ✅
        self.bb_mid = self.I(talib.SMA, close, timeperiod=self.bb_period, name='BB_Mid')
        self.bb_stddev = self.I(talib.STDDEV, close, timeperiod=self.bb_period, nbdev=1, name='BB_StdDev')

        # Bollinger Bands upper/lower - custom numpy wrapper (no backtesting.lib!) ✅
        std_mult = self.bb_std_mult
        self.bb_upper = self.I(lambda m, s, sm=std_mult: m + sm * s,
                               self.bb_mid, self.bb_stddev, name='BB_Upper')
        self.bb_lower = self.I(lambda m, s, sm=std_mult: m - sm * s,
                               self.bb_mid, self.bb_stddev, name='BB_Lower')

        # Bandwidth
        self.bandwidth = self.I(lambda u, l, m: (u - l) / np.where(m != 0, m, np.nan),
                                self.bb_upper, self.bb_lower, self.bb_mid, name='Bandwidth')

        # %B
        self.percent_b = self.I(lambda c, u, l: (c - l) / np.where((u - l) != 0, (u - l), np.nan),
                                close, self.bb_upper, self.bb_lower, name='PercentB')

        # Bandwidth percentile (rolling) - pure numpy, no backtesting.lib ✅
        bw_lb = self.bw_lookback

        def bw_pct(bw):
            out = np.full(len(bw), np.nan)
            for i in range(bw_lb, len(bw)):
                window = bw[i - bw_lb:i]
                window = window[~np.isnan(window)]
                if len(window) > 0:
                    out[i] = (np.sum(window < bw[i]) / len(window))
            return out

        self.bw_pct = self.I(bw_pct, self.bandwidth, name='BW_Pct')

        # RS Line (using Close momentum as benchmark proxy)
        self.rs_line = self.I(talib.SMA, close, timeperiod=5, name='RS_Line')
        self.rs_ma = self.I(talib.SMA, self.rs_line, timeperiod=self.rs_period, name='RS_MA')

        # RS slope - rolling window diff, no backtesting.lib ✅
        rs_slope_period = self.rs_slope_period
        self.rs_slope = self.I(lambda r, p=rs_slope_period: r - np.roll(r, p),
                               self.rs_line, name='RS_Slope')

        # Volume MA
        self.vol_ma = self.I(talib.SMA, volume, timeperiod=self.vol_ma_period, name='Vol_MA')

        # ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period, name='ATR')

        # Swing high/low
        self.swing_high = self.I(talib.MAX, high, timeperiod=20, name='Swing_High')
        self.swing_low = self.I(talib.MIN, low, timeperiod=20, name='Swing_Low')

        # State
        self.entry_bar = None
        self.entry_price = None

        print("🌙 Indicators initialized! ✨🚀")

    def next(self):
        if len(self.data) < self.bw_lookback + 5:
            return

        price = self.data.Close[-1]
        mid = self.bb_mid[-1]
        upper = self.bb_upper[-1]
        lower = self.bb_lower[-1]
        bw_pct = self.bw_pct[-1]
        pct_b = self.percent_b[-1]
        rs = self.rs_line[-1]
        rs_ma = self.rs_ma[-1]
        rs_slope = self.rs_slope[-1]
        vol = self.data.Volume[-1]
        vol_ma = self.vol_ma[-1]
        atr = self.atr[-1]

        if np.isnan([mid, upper, lower, bw_pct, pct_b, rs, rs_ma, rs_slope, vol_ma, atr]).any():
            return

        # Squeeze condition
        squeeze = bw_pct <= self.bw_squeeze_pct
        exhaust = bw_pct >= self.bw_exhaust_pct

        # RS conditions
        rs_rising = rs_slope > 0 and rs > rs_ma
        rs_falling = rs_slope < 0 and rs < rs_ma

        # Volume confirmation
        vol_ok = vol > vol_ma

        # ATR filter
        atr_ok = atr > 0

        # Entry logic
        if not self.position:
            # Long entry
            if squeeze and price > mid and rs_rising and pct_b > 0.5 and vol_ok and atr_ok:
                stop = min(lower, self.swing_low[-1])
                if stop < price:
                    risk = price - stop
                    if risk > 0:
                        size = int(round(1000000 / price))
                        if size > 0:
                            self.buy(size=size)
                            self.entry_bar = len(self.data)
                            self.entry_price = price
                            print(f"🌙🚀 LONG ENTRY! Price: {price:.2f} | Stop: {stop:.2f} | Size: {size} ✨")

            # Short entry
            elif squeeze and price < mid and rs_falling and pct_b < 0.5 and vol_ok and atr_ok:
                stop = max(upper, self.swing_high[-1])
                if stop > price:
                    risk = stop - price
                    if risk > 0:
                        size = int(round(1000000 / price))
                        if size > 0:
                            self.sell(size=size)
                            self.entry_bar = len(self.data)
                            self.entry_price = price
                            print(f"🌙🔻 SHORT ENTRY! Price: {price:.2f} | Stop: {stop:.2f} | Size: {size} ✨")

        # Exit logic
        else:
            bars_held = len(self.data) - self.entry_bar if self.entry_bar else 0

            if self.position.is_long:
                # Exit on close below mid + RS falling
                if price < mid and rs_falling:
                    self.position.close()
                    print(f"🌙 Exit LONG: Price {price:.2f} below mid {mid:.2f} with RS falling ✨")
                # Volatility exhaustion
                elif exhaust and pct_b > 1.0:
                    self.position.close()
                    print(f"🌙 Exit LONG: Volatility exhaustion ✨")
                # Time stop
                elif bars_held >= self.time_stop_bars:
                    self.position.close()
                    print(f"🌙 Exit LONG: Time stop ({bars_held} bars) ✨")

            elif self.position.is_short:
                # Exit on close above mid + RS rising
                if price > mid and rs_rising:
                    self.position.close()
                    print(f"🌙 Exit SHORT: Price {price:.2f} above mid {mid:.2f} with RS rising ✨")
                # Volatility exhaustion
                elif exhaust and pct_b < 0.0:
                    self.position.close()
                    print(f"🌙 Exit SHORT: Volatility exhaustion ✨")
                # Time stop
                elif bars_held >= self.time_stop_bars:
                    self.position.close()
                    print(f"🌙 Exit SHORT: Time stop ({bars_held} bars) ✨")


print("🌙✨ Setting up backtest... ✨🌙")
bt = Backtest(data, RelativeSqueeze, cash=1000000, commission=0.001)

print("🚀 Running backtest... ✨")
stats = bt.run()

print("🌙✨ Backtest Complete! ✨🌙")
print(stats)
print(stats._strategy)