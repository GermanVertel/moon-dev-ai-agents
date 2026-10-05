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
    'datetime': 'Datetime',
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

data['Datetime'] = pd.to_datetime(data['Datetime'])
data = data.set_index('Datetime')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

# Ensure numeric dtypes (talib requires double/float64)
for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    data[col] = pd.to_numeric(data[col], errors='coerce').astype('float64')

data = data.dropna()

print("🌙 Moon Dev VolumetricBandBreaker initializing... ✨")
print(f"🚀 Data loaded: {len(data)} bars")


class VolumetricBandBreaker(Strategy):
    # Strategy parameters
    vwap_period = 96          # rolling VWAP period (~24h on 15m)
    dev_period = 96           # std dev of deviation period
    k_mult = 2.0              # volatility multiplier
    vol_sma_period = 20       # volume average period
    vol_multiplier = 1.5      # volume surge multiplier
    atr_period = 14
    atr_stop_mult = 1.5
    risk_pct = 0.01           # 1% risk per trade
    time_stop_bars = 40       # time stop
    slope_period = 10         # VWAP slope lookback

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        # Rolling VWAP using typical price
        tp = (high + low + close) / 3.0
        tp_vol = tp * volume

        # Rolling sums via talib SMA (sum = sma * n)
        tp_vol_sum = self.I(talib.SMA, np.asarray(tp_vol, dtype=np.float64),
                            timeperiod=self.vwap_period, name='TPVOL_SMA')
        vol_sum = self.I(talib.SMA, np.asarray(volume, dtype=np.float64),
                         timeperiod=self.vwap_period, name='VOL_SMA_VWAP')

        # VWAP = sum(tp*vol) / sum(vol)
        def vwap_calc(tpv, v):
            tpv = np.asarray(tpv, dtype=np.float64)
            v = np.asarray(v, dtype=np.float64)
            with np.errstate(divide='ignore', invalid='ignore'):
                out = tpv / v
            return out

        self.vwap = self.I(vwap_calc, tp_vol_sum, vol_sum, name='VWAP')

        # Deviation from VWAP
        close_arr = np.asarray(close, dtype=np.float64)
        vwap_arr = np.asarray(self.vwap, dtype=np.float64)
        deviation = close_arr - vwap_arr

        self.dev_std = self.I(talib.STDDEV, deviation, timeperiod=self.dev_period,
                              nbdev=1, name='DEV_STD')

        # Bands
        def upper_band(v, s):
            v = np.asarray(v, dtype=np.float64)
            s = np.asarray(s, dtype=np.float64)
            return v + self.k_mult * s

        def lower_band(v, s):
            v = np.asarray(v, dtype=np.float64)
            s = np.asarray(s, dtype=np.float64)
            return v - self.k_mult * s

        self.upper = self.I(upper_band, self.vwap, self.dev_std, name='UPPER')
        self.lower = self.I(lower_band, self.vwap, self.dev_std, name='LOWER')

        # Volume SMA
        self.vol_sma = self.I(talib.SMA, np.asarray(volume, dtype=np.float64),
                              timeperiod=self.vol_sma_period, name='VOL_SMA')

        # ATR
        self.atr = self.I(talib.ATR,
                          np.asarray(high, dtype=np.float64),
                          np.asarray(low, dtype=np.float64),
                          np.asarray(close, dtype=np.float64),
                          timeperiod=self.atr_period, name='ATR')

        # VWAP slope (simple diff over slope_period)
        def vwap_slope(v):
            v = np.asarray(v, dtype=np.float64)
            out = np.full_like(v, np.nan, dtype=np.float64)
            out[self.slope_period:] = v[self.slope_period:] - v[:-self.slope_period]
            return out

        self.vwap_slope = self.I(vwap_slope, self.vwap, name='VWAP_SLOPE')

        print("🌙 Indicators initialized: VWAP, Bands, Vol SMA, ATR ✨")

    def next(self):
        # Skip if indicators not ready
        if len(self.data) < max(self.vwap_period, self.dev_period, self.atr_period) + 5:
            return

        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        vol = self.data.Volume[-1]

        vwap = self.vwap[-1]
        upper = self.upper[-1]
        lower = self.lower[-1]
        vol_sma = self.vol_sma[-1]
        atr = self.atr[-1]
        slope = self.vwap_slope[-1]

        if any(np.isnan(x) for x in [vwap, upper, lower, vol_sma, atr, slope]):
            return

        vol_surge = vol >= self.vol_multiplier * vol_sma

        # Manage open positions
        if self.position:
            bars_held = len(self.data) - self.position.entry_bar
            if self.position.is_long:
                # Exit at VWAP revert, time stop
                if price <= vwap:
                    print(f"🌙 LONG EXIT @ VWAP revert {price:.2f} (VWAP={vwap:.2f}) ✨")
                    self.position.close()
                elif bars_held >= self.time_stop_bars:
                    print(f"⏰ LONG TIME STOP after {bars_held} bars 🚀")
                    self.position.close()
            elif self.position.is_short:
                if price >= vwap:
                    print(f"🌙 SHORT EXIT @ VWAP revert {price:.2f} (VWAP={vwap:.2f}) ✨")
                    self.position.close()
                elif bars_held >= self.time_stop_bars:
                    print(f"⏰ SHORT TIME STOP after {bars_held} bars 🚀")
                    self.position.close()
            return

        # Entry logic
        if vol_surge and atr > 0:
            # Long: close above upper band + rising VWAP
            if price > upper and slope > 0:
                stop_dist = self.atr_stop_mult * atr
                stop_price = price - stop_dist
                risk_per_unit = price - stop_price
                if risk_per_unit <= 0:
                    return
                risk_amount = self.equity * self.risk_pct
                size = int(round(risk_amount / risk_per_unit))
                if size <= 0:
                    return
                print(f"🚀 LONG BREAKOUT @ {price:.2f} | Upper={upper:.2f} | VolSurge={vol:.0f}>{self.vol_multiplier*vol_sma:.0f} | Size={size} 🌙")
                self.buy(size=size, sl=stop_price)

            # Short: close below lower band + falling VWAP
            elif price < lower and slope < 0:
                stop_dist = self.atr_stop_mult * atr
                stop_price = price + stop_dist
                risk_per_unit = stop_price - price
                if risk_per_unit <= 0:
                    return
                risk_amount = self.equity * self.risk_pct
                size = int(round(risk_amount / risk_per_unit))
                if size <= 0:
                    return
                print(f"🚀 SHORT BREAKOUT @ {price:.2f} | Lower={lower:.2f} | VolSurge={vol:.0f}>{self.vol_multiplier*vol_sma:.0f} | Size={size} 🌙")
                self.sell(size=size, sl=stop_price)


bt = Backtest(data, VolumetricBandBreaker, cash=1_000_000, commission=0.0002, exclusive_orders=True)

stats = bt.run()
print(stats)
print(stats._strategy)