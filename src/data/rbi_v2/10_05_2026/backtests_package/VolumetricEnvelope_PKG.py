import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

print("🌙 Moon Dev's VolumetricEnvelope Backtest Loading... ✨🚀")

# Load data
data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper column mapping
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

# Set datetime index
if 'datetime' in data.columns:
    data['datetime'] = pd.to_datetime(data['datetime'])
    data.set_index('datetime', inplace=True)

data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

print(f"🌙 Data loaded: {len(data)} bars ✨")


# ============ Indicator helper functions (module-level, no backtesting.lib) ============

def rolling_vwap(close, volume, window):
    c = pd.Series(np.asarray(close, dtype=float))
    v = pd.Series(np.asarray(volume, dtype=float))
    tp = c  # close as typical price proxy
    pv = (tp * v).rolling(window).sum()
    vv = v.rolling(window).sum()
    return (pv / vv).values


def rolling_vwap_std(close, volume, window):
    c = pd.Series(np.asarray(close, dtype=float))
    v = pd.Series(np.asarray(volume, dtype=float))
    pv = (c * v).rolling(window).sum()
    vv = v.rolling(window).sum()
    vwap = pv / vv
    diff = c - vwap
    return diff.rolling(window).std().values


def bb_upper_func(close, period, std):
    u, m, l = talib.BBANDS(np.asarray(close, dtype=float),
                           timeperiod=period, nbdevup=std, nbdevdn=std, matype=0)
    return u


def bb_middle_func(close, period, std):
    u, m, l = talib.BBANDS(np.asarray(close, dtype=float),
                           timeperiod=period, nbdevup=std, nbdevdn=std, matype=0)
    return m


def bb_lower_func(close, period, std):
    u, m, l = talib.BBANDS(np.asarray(close, dtype=float),
                           timeperiod=period, nbdevup=std, nbdevdn=std, matype=0)
    return l


def bbw_calc(upper, middle, lower):
    u = np.asarray(upper, dtype=float)
    m = np.asarray(middle, dtype=float)
    l = np.asarray(lower, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        return (u - l) / m


def upper_env_func(close, volume, window, k):
    vwap = rolling_vwap(close, volume, window)
    std = rolling_vwap_std(close, volume, window)
    return vwap + k * std


def lower_env_func(close, volume, window, k):
    vwap = rolling_vwap(close, volume, window)
    std = rolling_vwap_std(close, volume, window)
    return vwap - k * std


class VolumetricEnvelope(Strategy):
    vwap_window = 30
    k_envelope = 2.0
    obv_ma_period = 20
    obv_slope_period = 7
    bb_period = 20
    bb_std = 2.0
    bbw_lookback = 150
    bbw_pct = 0.85
    atr_period = 14
    vol_ma_period = 20
    vol_mult = 1.5
    risk_pct = 0.01
    time_exit_bars = 50

    def init(self):
        print("🌙 Initializing VolumetricEnvelope indicators... ✨")

        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        # VWAP rolling
        self.vwap = self.I(rolling_vwap, close, volume, self.vwap_window, name="VWAP")

        # Std dev of (price - vwap)
        self.vwap_std = self.I(rolling_vwap_std, close, volume, self.vwap_window, name="VWAP_STD")

        # Envelope bands (computed from close/volume arrays, not from self.vwap)
        self.upper_env = self.I(upper_env_func, close, volume, self.vwap_window, self.k_envelope, name="UpperEnv")
        self.lower_env = self.I(lower_env_func, close, volume, self.vwap_window, self.k_envelope, name="LowerEnv")

        # OBV
        self.obv = self.I(talib.OBV, close, volume, name="OBV")
        self.obv_ma = self.I(talib.SMA, self.obv, timeperiod=self.obv_ma_period, name="OBV_MA")

        # Bollinger Bands (each band extracted separately — talib.BBANDS returns a tuple)
        self.bb_upper = self.I(bb_upper_func, close, self.bb_period, self.bb_std, name="BB_Upper")
        self.bb_middle = self.I(bb_middle_func, close, self.bb_period, self.bb_std, name="BB_Middle")
        self.bb_lower = self.I(bb_lower_func, close, self.bb_period, self.bb_std, name="BB_Lower")

        # BBW
        self.bbw = self.I(bbw_calc, self.bb_upper, self.bb_middle, self.bb_lower, name="BBW")

        # ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period, name="ATR")

        # Volume MA
        self.vol_ma = self.I(talib.SMA, volume, timeperiod=self.vol_ma_period, name="Vol_MA")

        # State
        self.entry_bar = None

    def bbw_percentile(self, idx):
        if idx < self.bbw_lookback:
            return np.nan
        window = self.bbw[idx - self.bbw_lookback:idx]
        window = window[~np.isnan(window)]
        if len(window) < 20:
            return np.nan
        return np.percentile(window, self.bbw_pct * 100)

    def next(self):
        i = len(self.data) - 1
        if i < max(self.vwap_window, self.bb_period, self.bbw_lookback, self.obv_ma_period) + 5:
            return

        price = self.data.Close[-1]
        vwap = self.vwap[-1]
        upper_env = self.upper_env[-1]
        lower_env = self.lower_env[-1]
        obv = self.obv[-1]
        obv_ma = self.obv_ma[-1]
        bbw = self.bbw[-1]
        bb_upper = self.bb_upper[-1]
        bb_lower = self.bb_lower[-1]
        atr = self.atr[-1]
        vol = self.data.Volume[-1]
        vol_ma = self.vol_ma[-1]

        if any(np.isnan(x) for x in [vwap, upper_env, lower_env, obv, obv_ma, bbw, bb_upper, bb_lower, atr, vol_ma]):
            return

        bbw_thresh = self.bbw_percentile(i)

        # OBV slope
        if i >= self.obv_slope_period:
            obv_slope = self.obv[-1] - self.obv[-1 - self.obv_slope_period]
        else:
            obv_slope = 0

        # OBV higher high check
        obv_recent_high = np.nanmax(self.obv[-self.obv_slope_period:]) if i >= self.obv_slope_period else obv
        obv_prev_high = np.nanmax(self.obv[-2 * self.obv_slope_period:-self.obv_slope_period]) if i >= 2 * self.obv_slope_period else obv
        obv_higher_high = obv_recent_high > obv_prev_high

        # ============ LONG ENTRY ============
        if not self.position:
            long_breakout = price > upper_env
            obv_confirm = obv > obv_ma and obv_slope > 0 and obv_higher_high
            vol_confirm = vol > self.vol_mult * vol_ma
            not_overextended = (not np.isnan(bbw_thresh)) and bbw < bbw_thresh

            if long_breakout and obv_confirm and vol_confirm and not_overextended:
                stop = min(lower_env, price - 1.5 * atr)
                risk = price - stop
                if risk > 0:
                    size = int(round(1_000_000 / price))
                    if size > 0:
                        print(f"🚀🌙 LONG ENTRY! Price={price:.2f} UpperEnv={upper_env:.2f} OBV={obv:.0f} BBW={bbw:.4f} Size={size}")
                        self.buy(size=size)
                        self.entry_bar = i

            # ============ SHORT ENTRY ============
            short_breakout = price < lower_env
            obv_confirm_short = obv < obv_ma and obv_slope < 0
            vol_confirm_short = vol > self.vol_mult * vol_ma

            if short_breakout and obv_confirm_short and vol_confirm_short and not_overextended:
                stop = max(upper_env, price + 1.5 * atr)
                risk = stop - price
                if risk > 0:
                    size = int(round(1_000_000 / price))
                    if size > 0:
                        print(f"🔻🌙 SHORT ENTRY! Price={price:.2f} LowerEnv={lower_env:.2f} OBV={obv:.0f} BBW={bbw:.4f} Size={size}")
                        self.sell(size=size)
                        self.entry_bar = i

        # ============ EXITS ============
        else:
            # LONG EXIT
            if self.position.is_long:
                primary_exit = (not np.isnan(bbw_thresh)) and bbw > bbw_thresh and price < bb_upper
                trail_exit = price < vwap
                stop_hit = price < lower_env
                time_exit = self.entry_bar is not None and (i - self.entry_bar) > self.time_exit_bars

                if primary_exit:
                    print(f"✨🌙 LONG EXIT - BBW Expansion! Price={price:.2f} BBW={bbw:.4f} Thresh={bbw_thresh:.4f}")
                    self.position.close()
                elif trail_exit:
                    print(f"🌙 LONG EXIT - Closed below VWAP! Price={price:.2f} VWAP={vwap:.2f}")
                    self.position.close()
                elif stop_hit:
                    print(f"🛑 LONG STOP HIT! Price={price:.2f} LowerEnv={lower_env:.2f}")
                    self.position.close()
                elif time_exit:
                    print(f"⏰ LONG TIME EXIT! Bars={i - self.entry_bar}")
                    self.position.close()

            # SHORT EXIT
            elif self.position.is_short:
                primary_exit = (not np.isnan(bbw_thresh)) and bbw > bbw_thresh and price > bb_lower
                trail_exit = price > vwap
                stop_hit = price > upper_env
                time_exit = self.entry_bar is not None and (i - self.entry_bar) > self.time_exit_bars

                if primary_exit:
                    print(f"✨🌙 SHORT EXIT - BBW Expansion! Price={price:.2f} BBW={bbw:.4f}")
                    self.position.close()
                elif trail_exit:
                    print(f"🌙 SHORT EXIT - Closed above VWAP! Price={price:.2f} VWAP={vwap:.2f}")
                    self.position.close()
                elif stop_hit:
                    print(f"🛑 SHORT STOP HIT! Price={price:.2f} UpperEnv={upper_env:.2f}")
                    self.position.close()
                elif time_exit:
                    print(f"⏰ SHORT TIME EXIT! Bars={i - self.entry_bar}")
                    self.position.close()


print("🌙 Setting up backtest... ✨")
bt = Backtest(data, VolumetricEnvelope, cash=1_000_000, commission=0.0002, exclusive_orders=True)

print("🚀 Running backtest... 🌙")
stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙✨ Backtest complete! 🚀")