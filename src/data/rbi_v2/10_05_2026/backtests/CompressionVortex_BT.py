import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# -----------------------------------------------
# 🌙 Moon Dev CompressionVortex Backtest 🌙
# -----------------------------------------------

data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"

print("🌙✨ Loading Moon Dev data from the cosmos... 🚀")
data = pd.read_csv(data_path)

# Clean columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Map to backtesting requirements
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

# Parse datetime
if 'datetime' in data.columns:
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')

data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
print(f"🌙✨ Data loaded: {len(data)} bars of pure lunar energy 🚀")


class CompressionVortex(Strategy):
    # Strategy parameters
    bb_period = 20
    bb_std = 2.0
    bbw_lookback = 126          # ~6 months of trading days (on 15m we scale)
    bbw_pct_threshold = 20      # 20th percentile
    atr_period = 14
    atr_exit_mult = 1.5
    atr_stop_mult = 1.5
    vol_ma_period = 20
    vol_surge_mult = 1.5
    time_stop_bars = 12
    risk_pct = 0.02

    def init(self):
        print("🌙 Initializing CompressionVortex indicators... ✨")
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        # Bollinger Bands
        self.bb_mid = self.I(talib.SMA, close, timeperiod=self.bb_period)
        self.bb_stddev = self.I(talib.STDDEV, close, timeperiod=self.bb_period, nbdev=1)
        # Upper/Lower computed manually from mid +- std*2 (talib BBANDS also possible)
        self.bb_upper = self.I(lambda m, s: m + self.bb_std * s, self.bb_mid, self.bb_stddev)
        self.bb_lower = self.I(lambda m, s: m - self.bb_std * s, self.bb_mid, self.bb_stddev)

        # BBW
        self.bbw = self.I(lambda u, l, m: (u - l) / m, self.bb_upper, self.bb_lower, self.bb_mid)

        # BBW rolling percentile - use a rolling rank approach
        bbw_series = pd.Series(self.bbw)
        def rolling_pct_rank(x):
            if len(x) < 2:
                return 50.0
            return (x[:-1] < x[-1]).mean() * 100.0
        self.bbw_pct = self.I(
            lambda arr: pd.Series(arr).rolling(self.bbw_lookback, min_periods=20).apply(
                lambda x: (x < x[-1]).mean() * 100.0, raw=True
            ).values,
            self.bbw
        )

        # VW-MACD: weight price by normalized volume
        vol_norm = self.I(lambda v: v / pd.Series(v).rolling(self.vol_ma_period, min_periods=1).mean().values, volume)
        vw_price = self.I(lambda c, vn: c * vn, close, vol_norm)

        self.macd, self.macd_signal, self.macd_hist = self.I(
            talib.MACD, vw_price, fastperiod=12, slowperiod=26, signalperiod=9
        )

        # ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)
        self.atr_ma = self.I(talib.SMA, self.atr, timeperiod=20)

        # Volume MA
        self.vol_ma = self.I(talib.SMA, volume, timeperiod=self.vol_ma_period)

        self.entry_bar = None
        print("🌙✨ Indicators ready! Let the vortex begin 🚀")

    def next(self):
        price = self.data.Close[-1]

        # Skip if indicators not ready
        if (np.isnan(self.bbw_pct[-1]) or np.isnan(self.macd_hist[-1])
                or np.isnan(self.atr[-1]) or np.isnan(self.atr_ma[-1])
                or np.isnan(self.vol_ma[-1]) or np.isnan(self.bb_mid[-1])):
            return

        # ---- Manage existing position ----
        if self.position:
            bars_held = len(self.data) - self.entry_bar if self.entry_bar else 0

            if self.position.is_long:
                # Primary exit: close above upper band
                if self.data.Close[-1] > self.bb_upper[-1]:
                    print(f"🌙✨ LONG BB BREAKOUT EXIT @ {price:.2f} 🚀")
                    self.position.close()
                    self.entry_bar = None
                    return
                # Secondary: ATR overheat
                if self.atr[-1] > self.atr_exit_mult * self.atr_ma[-1]:
                    print(f"🌙🔥 LONG ATR OVERHEAT EXIT @ {price:.2f} 🚀")
                    self.position.close()
                    self.entry_bar = None
                    return
                # Time stop
                if bars_held >= self.time_stop_bars:
                    print(f"🌙⏰ LONG TIME STOP EXIT @ {price:.2f} 🚀")
                    self.position.close()
                    self.entry_bar = None
                    return

            elif self.position.is_short:
                if self.data.Close[-1] < self.bb_lower[-1]:
                    print(f"🌙✨ SHORT BB BREAKOUT EXIT @ {price:.2f} 🚀")
                    self.position.close()
                    self.entry_bar = None
                    return
                if self.atr[-1] > self.atr_exit_mult * self.atr_ma[-1]:
                    print(f"🌙🔥 SHORT ATR OVERHEAT EXIT @ {price:.2f} 🚀")
                    self.position.close()
                    self.entry_bar = None
                    return
                if bars_held >= self.time_stop_bars:
                    print(f"🌙⏰ SHORT TIME STOP EXIT @ {price:.2f} 🚀")
                    self.position.close()
                    self.entry_bar = None
                    return
            return

        # ---- Entry Logic ----
        compressed = self.bbw_pct[-1] < self.bbw_pct_threshold
        vol_surge = self.data.Volume[-1] > self.vol_surge_mult * self.vol_ma[-1]

        hist_now = self.macd_hist[-1]
        hist_prev = self.macd_hist[-2] if len(self.macd_hist) > 1 else 0

        # Momentum surge: hist flips positive OR accelerates upward
        long_surge = (hist_now > 0 and hist_prev <= 0) or (hist_now > hist_prev and hist_now > 0)
        short_surge = (hist_now < 0 and hist_prev >= 0) or (hist_now < hist_prev and hist_now < 0)

        above_mid = self.data.Close[-1] > self.bb_mid[-1]
        below_mid = self.data.Close[-1] < self.bb_mid[-1]

        if compressed and vol_surge:
            if long_surge and above_mid:
                # Position sizing based on ATR risk
                risk_amount = self.equity * self.risk_pct
                stop_dist = self.atr_stop_mult * self.atr[-1]
                if stop_dist <= 0:
                    return
                size = risk_amount / stop_dist
                size = int(round(size))
                if size < 1:
                    size = 1
                print(f"🌙🚀 LONG VORTEX ENTRY @ {price:.2f} | BBW%={self.bbw_pct[-1]:.1f} | size={size} ✨")
                self.buy(size=size)
                self.entry_bar = len(self.data)

            elif short_surge and below_mid:
                risk_amount = self.equity * self.risk_pct
                stop_dist = self.atr_stop_mult * self.atr[-1]
                if stop_dist <= 0:
                    return
                size = risk_amount / stop_dist
                size = int(round(size))
                if size < 1:
                    size = 1
                print(f"🌙🚀 SHORT VORTEX ENTRY @ {price:.2f} | BBW%={self.bbw_pct[-1]:.1f} | size={size} ✨")
                self.sell(size=size)
                self.entry_bar = len(self.data)


print("🌙✨ Launching CompressionVortex backtest... 🚀")
bt = Backtest(data, CompressionVortex, cash=1_000_000, commission=0.002)
stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙✨ Backtest complete! Moon Dev out 🚀")