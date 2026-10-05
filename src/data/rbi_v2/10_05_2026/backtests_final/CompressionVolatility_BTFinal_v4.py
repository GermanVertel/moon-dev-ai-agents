import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

print("🌙 Moon Dev's CompressionVolatility Backtest Initializing... ✨")

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

# Ensure datetime index
if 'datetime' in data.columns:
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')

# Ensure numeric dtypes for talib
for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    data[col] = pd.to_numeric(data[col], errors='coerce').astype(float)

data = data.dropna()

print(f"🌙 Data loaded: {len(data)} bars 🚀")
print(f"🌙 Columns: {list(data.columns)} ✨")


class CompressionVolatility(Strategy):
    bb_period = 20
    bb_std = 2.0
    atr_period = 14
    atr20_period = 20
    squeeze_k = 0.5
    squeeze_min_bars = 4
    atr_stop_mult = 2.0
    risk_pct = 0.01
    use_volume_filter = True
    time_stop_bars = 15

    def init(self):
        print("🌙 Initializing indicators... ✨")
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        close_arr = np.asarray(close, dtype=np.float64)
        high_arr = np.asarray(high, dtype=np.float64)
        low_arr = np.asarray(low, dtype=np.float64)
        volume_arr = np.asarray(volume, dtype=np.float64)

        def _bb_upper(arr):
            u, m, l = talib.BBANDS(arr, timeperiod=self.bb_period,
                                   nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0)
            return u

        def _bb_middle(arr):
            u, m, l = talib.BBANDS(arr, timeperiod=self.bb_period,
                                   nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0)
            return m

        def _bb_lower(arr):
            u, m, l = talib.BBANDS(arr, timeperiod=self.bb_period,
                                   nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0)
            return l

        def _atr14(h, l, c):
            return talib.ATR(h, l, c, timeperiod=self.atr_period)

        def _atr20(h, l, c):
            return talib.ATR(h, l, c, timeperiod=self.atr20_period)

        def _sma_vol(v):
            return talib.SMA(v, timeperiod=20)

        self.bb_upper = self.I(_bb_upper, close_arr, name='BB_Upper')
        self.bb_middle = self.I(_bb_middle, close_arr, name='BB_Middle')
        self.bb_lower = self.I(_bb_lower, close_arr, name='BB_Lower')

        self.atr14 = self.I(_atr14, high_arr, low_arr, close_arr, name='ATR14')
        self.atr20 = self.I(_atr20, high_arr, low_arr, close_arr, name='ATR20')
        self.vol_sma = self.I(_sma_vol, volume_arr, name='SMA(V,20)')

        # BBW = (upper - lower) / middle
        def _bbw(upper, middle, lower):
            upper = np.asarray(upper, dtype=np.float64)
            middle = np.asarray(middle, dtype=np.float64)
            lower = np.asarray(lower, dtype=np.float64)
            with np.errstate(divide='ignore', invalid='ignore'):
                result = (upper - lower) / middle
            return np.where(np.isfinite(result), result, np.nan)

        self.bbw = self.I(_bbw, self.bb_upper, self.bb_middle, self.bb_lower, name='BBW')

        # Squeeze threshold: k * ATR20 / Price
        def _threshold(atr20, close):
            atr20 = np.asarray(atr20, dtype=np.float64)
            close = np.asarray(close, dtype=np.float64)
            with np.errstate(divide='ignore', invalid='ignore'):
                result = self.squeeze_k * atr20 / close
            return np.where(np.isfinite(result), result, np.nan)

        self.squeeze_threshold = self.I(_threshold, self.atr20, self.data.Close, name='SqueezeThresh')

        # Squeeze condition
        def _squeeze(bbw, threshold):
            bbw = np.asarray(bbw, dtype=np.float64)
            threshold = np.asarray(threshold, dtype=np.float64)
            result = (bbw < threshold).astype(float)
            return np.where(np.isnan(bbw) | np.isnan(threshold), 0.0, result)

        self.squeeze = self.I(_squeeze, self.bbw, self.squeeze_threshold, name='Squeeze')

        # Entry state tracking
        self.squeeze_count = 0
        self.squeeze_atr_ref = np.nan
        self.entry_bar = 0
        self.stop_price = np.nan

    def next(self):
        price = self.data.Close[-1]
        low = self.data.Low[-1]

        if len(self.data) < max(self.bb_period, self.atr20_period) + 5:
            return

        # Guard against NaN indicators
        if (np.isnan(self.bb_upper[-1]) or np.isnan(self.bb_lower[-1])
                or np.isnan(self.atr14[-1]) or np.isnan(self.atr20[-1])
                or np.isnan(self.vol_sma[-1])):
            return

        # Track squeeze duration
        if self.squeeze[-1] == 1:
            self.squeeze_count += 1
        else:
            self.squeeze_count = 0

        # Manage open position
        if self.position:
            # Update trailing stop
            if not np.isnan(self.squeeze_atr_ref):
                new_stop = price - (self.atr_stop_mult * self.squeeze_atr_ref)
                if new_stop > self.stop_price:
                    self.stop_price = new_stop
                    print(f"🌙 Trailing stop raised to {self.stop_price:.2f} 🚀")

            # Check stop hit
            if low <= self.stop_price:
                print(f"🌙 💥 Stop hit at {self.stop_price:.2f}, exit at {price:.2f}")
                self.position.close()
                return

            # Time stop
            bars_in_trade = len(self.data) - self.entry_bar
            if bars_in_trade >= self.time_stop_bars:
                ref_idx = -bars_in_trade
                if abs(ref_idx) <= len(self.data) and price < self.data.Close[ref_idx]:
                    print(f"🌙 ⏰ Time stop triggered after {bars_in_trade} bars")
                    self.position.close()
                    return

        # Entry logic
        if not self.position:
            # Need confirmed squeeze (min bars) and breakout above upper band
            confirmed_squeeze = self.squeeze_count >= self.squeeze_min_bars
            breakout = price > self.bb_upper[-1]

            if confirmed_squeeze and breakout:
                # Volume confirmation
                vol_ok = True
                if self.use_volume_filter:
                    vol_ok = self.data.Volume[-1] > self.vol_sma[-1]

                # Avoid chasing: price must not be > 1x ATR above upper band
                atr_ref = self.atr14[-1]
                not_chasing = price <= self.bb_upper[-1] + atr_ref

                if vol_ok and not_chasing:
                    # Position sizing based on risk %
                    equity = self.equity
                    risk_amount = equity * self.risk_pct
                    stop_dist = self.atr_stop_mult * atr_ref
                    if stop_dist > 0:
                        size = int(round(risk_amount / stop_dist))
                        if size > 0:
                            self.squeeze_atr_ref = atr_ref
                            self.stop_price = price - stop_dist
                            self.entry_bar = len(self.data)
                            print(f"🌙 🚀 BREAKOUT ENTRY at {price:.2f} | Size: {size} | Stop: {self.stop_price:.2f} | ATR_ref: {atr_ref:.2f}")
                            self.buy(size=size)
                else:
                    if not vol_ok:
                        print("🌙 ⚠️ Breakout skipped: volume below SMA")
                    elif not not_chasing:
                        print("🌙 ⚠️ Breakout skipped: price extended too far above band")


bt = Backtest(data, CompressionVolatility, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)