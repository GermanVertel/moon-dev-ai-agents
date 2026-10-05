import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Load data
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
data = data.dropna()

print("🌙 Moon Dev CotrendDeltaNeutral backtest initializing... ✨")
print(f"📊 Data loaded: {len(data)} bars 🚀")


class CotrendDeltaNeutral(Strategy):
    # Strategy parameters
    corr_period = 20
    atr_period = 14
    corr_entry_threshold = -0.65
    corr_exit_threshold = -0.40
    rr_entry_min = 3.0
    rr_exit_min = 1.5
    risk_pct = 0.02
    time_stop_bars = 40  # ~10 trading days on 15m bars (approx)
    size_units = 1000000

    def init(self):
        print("🌙 Initializing indicators... ✨")
        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)

        # ATR for stop/target calibration
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        # Rolling correlation between returns and inverse of lagged returns (co-trend proxy)
        def rolling_corr(arr):
            s = pd.Series(arr)
            r = s.pct_change()
            inv_r = -r.shift(1)
            return r.rolling(self.corr_period).corr(inv_r).values

        self.corr = self.I(rolling_corr, close.values)

        # VIX term structure proxy: ATR/Close ratio smoothed
        def vol_regime_fn(atr_arr, close_arr):
            atr_s = pd.Series(atr_arr)
            close_s = pd.Series(close_arr)
            vol_ratio = atr_s / close_s
            return vol_ratio.rolling(20).mean().values

        self.vol_regime = self.I(vol_regime_fn, self.atr, close.values)

        # Swing highs/lows for entry logic
        self.swing_high = self.I(talib.MAX, high, timeperiod=10)
        self.swing_low = self.I(talib.MIN, low, timeperiod=10)

        # Beta of spread (proxy: beta of returns vs inverse of lagged returns)
        def rolling_beta(arr):
            s = pd.Series(arr)
            r = s.pct_change()
            inv_r = -r.shift(1)
            cov = r.rolling(self.corr_period).cov(inv_r)
            var = inv_r.rolling(self.corr_period).var()
            return (cov / var).values

        self.beta = self.I(rolling_beta, close.values)

        # Track entry bar for time stop
        self.entry_bar = None
        self.entry_price = None
        self.stop_price = None
        self.target_price = None
        self.trade_dir = None

        print("🚀 Indicators ready!")

    def next(self):
        price = self.data.Close[-1]
        atr_val = self.atr[-1]
        corr_val = self.corr[-1]
        beta_val = self.beta[-1]
        vol_regime_val = self.vol_regime[-1]

        if np.isnan(atr_val) or np.isnan(corr_val) or np.isnan(beta_val) or np.isnan(vol_regime_val):
            return

        # Time stop check
        if self.position and self.entry_bar is not None:
            bars_held = len(self.data) - self.entry_bar
            if bars_held >= self.time_stop_bars:
                print(f"⏰ Moon Dev TIME STOP hit after {bars_held} bars 🌙 Closing position")
                self.position.close()
                self.entry_bar = None
                return

        # Exit on correlation decoupling
        if self.position and corr_val > self.corr_exit_threshold:
            print(f"⚠️ Correlation decoupled ({corr_val:.2f} > {self.corr_exit_threshold}) — Moon Dev risk-off exit 🌙")
            self.position.close()
            self.entry_bar = None
            return

        # Exit on stop/target
        if self.position and self.stop_price and self.target_price:
            if self.trade_dir == 'long':
                if self.data.Low[-1] <= self.stop_price:
                    print(f"🛑 Moon Dev STOP hit at {self.stop_price:.2f} 🌙")
                    self.position.close()
                    self.entry_bar = None
                    return
                if self.data.High[-1] >= self.target_price:
                    print(f"🎯 Moon Dev TARGET hit at {self.target_price:.2f} 🚀")
                    self.position.close()
                    self.entry_bar = None
                    return
            else:
                if self.data.High[-1] >= self.stop_price:
                    print(f"🛑 Moon Dev STOP hit at {self.stop_price:.2f} 🌙")
                    self.position.close()
                    self.entry_bar = None
                    return
                if self.data.Low[-1] <= self.target_price:
                    print(f"🎯 Moon Dev TARGET hit at {self.target_price:.2f} 🚀")
                    self.position.close()
                    self.entry_bar = None
                    return

        if self.position:
            return

        # Entry conditions
        # Co-trendality confirmed
        cotrend_ok = corr_val < self.corr_entry_threshold

        # Regime filter: avoid extreme vol
        regime_ok = vol_regime_val < 0.05  # low vol regime

        if not (cotrend_ok and regime_ok):
            return

        # Long spread: higher high; Short spread: lower low
        higher_high = self.data.High[-1] >= self.swing_high[-1]
        lower_low = self.data.Low[-1] <= self.swing_low[-1]

        # ATR-based stop and target
        stop_dist = 1.5 * atr_val
        target_dist = 3.0 * stop_dist  # 3:1 RR

        if higher_high and beta_val > 0:
            # Long spread
            stop_price = price - stop_dist
            target_price = price + target_dist
            rr = (target_price - price) / (price - stop_price)
            if rr >= self.rr_entry_min:
                size = int(round(self.size_units))
                print(f"🌙 Moon Dev LONG spread signal ✨ price={price:.2f} corr={corr_val:.2f} beta={beta_val:.2f} RR={rr:.2f} size={size} 🚀")
                self.buy(size=size)
                self.entry_bar = len(self.data)
                self.entry_price = price
                self.stop_price = stop_price
                self.target_price = target_price
                self.trade_dir = 'long'

        elif lower_low and beta_val > 0:
            # Short spread
            stop_price = price + stop_dist
            target_price = price - target_dist
            rr = (price - target_price) / (stop_price - price)
            if rr >= self.rr_entry_min:
                size = int(round(self.size_units))
                print(f"🌙 Moon Dev SHORT spread signal ✨ price={price:.2f} corr={corr_val:.2f} beta={beta_val:.2f} RR={rr:.2f} size={size} 🚀")
                self.sell(size=size)
                self.entry_bar = len(self.data)
                self.entry_price = price
                self.stop_price = stop_price
                self.target_price = target_price
                self.trade_dir = 'short'


bt = Backtest(data, CotrendDeltaNeutral, cash=10000000, commission=0.002)
stats = bt.run()
print(stats)
print(stats._strategy)