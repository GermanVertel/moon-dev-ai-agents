import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from sklearn.ensemble import RandomForestClassifier

DATA_PATH = '/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv'


def load_data():
    print("🌙 Loading data from the Moon Dev vault...")
    data = pd.read_csv(DATA_PATH)
    data.columns = data.columns.str.strip().str.lower()
    data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
    if 'datetime' in data.columns:
        data['datetime'] = pd.to_datetime(data['datetime'])
        data = data.set_index('datetime')
    rename_map = {'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}
    data = data.rename(columns={k: v for k, v in rename_map.items() if k in data.columns})
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    print(f"✨ Moon data loaded: {len(data)} bars 🚀")
    return data


def rolling_had(x, window):
    """Mean Absolute Deviation over rolling window."""
    s = pd.Series(x)
    return s.rolling(window).apply(lambda w: np.mean(np.abs(w - np.mean(w))), raw=True).values


class CointegratedDeviation(Strategy):
    # --- Strategy params ---
    zscore_lookback = 60
    had_lookback = 200
    ml_lookback = 300
    z_entry = 1.5
    had_mult = 2.0
    confidence_threshold = 0.55
    max_hold = 40
    stop_had_mult = 3.0
    size = 0.95  # fraction of equity (backtesting.py requirement)

    def init(self):
        print("🌙✨ Initializing CointegratedDeviation strategy...")
        close = pd.Series(self.data.Close, index=self.data.index)

        # ✅ talib for SMA and STD
        self.sma = self.I(talib.SMA, self.data.Close, timeperiod=self.zscore_lookback, name='SMA')
        self.std = self.I(talib.STDDEV, self.data.Close, timeperiod=self.zscore_lookback, name='STD')

        # ✅ Custom indicator wrapped in self.I with explicit name
        self.had = self.I(rolling_had, self.data.Close, self.had_lookback, name='HAD')

        # ✅ talib for volume SMA
        self.vol_sma = self.I(talib.SMA, self.data.Volume, timeperiod=20, name='VOL_SMA')

        # Precompute z-scores for ML training
        self.z_series = (close - close.rolling(self.zscore_lookback).mean()) / close.rolling(
            self.zscore_lookback).std()

        # Build ML features & target on historical data
        print("🌙 Training hybrid ML ensemble (RandomForest) on rolling features...")
        feats = pd.DataFrame(index=close.index)
        feats['z'] = self.z_series
        feats['z_lag1'] = self.z_series.shift(1)
        feats['z_lag2'] = self.z_series.shift(2)
        feats['ret'] = close.pct_change()
        feats['vol'] = close.pct_change().rolling(20).std()
        feats['vol_ratio'] = self.data.Volume / pd.Series(self.data.Volume).rolling(20).mean()
        feats['had'] = pd.Series(self.had)
        feats['abs_z'] = feats['z'].abs()

        # Target: will |z| exceed 2*HAD-based threshold within next K bars?
        K = 5
        had_series = pd.Series(self.had)
        threshold = self.had_mult * had_series
        future_extreme = pd.Series(0, index=close.index)
        abs_spread = (close - close.rolling(self.zscore_lookback).mean()).abs()
        for i in range(len(close) - K):
            window = abs_spread.iloc[i + 1:i + 1 + K]
            thr = threshold.iloc[i]
            if not np.isnan(thr) and window.max() > thr:
                future_extreme.iloc[i] = 1

        feats['target'] = future_extreme
        train_df = feats.dropna()
        if len(train_df) > 500:
            X = train_df.drop(columns=['target']).values
            y = train_df['target'].values
            self.model = RandomForestClassifier(n_estimators=50, max_depth=5,
                                                random_state=42, n_jobs=-1)
            self.model.fit(X, y)
            self.model_ready = True
            print(f"🚀 ML model trained on {len(train_df)} samples. Moon Dev approved! 🌙")
        else:
            self.model_ready = False
            print("⚠️ Not enough data to train ML model.")

        self.entry_bar = None
        self.entry_z = None
        self.entry_had = None
        self.trade_count = 0

    def _predict(self, i):
        if not self.model_ready or i < 3:
            return 0.0
        z = self.z_series.iloc[i]
        z1 = self.z_series.iloc[i - 1]
        z2 = self.z_series.iloc[i - 2]
        ret = (self.data.Close[i] - self.data.Close[i - 1]) / self.data.Close[i - 1]
        vol = np.std(np.diff(self.data.Close[max(0, i - 20):i + 1]) /
                     self.data.Close[max(0, i - 20):i + 1]) if i > 20 else 0.0
        vol_ratio = self.data.Volume[i] / self.vol_sma[i] if self.vol_sma[i] and not np.isnan(
            self.vol_sma[i]) else 1.0
        had = self.had[i] if not np.isnan(self.had[i]) else 0.0
        abs_z = abs(z) if not np.isnan(z) else 0.0
        if any(np.isnan([z, z1, z2, ret, vol, vol_ratio, had])):
            return 0.0
        X = np.array([[z, z1, z2, ret, vol, vol_ratio, had, abs_z]])
        try:
            p = self.model.predict_proba(X)[0][1]
        except Exception:
            p = 0.0
        return p

    def next(self):
        i = len(self.data) - 1
        if i < max(self.zscore_lookback, self.had_lookback) + 5:
            return

        price = self.data.Close[-1]
        z = self.z_series.iloc[i]
        had = self.had[-1]

        if np.isnan(z) or np.isnan(had) or had <= 0:
            return

        # ---- Manage open position ----
        if self.position:
            bars_held = i - self.entry_bar if self.entry_bar is not None else 0
            if (self.position.is_long and z >= -0.25) or (self.position.is_short and z <= 0.25):
                print(f"🌙✨ Profit target hit! z={z:.2f} | Closing at {price:.2f} 🚀")
                self.position.close()
                self.entry_bar = None
                return

            if bars_held >= self.max_hold:
                print(f"⏰ Time stop hit ({bars_held} bars). Closing at {price:.2f} 🌙")
                self.position.close()
                self.entry_bar = None
                return

            if self.entry_z is not None and self.entry_had is not None:
                if self.position.is_long and z < self.entry_z - (self.stop_had_mult - self.had_mult) * (
                        self.entry_had / max(had, 1e-9)):
                    print(f"🛑 Stop loss hit (long). z={z:.2f} 🌙")
                    self.position.close()
                    self.entry_bar = None
                    return
                if self.position.is_short and z > self.entry_z + (self.stop_had_mult - self.had_mult) * (
                        self.entry_had / max(had, 1e-9)):
                    print(f"🛑 Stop loss hit (short). z={z:.2f} 🌙")
                    self.position.close()
                    self.entry_bar = None
                    return
            return

        # ---- Entry logic ----
        p = self._predict(i)
        if p < self.confidence_threshold:
            return

        threshold = self.had_mult * had
        spread_dev = abs(price - self.sma[-1])

        if spread_dev < threshold:
            return

        conf_scale = (p - self.confidence_threshold) / (1 - self.confidence_threshold)
        conf_scale = max(0.1, min(1.0, conf_scale))
        # Use fraction of equity (0 < size < 1) for backtesting.py compatibility
        trade_size = max(0.01, min(0.99, self.size * conf_scale))

        if z <= -self.z_entry:
            print(f"🌙🚀 LONG spread signal! z={z:.2f} p={p:.2f} size={trade_size:.3f} @ {price:.2f}")
            self.buy(size=trade_size)
            self.entry_bar = i
            self.entry_z = z
            self.entry_had = had
            self.trade_count += 1
        elif z >= self.z_entry:
            print(f"🌙🚀 SHORT spread signal! z={z:.2f} p={p:.2f} size={trade_size:.3f} @ {price:.2f}")
            self.sell(size=trade_size)
            self.entry_bar = i
            self.entry_z = z
            self.entry_had = had
            self.trade_count += 1


if __name__ == '__main__':
    print("🌙 Moon Dev Backtest Engine warming up... ✨")
    data = load_data()
    bt = Backtest(data, CointegratedDeviation, cash=1_000_000, commission=0.001, exclusive_orders=True)
    stats = bt.run()
    print(stats)
    print(stats._strategy)
    print("🌙✨ Backtest complete. To the moon! 🚀")