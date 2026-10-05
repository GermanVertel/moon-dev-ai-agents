import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# ============================================================
# 🌙 MOON DEV BACKTEST - ConfluencePairMomentum
# ============================================================

def compute_ci(high, low, close, window=50, ema_smooth=5):
    """Compute Confluence Index (0-100) blending trend, consistency, momentum."""
    high = pd.Series(np.asarray(high, dtype=float))
    low = pd.Series(np.asarray(low, dtype=float))
    close = pd.Series(np.asarray(close, dtype=float))

    adx = talib.ADX(high.values, low.values, close.values, timeperiod=14)
    adx = pd.Series(adx, index=close.index)

    ema = talib.EMA(close.values, timeperiod=20)
    ema = pd.Series(ema, index=close.index)
    slope = ema.diff(window) / window

    up = (close.diff() > 0).astype(float)
    consistency = up.rolling(window).mean() * 100

    ret = close.pct_change()
    mom = (ret - ret.rolling(window).mean()) / (ret.rolling(window).std() + 1e-9)
    mom = mom.rolling(window).mean()
    mom_scaled = (mom - mom.rolling(window).min()) / (
        mom.rolling(window).max() - mom.rolling(window).min() + 1e-9
    ) * 100

    slope_scaled = (slope - slope.rolling(window).min()) / (
        slope.rolling(window).max() - slope.rolling(window).min() + 1e-9
    ) * 100

    ci = 0.35 * adx + 0.25 * consistency + 0.20 * mom_scaled + 0.20 * slope_scaled
    ci = ci.ewm(span=ema_smooth, adjust=False).mean()
    return ci.clip(0, 100)


def rolling_ols_beta(y, x, window=60):
    """Rolling OLS hedge ratio beta."""
    y = pd.Series(np.asarray(y, dtype=float))
    x = pd.Series(np.asarray(x, dtype=float))
    cov = y.rolling(window).cov(x)
    var = x.rolling(window).var()
    beta = cov / (var + 1e-9)
    return beta


class ConfluencePairMomentum(Strategy):
    ci_window = 50
    ci_ema = 5
    spread_window = 30
    beta_window = 60
    corr_window = 60

    ci_high = 70
    ci_low = 30
    ci_gap_min = 25
    z_entry = 1.5
    z_exit = 0.5
    z_stop = 3.0
    harmony_gap = 10
    trend_exhaust_ci = 50
    time_stop_bars = 20

    risk_pct = 0.01
    max_drawdown = 0.08

    def init(self):
        print("🌙✨ Initializing ConfluencePairMomentum strategy...")
        close = np.asarray(self.data.Close, dtype=float)
        high = np.asarray(self.data.High, dtype=float)
        low = np.asarray(self.data.Low, dtype=float)

        # Leg B = previous bar (proxy for paired asset)
        legB_close = pd.Series(close).shift(1).bfill().values
        legB_high = pd.Series(high).shift(1).bfill().values
        legB_low = pd.Series(low).shift(1).bfill().values

        ci_a = compute_ci(high, low, close, self.ci_window, self.ci_ema)
        ci_b = compute_ci(legB_high, legB_low, legB_close,
                          self.ci_window, self.ci_ema)

        ci_a_vals = np.asarray(ci_a.values, dtype=float)
        ci_b_vals = np.asarray(ci_b.values, dtype=float)

        log_a = np.log(pd.Series(close))
        log_b = np.log(pd.Series(legB_close))
        beta = rolling_ols_beta(log_a, log_b, self.beta_window)
        beta = beta.fillna(1.0)
        spread = log_a - beta * log_b

        beta_vals = np.asarray(beta.values, dtype=float)
        spread_vals = np.asarray(spread.values, dtype=float)

        sp_mean = spread.rolling(self.spread_window).mean()
        sp_std = spread.rolling(self.spread_window).std()
        z = (spread - sp_mean) / (sp_std + 1e-9)
        z_vals = np.asarray(z.values, dtype=float)

        corr = pd.Series(close).rolling(self.corr_window).corr(
            pd.Series(legB_close))
        corr_vals = np.asarray(corr.values, dtype=float)

        spread_series = pd.Series(spread)
        atr_spread = spread_series.diff().abs().rolling(14).mean()
        atr_spread_vals = np.asarray(atr_spread.values, dtype=float)

        self.ci_a = self.I(lambda: ci_a_vals, name="CI_A")
        self.ci_b = self.I(lambda: ci_b_vals, name="CI_B")
        self.beta = self.I(lambda: beta_vals, name="Beta")
        self.spread = self.I(lambda: spread_vals, name="Spread")
        self.z = self.I(lambda: z_vals, name="ZScore")
        self.corr = self.I(lambda: corr_vals, name="Corr")
        self.atr_spread = self.I(lambda: atr_spread_vals, name="ATR_Spread")

        self.entry_bar = None
        self.entry_z = None
        self.peak_equity = self.equity

    def next(self):
        i = len(self.data) - 1
        if i < max(self.ci_window, self.beta_window, self.spread_window) + 5:
            return

        ci_a = self.ci_a[-1]
        ci_b = self.ci_b[-1]
        z = self.z[-1]
        corr = self.corr[-1]
        beta = self.beta[-1]

        if np.isnan(ci_a) or np.isnan(ci_b) or np.isnan(z):
            return

        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
        dd = (self.peak_equity - self.equity) / self.peak_equity
        if dd > self.max_drawdown:
            if self.position:
                print(f"🛑 Circuit breaker! DD={dd:.2%}. Closing position.")
                self.position.close()
            return

        if not np.isnan(corr) and corr < 0.5:
            if self.position:
                print(f"⚠️ Correlation dropped to {corr:.2f} < 0.5. Closing pair.")
                self.position.close()
            return

        ci_gap = abs(ci_a - ci_b)

        # ============ ENTRY ============
        if not self.position:
            long_signal = (ci_a > self.ci_high and ci_b < self.ci_low
                           and ci_gap > self.ci_gap_min and z < -self.z_entry)
            short_signal = (ci_a < self.ci_low and ci_b > self.ci_high
                            and ci_gap > self.ci_gap_min and z > self.z_entry)

            if long_signal:
                size = 0.95
                print(f"🚀🌙 LONG A / SHORT B | CI_A={ci_a:.1f} CI_B={ci_b:.1f} "
                      f"gap={ci_gap:.1f} z={z:.2f} β={beta:.3f}")
                self.buy(size=size)
                self.entry_bar = i
                self.entry_z = z

            elif short_signal:
                size = 0.95
                print(f"🔻🌙 SHORT A / LONG B | CI_A={ci_a:.1f} CI_B={ci_b:.1f} "
                      f"gap={ci_gap:.1f} z={z:.2f} β={beta:.3f}")
                self.sell(size=size)
                self.entry_bar = i
                self.entry_z = z

        # ============ EXIT ============
        else:
            bars_held = i - self.entry_bar
            exit_now = False
            reason = ""

            if ci_gap < self.harmony_gap:
                exit_now = True
                reason = f"harmony (gap={ci_gap:.1f})"
            elif abs(z) < self.z_exit:
                exit_now = True
                reason = f"z-revert (z={z:.2f})"
            elif self.position.is_long and ci_a < self.trend_exhaust_ci:
                exit_now = True
                reason = f"trend exhaust CI_A={ci_a:.1f}"
            elif self.position.is_short and ci_b < self.trend_exhaust_ci:
                exit_now = True
                reason = f"trend exhaust CI_B={ci_b:.1f}"
            elif abs(z) > self.z_stop:
                exit_now = True
                reason = f"hard stop z={z:.2f}"
            elif bars_held >= self.time_stop_bars:
                exit_now = True
                reason = f"time stop ({bars_held} bars)"

            if exit_now:
                print(f"✅ Exiting: {reason}")
                self.position.close()
                self.entry_bar = None
                self.entry_z = None


# ============================================================
# DATA LOADING
# ============================================================
data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
print("🌙 Loading data...")
data = pd.read_csv(data_path)
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
print(f"🌙 Data loaded: {len(data)} bars")

# ============================================================
# RUN BACKTEST
# ============================================================
bt = Backtest(data, ConfluencePairMomentum, cash=1_000_000, commission=0.0002)
stats = bt.run()
print(stats)
print(stats._strategy)