import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# ── Moon Dev Data Loading 🌙 ──────────────────────────────────────────────
DATA_PATH = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"

print("🌙 Moon Dev loading data from:", DATA_PATH)
data = pd.read_csv(DATA_PATH)
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data.columns = [c.capitalize() for c in data.columns]

# Ensure datetime index
if 'Datetime' in data.columns:
    data['Datetime'] = pd.to_datetime(data['Datetime'])
    data = data.set_index('Datetime')
    data.index.name = None
elif 'Date' in data.columns:
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.set_index('Date')
    data.index.name = None

data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
print("✨ Data shape:", data.shape)


class EntropicSqueeze(Strategy):
    # ── Parameters ───────────────────────────────────────────────────────
    ema_period = 20
    atr_kc_period = 10
    kc_mult = 2.0
    bb_period = 20
    bb_mult = 2.0
    atr_period = 14
    lookback = 252
    sqz_pct_threshold = 80
    entropy_pct_threshold = 20
    gap_pct_threshold = 80
    profit_atr_mult = 2.5
    stop_atr_mult = 1.0
    trail_trigger_atr = 1.5
    trail_atr_mult = 1.0
    time_stop_bars = 5
    risk_pct = 0.01
    vol_sma_period = 20

    def init(self):
        print("🌙 Moon Dev initializing EntropicSqueeze strategy ✨")
        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        volume = pd.Series(self.data.Volume)

        # ── Keltner Channel ──────────────────────────────────────────────
        ema = talib.EMA(close.values, timeperiod=self.ema_period)
        atr_kc = talib.ATR(high.values, low.values, close.values, timeperiod=self.atr_kc_period)
        kc_upper = ema + self.kc_mult * atr_kc
        kc_lower = ema - self.kc_mult * atr_kc
        kc_width = (kc_upper - kc_lower) / ema

        # ── Bollinger Bands ──────────────────────────────────────────────
        sma = talib.SMA(close.values, timeperiod=self.bb_period)
        std = talib.STDDEV(close.values, timeperiod=self.bb_period, nbdev=1)
        bb_upper = sma + self.bb_mult * std
        bb_lower = sma - self.bb_mult * std
        bb_width = (bb_upper - bb_lower) / sma

        # ── Normalized Squeeze Metric ────────────────────────────────────
        sqz = (kc_width - bb_width) / kc_width
        sqz = np.where(np.isfinite(sqz), sqz, 0.0)

        # Rolling percentile rank of SQZ
        sqz_series = pd.Series(sqz)
        sqz_rank = sqz_series.rolling(self.lookback, min_periods=20).apply(
            lambda x: (x[-1] >= x).mean() * 100 if len(x) > 0 else np.nan, raw=True
        ).values

        # ── 3-Day Volume Entropy ─────────────────────────────────────────
        # Use rolling 3-bar volume entropy (bar-level proxy for daily)
        vol_arr = volume.values
        entropy = np.full(len(vol_arr), np.nan)
        for i in range(2, len(vol_arr)):
            v = vol_arr[i - 2:i + 1]
            s = v.sum()
            if s <= 0:
                continue
            p = v / s
            p = p[p > 0]
            h = -np.sum(p * np.log(p))
            entropy[i] = h / np.log(3)

        entropy_series = pd.Series(entropy)
        entropy_rank = entropy_series.rolling(self.lookback, min_periods=20).apply(
            lambda x: (x[-1] >= x).mean() * 100 if len(x) > 0 else np.nan, raw=True
        ).values

        # ── Overnight Gap ────────────────────────────────────────────────
        open_arr = pd.Series(self.data.Open).values
        prev_close = pd.Series(self.data.Close).shift(1).values
        gap_pct = np.where(prev_close != 0, (open_arr - prev_close) / prev_close, 0.0)
        abs_gap = np.abs(gap_pct)

        abs_gap_series = pd.Series(abs_gap)
        gap_rank = abs_gap_series.rolling(self.lookback, min_periods=20).apply(
            lambda x: (x[-1] >= x).mean() * 100 if len(x) > 0 else np.nan, raw=True
        ).values

        # ── ATR for exits ────────────────────────────────────────────────
        atr = talib.ATR(high.values, low.values, close.values, timeperiod=self.atr_period)

        # ── Volume SMA sanity filter ─────────────────────────────────────
        vol_sma = talib.SMA(volume.values, timeperiod=self.vol_sma_period)

        # ── Register indicators ──────────────────────────────────────────
        self.sqz_rank = self.I(lambda: sqz_rank, name="SQZ_Rank")
        self.entropy_rank = self.I(lambda: entropy_rank, name="Entropy_Rank")
        self.gap_rank = self.I(lambda: gap_rank, name="Gap_Rank")
        self.gap_pct = self.I(lambda: gap_pct, name="Gap_Pct")
        self.atr = self.I(lambda: atr, name="ATR")
        self.vol_sma = self.I(lambda: vol_sma, name="Vol_SMA")

        print("🚀 Moon Dev indicators ready! Squeeze the entropy! 🌙")

    def next(self):
        i = len(self.data) - 1
        if i < 2:
            return

        # Skip if any indicator is NaN
        vals = [self.sqz_rank[i], self.entropy_rank[i], self.gap_rank[i],
                self.atr[i], self.vol_sma[i], self.gap_pct[i]]
        if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in vals):
            return

        # Volume sanity filter
        if self.data.Volume[i] < 0.3 * self.vol_sma[i]:
            return

        sqz_ok = self.sqz_rank[i] >= self.sqz_pct_threshold
        ent_ok = self.entropy_rank[i] <= self.entropy_pct_threshold
        gap_ok = self.gap_rank[i] >= self.gap_pct_threshold

        # ── Manage existing position ─────────────────────────────────────
        if self.position:
            entry = self.trades[-1].entry_price
            atr_val = self.atr[i]
            bars_held = len(self.data) - 1 - self.trades[-1].entry_bar

            if self.position.is_long:
                # Profit target
                if self.data.High[i] >= entry + self.profit_atr_mult * atr_val:
                    print(f"🎯 Moon Dev LONG profit target hit! Closing at {self.data.Close[i]:.2f} 🌙")
                    self.position.close()
                    return
                # Trailing stop
                if self.data.High[i] >= entry + self.trail_trigger_atr * atr_val:
                    trail_stop = self.data.High[i] - self.trail_atr_mult * atr_val
                    if self.data.Low[i] <= trail_stop:
                        print(f"📉 Moon Dev LONG trailing stop! {trail_stop:.2f} 🌙")
                        self.position.close()
                        return
                # Time stop
                if bars_held >= self.time_stop_bars:
                    print(f"⏰ Moon Dev LONG time stop after {bars_held} bars 🌙")
                    self.position.close()
                    return
            else:
                if self.data.Low[i] <= entry - self.profit_atr_mult * atr_val:
                    print(f"🎯 Moon Dev SHORT profit target hit! Closing at {self.data.Close[i]:.2f} 🌙")
                    self.position.close()
                    return
                if self.data.Low[i] <= entry - self.trail_trigger_atr * atr_val:
                    trail_stop = self.data.Low[i] + self.trail_atr_mult * atr_val
                    if self.data.High[i] >= trail_stop:
                        print(f"📉 Moon Dev SHORT trailing stop! {trail_stop:.2f} 🌙")
                        self.position.close()
                        return
                if bars_held >= self.time_stop_bars:
                    print(f"⏰ Moon Dev SHORT time stop after {bars_held} bars 🌙")
                    self.position.close()
                    return
            return

        # ── Entry logic ──────────────────────────────────────────────────
        if sqz_ok and ent_ok and gap_ok:
            atr_val = self.atr[i]
            if atr_val <= 0:
                return

            # Volatility-adjusted sizing based on risk %
            equity = self.equity
            risk_amount = equity * self.risk_pct
            size = int(round(risk_amount / atr_val))
            if size < 1:
                size = 1

            sl_dist = self.stop_atr_mult * atr_val
            tp_dist = self.profit_atr_mult * atr_val

            if self.gap_pct[i] > 0:
                sl = self.data.Close[i] - sl_dist
                tp = self.data.Close[i] + tp_dist
                print(f"🚀🌙 Moon Dev LONG ENTRY! SQZ={self.sqz_rank[i]:.1f} ENT={self.entropy_rank[i]:.1f} GAP={self.gap_rank[i]:.1f} size={size} ✨")
                self.buy(size=size, sl=sl, tp=tp)
            elif self.gap_pct[i] < 0:
                sl = self.data.Close[i] + sl_dist
                tp = self.data.Close[i] - tp_dist
                print(f"🔻🌙 Moon Dev SHORT ENTRY! SQZ={self.sqz_rank[i]:.1f} ENT={self.entropy_rank[i]:.1f} GAP={self.gap_rank[i]:.1f} size={size} ✨")
                self.sell(size=size, sl=sl, tp=tp)


print("🌙 Moon Dev launching EntropicSqueeze backtest! 🚀✨")
bt = Backtest(data, EntropicSqueeze, cash=1_000_000, commission=0.002)
stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙 Moon Dev backtest complete! ✨🚀")