import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# ============================================================
# 🌙 MOON DEV - NOCTURNAL ENGULF STRATEGY 🌙
# ============================================================

def load_data(path):
    data = pd.read_csv(path)
    data.columns = data.columns.str.strip().str.lower()
    data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
    data = data.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume',
    })
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    data = data.dropna()
    return data


class NocturnalEngulf(Strategy):
    adx_period = 14
    adx_threshold_entry = 20
    adx_threshold_exit = 25
    atr_period = 14
    range_lookback = 20
    body_ratio_max = 0.30
    lower_wick_mult = 2.0
    size_expansion = 1.10
    risk_pct = 0.01
    rr_target = 1.5
    atr_target_mult = 1.5
    atr_stop_mult = 1.0
    range_width_atr_mult = 1.5

    def init(self):
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period)
        self.adx = self.I(talib.ADX, self.data.High, self.data.Low, self.data.Close, timeperiod=self.adx_period)
        self.range_high = self.I(talib.MAX, self.data.High, timeperiod=self.range_lookback)
        self.range_low = self.I(talib.MIN, self.data.Low, timeperiod=self.range_lookback)
        print("🌙✨ NocturnalEngulf initialized — indicators loaded 🚀")

    def _is_hanging(self, o, h, l, c):
        total_range = h - l
        if total_range <= 0:
            return False, 0.0, 0.0
        body = abs(c - o)
        lower_wick = min(o, c) - l
        body_ratio = body / total_range
        if body_ratio > self.body_ratio_max:
            return False, 0.0, 0.0
        body_floor = max(body, total_range * 0.01)
        if lower_wick < self.lower_wick_mult * body_floor:
            return False, 0.0, 0.0
        if c < (l + total_range * 0.5):
            return False, 0.0, 0.0
        return True, total_range, lower_wick

    def _is_overnight(self, ts):
        try:
            hour = ts.hour
        except AttributeError:
            hour = pd.Timestamp(ts).hour
        return (hour >= 22) or (hour < 6)

    def next(self):
        i = len(self.data) - 1
        if i < max(self.range_lookback, self.adx_period, self.atr_period) + 5:
            return

        if self.position:
            self._manage_position(i)
            return

        o1, h1, l1, c1 = self.data.Open[-3], self.data.High[-3], self.data.Low[-3], self.data.Close[-3]
        o2, h2, l2, c2 = self.data.Open[-2], self.data.High[-2], self.data.Low[-2], self.data.Close[-2]

        ts2 = self.data.index[-2]

        if not self._is_overnight(ts2):
            return

        ok1, rng1, lw1 = self._is_hanging(o1, h1, l1, c1)
        ok2, rng2, lw2 = self._is_hanging(o2, h2, l2, c2)
        if not (ok1 and ok2):
            return

        if rng2 < self.size_expansion * rng1:
            return

        adx_now = self.adx[-2]
        if np.isnan(adx_now) or adx_now >= self.adx_threshold_entry:
            return

        rh = self.range_high[-2]
        rl = self.range_low[-2]
        atr_now = self.atr[-2]
        if np.isnan(atr_now) or atr_now <= 0:
            return
        range_width = rh - rl
        if range_width > self.range_width_atr_mult * atr_now:
            return

        if h2 <= h1:
            return

        entry_price = self.data.Open[-1]
        stop_price = min(l2, entry_price - self.atr_stop_mult * atr_now)
        target_atr = entry_price + self.atr_target_mult * atr_now
        target_range = rh
        target_price = min(target_atr, target_range) if target_range > entry_price else target_atr

        risk = entry_price - stop_price
        reward = target_price - entry_price
        if risk <= 0 or reward <= 0:
            return
        if reward / risk < self.rr_target:
            print(f"🌙 Skipping — R:R {reward/risk:.2f} < {self.rr_target}")
            return

        # 🌙 Moon Dev fix: use fixed fractional risk-based sizing (fraction of equity)
        # risk_pct is fraction of equity to risk; convert to position size fraction
        # position size fraction = risk_pct / (risk / entry_price)
        size_fraction = self.risk_pct / (risk / entry_price)
        if size_fraction <= 0:
            return
        if size_fraction >= 1.0:
            size_fraction = 0.99

        print(f"🌙🚀 LONG signal @ {entry_price:.2f} | SL {stop_price:.2f} | TP {target_price:.2f} | size_frac {size_fraction:.4f} | ADX {adx_now:.1f}")

        self.buy(size=size_fraction, sl=stop_price, tp=target_price)

    def _manage_position(self, i):
        adx_now = self.adx[-1]
        if not np.isnan(adx_now) and adx_now > self.adx_threshold_exit:
            print(f"🌙⚠️ Regime exit — ADX {adx_now:.1f} > {self.adx_threshold_exit}")
            self.position.close()


# ============================================================
# 🌙 RUN BACKTEST
# ============================================================
if __name__ == "__main__":
    data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
    print("🌙 Loading data...")
    df = load_data(data_path)
    print(f"🌙 Data loaded: {len(df)} rows")
    print(f"🌙 Columns: {list(df.columns)}")
    print(f"🌙 Index range: {df.index[0]} → {df.index[-1]}")

    bt = Backtest(df, NocturnalEngulf, cash=1_000_000, commission=0.0002)
    stats = bt.run()
    print(stats)
    print(stats._strategy)