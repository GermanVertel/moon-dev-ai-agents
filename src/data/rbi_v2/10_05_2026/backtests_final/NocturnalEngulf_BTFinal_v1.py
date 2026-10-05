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
    # Map columns properly
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
    # Strategy parameters
    adx_period = 14
    adx_threshold_entry = 20
    adx_threshold_exit = 25
    atr_period = 14
    range_lookback = 20
    body_ratio_max = 0.30          # body <= 30% of total range
    lower_wick_mult = 2.0          # lower wick >= 2x body
    size_expansion = 1.10          # candle 2 range >= 110% candle 1 range
    risk_pct = 0.01                # 1% risk per trade
    rr_target = 1.5                # minimum reward:risk
    atr_target_mult = 1.5          # profit target = 1.5x ATR
    atr_stop_mult = 1.0            # stop = 1x ATR
    range_width_atr_mult = 1.5     # range width <= 1.5x ATR

    def init(self):
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period)
        self.adx = self.I(talib.ADX, self.data.High, self.data.Low, self.data.Close, timeperiod=self.adx_period)
        self.range_high = self.I(talib.MAX, self.data.High, timeperiod=self.range_lookback)
        self.range_low = self.I(talib.MIN, self.data.Low, timeperiod=self.range_lookback)
        print("🌙✨ NocturnalEngulf initialized — indicators loaded 🚀")

    # ---------------- helpers ----------------
    def _is_hanging(self, o, h, l, c):
        total_range = h - l
        if total_range <= 0:
            return False, 0.0, 0.0
        body = abs(c - o)
        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l
        body_ratio = body / total_range
        # small body
        if body_ratio > self.body_ratio_max:
            return False, 0.0, 0.0
        # lower wick >= 2x body (with body floor to avoid div-by-zero)
        body_floor = max(body, total_range * 0.01)
        if lower_wick < self.lower_wick_mult * body_floor:
            return False, 0.0, 0.0
        # close in upper half
        if c < (l + total_range * 0.5):
            return False, 0.0, 0.0
        return True, total_range, lower_wick

    def _is_overnight(self, ts):
        # Overnight / off-hours: 22:00 - 06:00 UTC
        try:
            hour = ts.hour
        except AttributeError:
            hour = pd.Timestamp(ts).hour
        return (hour >= 22) or (hour < 6)

    # ---------------- main loop ----------------
    def next(self):
        i = len(self.data) - 1
        if i < max(self.range_lookback, self.adx_period, self.atr_period) + 5:
            return

        # Manage open trade
        if self.position:
            self._manage_position(i)
            return

        # Need previous two candles
        o1, h1, l1, c1 = self.data.Open[-3], self.data.High[-3], self.data.Low[-3], self.data.Close[-3]
        o2, h2, l2, c2 = self.data.Open[-2], self.data.High[-2], self.data.Low[-2], self.data.Close[-2]

        ts2 = self.data.index[-2]

        # 1) Session filter — signal candle must be overnight
        if not self._is_overnight(ts2):
            return

        # 2) Both candles must be hanging candles
        ok1, rng1, lw1 = self._is_hanging(o1, h1, l1, c1)
        ok2, rng2, lw2 = self._is_hanging(o2, h2, l2, c2)
        if not (ok1 and ok2):
            return

        # 3) Size expansion: candle 2 >= 110% of candle 1
        if rng2 < self.size_expansion * rng1:
            return

        # 4) Consolidation regime: ADX < 20
        adx_now = self.adx[-2]
        if np.isnan(adx_now) or adx_now >= self.adx_threshold_entry:
            return

        # 5) Range width <= 1.5x ATR
        rh = self.range_high[-2]
        rl = self.range_low[-2]
        atr_now = self.atr[-2]
        if np.isnan(atr_now) or atr_now <= 0:
            return
        range_width = rh - rl
        if range_width > self.range_width_atr_mult * atr_now:
            return

        # 6) Optional confirmation: candle 2 high > candle 1 high
        if h2 <= h1:
            return

        # Entry
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

        # Position sizing: fraction of equity (0 < size < 1) for percentage-based sizing
        # Compute fraction of equity to allocate so that risk matches risk_pct
        # size_fraction = risk_amount / entry_price  (as fraction of equity)
        # Because backtesting.py interprets 0<size<1 as fraction of equity used for the order.
        # We want the loss at stop to equal risk_amount = equity * risk_pct.
        # Loss per unit = entry_price - stop_price = risk.
        # Units = risk_amount / risk.
        # Notional = units * entry_price.
        # Fraction of equity = Notional / equity = (risk_amount / risk) * entry_price / equity
        #                    = (equity * risk_pct / risk) * entry_price / equity
        #                    = risk_pct * entry_price / risk.
        size_fraction = (self.risk_pct * entry_price) / risk
        # Clamp to valid range
        if size_fraction <= 0:
            return
        if size_fraction > 1.0:
            size_fraction = 1.0
        # Ensure it's a valid fraction (not exactly 1.0 which means 100% — still valid but risky)
        if size_fraction >= 1.0:
            size_fraction = 0.99

        print(f"🌙🚀 LONG signal @ {entry_price:.2f} | SL {stop_price:.2f} | TP {target_price:.2f} | size_frac {size_fraction:.4f} | ADX {adx_now:.1f}")

        self.buy(size=size_fraction, sl=stop_price, tp=target_price)

    def _manage_position(self, i):
        # Regime exit: ADX > 25 against position (long)
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

    bt = Backtest(df, NocturnalEngulf, cash=1_000_000, commission=0.0002)
    stats = bt.run()
    print(stats)
    print(stats._strategy)