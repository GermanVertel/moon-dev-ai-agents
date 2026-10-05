import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

print("🌙✨ Moon Dev's FiboBand Tracker initializing... 🚀")

DATA_PATH = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"

data = pd.read_csv(DATA_PATH)
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

print(f"🌙 Data loaded: {len(data)} bars 📊")


class FiboBandTracker(Strategy):
    bb_period = 20
    bb_std = 2.0
    sma_trend = 200
    swing_lookback = 30
    risk_pct = 0.01
    max_bars_hold = 15

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low

        self.bb_upper, self.bb_mid, self.bb_lower = self.I(
            talib.BBANDS, close, timeperiod=self.bb_period,
            nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0
        )
        self.sma200 = self.I(talib.SMA, close, timeperiod=self.sma_trend)
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=14)
        self.swing_high = self.I(talib.MAX, high, timeperiod=self.swing_lookback)
        self.swing_low = self.I(talib.MIN, low, timeperiod=self.swing_lookback)

        self.entry_bar = 0
        print("🌙✨ Indicators ready: BB, SMA200, ATR, Swing H/L 🚀")

    def _bb_width(self, i):
        if self.bb_mid[i] and self.bb_mid[i] > 0:
            return (self.bb_upper[i] - self.bb_lower[i]) / self.bb_mid[i]
        return 0

    def _bullish_reversal(self, i):
        o, c = self.data.Open[i], self.data.Close[i]
        o1, c1 = self.data.Open[i-1], self.data.Close[i-1]
        body = abs(c - o)
        rng = self.data.High[i] - self.data.Low[i]
        if rng <= 0:
            return False
        lower_wick = min(o, c) - self.data.Low[i]
        hammer = (lower_wick > 2 * body) and (body / rng < 0.4)
        engulf = (c > o) and (c1 < o1) and (c >= o1) and (o <= c1)
        return hammer or engulf

    def _bearish_reversal(self, i):
        o, c = self.data.Open[i], self.data.Close[i]
        o1, c1 = self.data.Open[i-1], self.data.Close[i-1]
        body = abs(c - o)
        rng = self.data.High[i] - self.data.Low[i]
        if rng <= 0:
            return False
        upper_wick = self.data.High[i] - max(o, c)
        shooting = (upper_wick > 2 * body) and (body / rng < 0.4)
        engulf = (c < o) and (c1 > o1) and (c <= o1) and (o >= c1)
        return shooting or engulf

    def next(self):
        i = len(self.data) - 1
        if i < self.sma_trend + 5 or i < self.swing_lookback + 5:
            return

        price = self.data.Close[i]
        sma200 = self.sma200[i]
        bb_u = self.bb_upper[i]
        bb_m = self.bb_mid[i]
        bb_l = self.bb_lower[i]
        sh = self.swing_high[i]
        sl = self.swing_low[i]

        if any(v is None or np.isnan(v) for v in [sma200, bb_u, bb_m, bb_l, sh, sl]):
            return

        if self.position:
            self.entry_bar += 1
            bars_held = self.entry_bar
            if self.position.is_long:
                if price >= bb_u or price < bb_m:
                    print(f"🌙 Long exit @ {price:.2f} (BB target/trail) ✨")
                    self.position.close()
                elif bars_held >= self.max_bars_hold:
                    print(f"🌙 Long time exit @ {price:.2f} ⏱️")
                    self.position.close()
            elif self.position.is_short:
                if price <= bb_l or price > bb_m:
                    print(f"🌙 Short exit @ {price:.2f} (BB target/trail) ✨")
                    self.position.close()
                elif bars_held >= self.max_bars_hold:
                    print(f"🌙 Short time exit @ {price:.2f} ⏱️")
                    self.position.close()
            return

        sma_slope = self.sma200[i] - self.sma200[i-3]
        mid_slope = bb_m - self.bb_mid[i-3]

        trend_up = price > sma200 and mid_slope > 0
        trend_down = price < sma200 and mid_slope < 0

        rng_sw = sh - sl
        if rng_sw <= 0:
            return

        fib_382 = sh - 0.382 * rng_sw
        fib_500 = sh - 0.500 * rng_sw
        fib_618 = sh - 0.618 * rng_sw

        fib_382_dn = sl + 0.382 * rng_sw
        fib_500_dn = sl + 0.500 * rng_sw
        fib_618_dn = sl + 0.618 * rng_sw

        near_fib_long = (
            abs(price - fib_382) / price < 0.005 or
            abs(price - fib_500) / price < 0.005 or
            abs(price - fib_618) / price < 0.005
        )
        near_fib_short = (
            abs(price - fib_382_dn) / price < 0.005 or
            abs(price - fib_500_dn) / price < 0.005 or
            abs(price - fib_618_dn) / price < 0.005
        )

        touch_bb_long = self.data.Low[i] <= bb_l * 1.002 or abs(price - bb_m) / price < 0.004
        touch_bb_short = self.data.High[i] >= bb_u * 0.998 or abs(price - bb_m) / price < 0.004

        atr_val = self.atr[i]
        if atr_val is None or np.isnan(atr_val) or atr_val <= 0:
            return

        if trend_up and near_fib_long and touch_bb_long and self._bullish_reversal(i):
            stop = min(self.data.Low[i], sl) - 0.5 * atr_val
            risk = price - stop
            if risk <= 0:
                return
            size = int(round((self.equity * self.risk_pct) / risk))
            if size <= 0:
                return
            print(f"🚀🌙 LONG entry @ {price:.2f} | Fib zone | stop {stop:.2f} | size {size} ✨")
            self.buy(size=size)
            self.entry_bar = 0

        elif trend_down and near_fib_short and touch_bb_short and self._bearish_reversal(i):
            stop = max(self.data.High[i], sh) + 0.5 * atr_val
            risk = stop - price
            if risk <= 0:
                return
            size = int(round((self.equity * self.risk_pct) / risk))
            if size <= 0:
                return
            print(f"🔻🌙 SHORT entry @ {price:.2f} | Fib zone | stop {stop:.2f} | size {size} ✨")
            self.sell(size=size)
            self.entry_bar = 0


bt = Backtest(data, FiboBandTracker, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)