import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

print("🌙✨ Starting Moon Dev's FibonacciStochastic Backtest ✨🌙")

# Load and clean data
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
print(f"🚀 Data loaded: {len(data)} bars from {data.index[0]} to {data.index[-1]}")


class FibonacciStochastic(Strategy):
    # Trend filter
    ema_period = 200
    # Stochastic
    stoch_k = 14
    stoch_d = 3
    stoch_smooth = 3
    # Swing lookback
    swing_lookback = 50
    # Risk
    risk_pct = 0.01
    rr_target = 2.0

    def init(self):
        print("🌙 Initializing FibonacciStochastic indicators...")
        self.ema200 = self.I(talib.EMA, self.data.Close, timeperiod=self.ema_period)
        self.stoch_k, self.stoch_d = self.I(
            talib.STOCH, self.data.High, self.data.Low, self.data.Close,
            fastk_period=self.stoch_k,
            slowk_period=self.stoch_smooth,
            slowk_matype=0,
            slowd_period=self.stoch_d,
            slowd_matype=0
        )
        # Swing high/low using talib MAX/MIN
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=self.swing_lookback)
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=self.swing_lookback)
        print("✨ Indicators ready: EMA200, Stochastic, Swing High/Low")

    def next(self):
        if len(self.data) < self.swing_lookback + 5:
            return

        price = self.data.Close[-1]
        ema = self.ema200[-1]
        k = self.stoch_k[-1]
        d = self.stoch_d[-1]
        k_prev = self.stoch_k[-2]
        d_prev = self.stoch_d[-2]

        sh = self.swing_high[-1]
        sl = self.swing_low[-1]

        if np.isnan(sh) or np.isnan(sl) or np.isnan(ema) or np.isnan(k) or np.isnan(d):
            return

        swing_range = sh - sl
        if swing_range <= 0:
            return

        # Fibonacci levels (retracement measured from swing high down)
        fib_382 = sh - 0.382 * swing_range
        fib_500 = sh - 0.500 * swing_range
        fib_618 = sh - 0.618 * swing_range
        fib_786 = sh - 0.786 * swing_range

        # Trend filters
        uptrend = price > ema
        downtrend = price < ema

        # Fib zone: price between 38.2% and 78.6% retracement
        in_fib_zone_long = (price <= fib_382) and (price >= fib_786)
        in_fib_zone_short = (price >= fib_382) and (price <= fib_786)

        # Stochastic crossover
        bull_cross = (k_prev < d_prev) and (k > d) and (k_prev < 20 or k < 30)
        bear_cross = (k_prev > d_prev) and (k < d) and (k_prev > 80 or k > 70)

        # ----- LONG ENTRY -----
        if not self.position and uptrend and in_fib_zone_long and bull_cross:
            entry = price
            stop = min(sl, fib_786) * 0.999
            risk = entry - stop
            if risk > 0:
                tp = entry + self.rr_target * risk
                size = int(round((self.equity * self.risk_pct) / risk))
                if size > 0:
                    print(f"🌙🚀 LONG SIGNAL | Entry={entry:.2f} SL={stop:.2f} TP={tp:.2f} Size={size} | K={k:.1f} D={d:.1f}")
                    self.buy(size=size, sl=stop, tp=tp)

        # ----- SHORT ENTRY -----
        elif not self.position and downtrend and in_fib_zone_short and bear_cross:
            entry = price
            stop = max(sh, fib_786) * 1.001
            risk = stop - entry
            if risk > 0:
                tp = entry - self.rr_target * risk
                size = int(round((self.equity * self.risk_pct) / risk))
                if size > 0:
                    print(f"🌙🔻 SHORT SIGNAL | Entry={entry:.2f} SL={stop:.2f} TP={tp:.2f} Size={size} | K={k:.1f} D={d:.1f}")
                    self.sell(size=size, sl=stop, tp=tp)

        # Trend failure exit
        if self.position.is_long and price < ema:
            print(f"🌙⚠️ Trend failure exit LONG at {price:.2f}")
            self.position.close()
        elif self.position.is_short and price > ema:
            print(f"🌙⚠️ Trend failure exit SHORT at {price:.2f}")
            self.position.close()


bt = Backtest(data, FibonacciStochastic, cash=1_000_000, commission=0.001, exclusive_orders=True)
stats = bt.run()
print(stats)
print(stats._strategy)