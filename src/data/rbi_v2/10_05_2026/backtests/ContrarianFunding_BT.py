import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

print("🌙✨ ContrarianFunding Strategy Loading... Let's fade the crowd! ✨🌙")

# Load and clean data
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper column mapping
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume',
    'datetime': 'Datetime'
}, inplace=True)

data['Datetime'] = pd.to_datetime(data['Datetime'])
data.set_index('Datetime', inplace=True)
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

print(f"🚀 Data loaded: {len(data)} bars from {data.index[0]} to {data.index[-1]}")

# Synthesize a funding rate proxy since raw funding data is unavailable in this CSV.
# We use a rolling z-score of returns as a proxy for crowded positioning.
# Extreme positive = crowded longs, extreme negative = crowded shorts.
# In production, this would be replaced by actual 8h funding rate data.

class ContrarianFunding(Strategy):
    # Strategy parameters
    adx_period = 14
    adx_entry_threshold = 25      # enter only when ADX < 25 (ranging regime)
    adx_exit_threshold = 35       # exit when ADX > 35 (trend regime)
    funding_lookback = 90         # ~30 days at 8h cadence
    funding_extreme_pct = 10      # 10th/90th percentile thresholds
    atr_period = 14
    atr_stop_mult = 1.75          # stop at 1.75x ATR
    atr_tp_mult = 3.0             # secondary TP at 3x ATR
    time_stop_bars = 36           # ~36 bars time stop
    risk_pct = 0.01               # 1% equity risk per trade
    cooldown_bars = 10            # cooldown after ADX>35 exit

    def init(self):
        print("🌙 Initializing ContrarianFunding indicators...")

        # ADX for regime filter
        self.adx = self.I(talib.ADX, self.data.High, self.data.Low, self.data.Close,
                          timeperiod=self.adx_period, name='ADX')

        # ATR for stop sizing
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close,
                          timeperiod=self.atr_period, name='ATR')

        # Funding rate proxy: rolling z-score of returns
        # (crowded longs -> positive, crowded shorts -> negative)
        close = pd.Series(self.data.Close)
        ret = close.pct_change().fillna(0)
        roll_mean = ret.rolling(self.funding_lookback, min_periods=20).mean()
        roll_std = ret.rolling(self.funding_lookback, min_periods=20).std()
        funding_proxy = (ret - roll_mean) / roll_std.replace(0, np.nan)
        funding_proxy = funding_proxy.fillna(0)

        self.funding = self.I(lambda: funding_proxy.values, name='FundingProxy')

        # Rolling percentile thresholds for funding extremes
        def lower_thresh():
            return funding_proxy.rolling(self.funding_lookback, min_periods=20).quantile(
                self.funding_extreme_pct / 100).fillna(0).values

        def upper_thresh():
            return funding_proxy.rolling(self.funding_lookback, min_periods=20).quantile(
                1 - self.funding_extreme_pct / 100).fillna(0).values

        def median_funding():
            return funding_proxy.rolling(self.funding_lookback, min_periods=20).median().fillna(0).values

        self.funding_low = self.I(lower_thresh, name='FundingLow')
        self.funding_high = self.I(upper_thresh, name='FundingHigh')
        self.funding_median = self.I(median_funding, name='FundingMedian')

        self.entry_bar = None
        self.cooldown_until = 0
        print("✨ Indicators ready! Ready to fade the crowd 🌙")

    def next(self):
        price = self.data.Close[-1]
        adx_val = self.adx[-1]
        atr_val = self.atr[-1]

        if np.isnan(adx_val) or np.isnan(atr_val) or atr_val <= 0:
            return

        # ---- Manage open position ----
        if self.position:
            bars_held = len(self.data) - 1 - self.entry_bar

            # Exit 1: ADX crosses above 35 (trend regime invalidation)
            if adx_val > self.adx_exit_threshold:
                print(f"🚨 ADX={adx_val:.1f} > 35 — trend formed! Exiting (regime invalidation) 🌙")
                self.position.close()
                self.cooldown_until = len(self.data) + self.cooldown_bars
                return

            # Exit 2: funding normalized back to median (thesis complete)
            f = self.funding[-1]
            fmed = self.funding_median[-1]
            if self.position.is_long and f >= fmed:
                print(f"✅ Funding normalized ({f:.2f} >= median {fmed:.2f}) — long thesis complete 🎯")
                self.position.close()
                return
            if self.position.is_short and f <= fmed:
                print(f"✅ Funding normalized ({f:.2f} <= median {fmed:.2f}) — short thesis complete 🎯")
                self.position.close()
                return

            # Exit 3: time stop
            if bars_held >= self.time_stop_bars:
                print(f"⏰ Time stop hit ({bars_held} bars) — closing dead capital trade 🌙")
                self.position.close()
                return

            return

        # ---- Cooldown check ----
        if len(self.data) < self.cooldown_until:
            return

        # ---- Entry logic ----
        f = self.funding[-1]
        flow = self.funding_low[-1]
        fhigh = self.funding_high[-1]

        # Stop distance & position size (1,000,000 size per spec)
        stop_dist = self.atr_stop_mult * atr_val
        if stop_dist <= 0:
            return

        # Long entry: fade crowded shorts
        if f < flow and adx_val < self.adx_entry_threshold:
            sl = price - stop_dist
            tp = price + self.atr_tp_mult * atr_val
            print(f"🌙🚀 LONG ENTRY (fade crowded shorts) | price={price:.2f} "
                  f"funding={f:.2f} < low={flow:.2f} | ADX={adx_val:.1f} | SL={sl:.2f} TP={tp:.2f}")
            self.buy(size=1000000, sl=sl, tp=tp)
            self.entry_bar = len(self.data) - 1

        # Short entry: fade crowded longs
        elif f > fhigh and adx_val < self.adx_entry_threshold:
            sl = price + stop_dist
            tp = price - self.atr_tp_mult * atr_val
            print(f"🌙🚀 SHORT ENTRY (fade crowded longs) | price={price:.2f} "
                  f"funding={f:.2f} > high={fhigh:.2f} | ADX={adx_val:.1f} | SL={sl:.2f} TP={tp:.2f}")
            self.sell(size=1000000, sl=sl, tp=tp)
            self.entry_bar = len(self.data) - 1


print("🌙✨ Running ContrarianFunding Backtest... ✨🌙")
bt = Backtest(data, ContrarianFunding, cash=1_000_000, commission=0.0002, exclusive_orders=True)
stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙✨ Backtest complete! Moon Dev out 🚀")