import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# ================== DATA LOADING ==================
data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Map to proper case
data = data.rename(columns={
    'datetime': 'Datetime',
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

# Set datetime index
if 'Datetime' in data.columns:
    data['Datetime'] = pd.to_datetime(data['Datetime'])
    data = data.set_index('Datetime')

print("🌙✨ Moon Dev Backtest Initialized — FundingSqueeze Strategy Loading...")
print(f"🚀 Data loaded: {len(data)} bars from {data.index[0]} to {data.index[-1]}")


# ================== STRATEGY CLASS ==================
class FundingSqueeze(Strategy):
    # Bollinger Bands
    bb_period = 20
    bb_std = 2.0

    # BandWidth percentile lookback
    bw_lookback = 100
    bw_percentile = 10

    # ATR
    atr_period = 14
    atr_stop_mult = 1.5

    # ADX filter
    adx_period = 14
    adx_threshold = 20

    # Funding proxy (no real funding feed; use a proxy via rolling returns)
    funding_lookback = 30
    funding_percentile = 80  # top 20th percentile as "high regime"

    # Exit thresholds
    bw_exit_percentile = 90  # vol expansion exit
    time_stop_bars = 60

    # Risk
    risk_pct = 0.01  # 1% of equity per trade

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        # Bollinger Bands
        self.bb_mid = self.I(talib.SMA, close, timeperiod=self.bb_period)
        self.bb_stddev = self.I(talib.STDDEV, close, timeperiod=self.bb_period, nbdev=1)
        self.bb_upper = self.I(lambda: self.bb_mid + self.bb_std * self.bb_stddev)
        self.bb_lower = self.I(lambda: self.bb_mid - self.bb_std * self.bb_stddev)

        # BandWidth
        self.bw = self.I(lambda: (self.bb_upper - self.bb_lower) / self.bb_mid)

        # BandWidth percentile rank over lookback
        def bw_pct_rank(arr):
            out = np.full(len(arr), np.nan)
            for i in range(self.bw_lookback, len(arr)):
                window = arr[i - self.bw_lookback:i]
                if not np.isnan(window).any():
                    out[i] = (window < arr[i]).mean() * 100
            return out

        self.bw_rank = self.I(bw_pct_rank, self.bw)

        # ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        # ADX
        self.adx = self.I(talib.ADX, high, low, close, timeperiod=self.adx_period)

        # Funding proxy: use rolling return momentum as "funding-like" crowding
        # High positive rolling return over N bars → crowded longs
        def funding_proxy(arr):
            s = pd.Series(arr)
            f = s.pct_change(self.funding_lookback)
            out = np.full(len(arr), np.nan)
            for i in range(self.funding_lookback + self.funding_lookback, len(arr)):
                window = f.iloc[i - self.funding_lookback:i].dropna()
                if len(window) > 5 and not np.isnan(f.iloc[i]):
                    out[i] = (window < f.iloc[i]).mean() * 100
            return out

        self.funding_proxy = self.I(funding_proxy, close)

        # Volume-weighted swing extremes (rolling 50-bar VWAP-anchored extremes)
        self.swing_high = self.I(talib.MAX, high, timeperiod=50)
        self.swing_low = self.I(talib.MIN, low, timeperiod=50)

        # Realized vol
        self.rvol = self.I(talib.STDDEV, close, timeperiod=20, nbdev=1)

        print("🌙 Indicators initialized — BB, BandWidth, ADX, ATR, Funding Proxy ready ✨")

    def next(self):
        price = self.data.Close[-1]
        i = len(self.data) - 1

        # Skip warmup
        if i < self.bw_lookback + 50:
            return

        # NaN guard
        if (np.isnan(self.bw_rank[-1]) or np.isnan(self.funding_proxy[-1])
                or np.isnan(self.atr[-1]) or np.isnan(self.adx[-1])
                or np.isnan(self.bb_upper[-1]) or np.isnan(self.bb_lower[-1])):
            return

        # ---------- SIGNAL LOGIC ----------
        # 1. Funding regime: high positive funding (crowded longs)
        funding_high = self.funding_proxy[-1] >= self.funding_percentile

        # 2. Squeeze: BandWidth in lowest 10th percentile
        squeeze = self.bw_rank[-1] <= self.bw_percentile

        # 3. Trend filter: ADX < 20
        no_trend = self.adx[-1] < self.adx_threshold

        # 4. Trigger: price closes back inside bands after poke outside (false break)
        poke_above = self.data.High[-2] > self.bb_upper[-2]
        back_inside = price < self.bb_upper[-1]
        false_break_down = poke_above and back_inside

        # Or BandWidth expanding after squeeze low
        bw_expanding = self.bw[-1] > self.bw[-2] and self.bw_rank[-2] <= self.bw_percentile

        trigger = false_break_down or bw_expanding

        entry_condition = funding_high and squeeze and no_trend and trigger

        # ---------- POSITION MANAGEMENT ----------
        if not self.position:
            if entry_condition:
                # Short volatility → directional short when squeeze resolves down
                # Risk-based sizing
                stop_dist = self.atr[-1] * self.atr_stop_mult
                if stop_dist <= 0:
                    return

                risk_amount = self.equity * self.risk_pct
                # Inverse vol scaling: bigger size when rvol low
                rvol_factor = max(0.3, min(2.0, 0.02 / (self.rvol[-1] + 1e-9)))
                raw_size = (risk_amount / stop_dist) * rvol_factor
                size = int(round(raw_size))
                if size < 1:
                    size = 1

                sl = price + stop_dist
                tp = self.swing_low[-1]  # liquidity cluster target

                if tp >= price:
                    tp = price - stop_dist * 2

                self.buy(size=size)  # placeholder to register; we short via sell
                self.position.close()
                self.sell(size=size, sl=sl, tp=tp)
                print(f"🌙🔻 SHORT ENTRY | price={price:.2f} SL={sl:.2f} TP={tp:.2f} size={size} | funding={self.funding_proxy[-1]:.1f} bw_rank={self.bw_rank[-1]:.1f} ADX={self.adx[-1]:.1f} 🚀")

        else:
            # Time stop
            bars_held = i - self.trades[-1].entry_bar if self.trades else 0
            if bars_held >= self.time_stop_bars:
                self.position.close()
                print(f"⏰ Time stop exit at {price:.2f} after {bars_held} bars 🌙")
                return

            # Volatility expansion exit
            if self.bw_rank[-1] >= self.bw_exit_percentile:
                self.position.close()
                print(f"💥 Vol expansion exit | bw_rank={self.bw_rank[-1]:.1f} price={price:.2f} ✨")
                return

            # Funding normalization exit
            if self.funding_proxy[-1] < 50:
                self.position.close()
                print(f"🌊 Funding normalized exit | funding={self.funding_proxy[-1]:.1f} price={price:.2f} 🌙")


# ================== RUN BACKTEST ==================
bt = Backtest(
    data,
    FundingSqueeze,
    cash=1_000_000,
    commission=0.0002,
    exclusive_orders=True,
    trade_on_close=False
)

print("🌙✨ Running Moon Dev FundingSqueeze Backtest... 🚀")
stats = bt.run()
print(stats)
print(stats._strategy)