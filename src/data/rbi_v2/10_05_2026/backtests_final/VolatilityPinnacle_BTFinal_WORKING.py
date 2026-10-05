import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# ─── Moon Dev Data Loader 🌙 ────────────────────────────────────────────────
print("🌙 Moon Dev: Loading BTC-USD 15m data...")
data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Map to backtesting.py required columns
data = data.rename(columns={
    'datetime': 'Date',
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume',
})

# Ensure datetime index
if 'Date' in data.columns:
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.set_index('Date')

data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
print(f"🌙 Moon Dev: Data loaded — {len(data)} bars ✨")


# ─── VIX Proxy Loader (synthetic — no VIX in dataset) ──────────────────────
def build_vix_proxy(df):
    """
    Since the dataset has no VIX, we build a synthetic fear proxy.
    Realized volatility (rolling std of returns) scaled, used as VIX stand-in.
    """
    ret = df['Close'].pct_change()
    rv = ret.rolling(20).std() * np.sqrt(96 * 365) * 100  # annualized-ish
    return rv.bfill().ffill()


# ─── Strategy ──────────────────────────────────────────────────────────────
class VolatilityPinnacle(Strategy):
    vix_lookback = 252
    vix_pct_thresh = 0.90
    bb_period = 20
    bb_std = 2.0
    atr_period = 14
    atr_ma_period = 50
    rsi_period = 14
    bbw_lookback = 20
    time_stop = 10
    risk_pct = 0.01          # 1% equity risk per trade
    atr_stop_mult = 1.5

    def init(self):
        print("🌙 Moon Dev: Initializing VolatilityPinnacle indicators... ✨")

        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)

        # Bollinger Bands
        self.bb_upper = self.I(
            lambda c: talib.BBANDS(c, self.bb_period, self.bb_std, self.bb_std)[0],
            close
        )
        self.bb_middle = self.I(
            lambda c: talib.BBANDS(c, self.bb_period, self.bb_std, self.bb_std)[1],
            close
        )
        self.bb_lower = self.I(
            lambda c: talib.BBANDS(c, self.bb_period, self.bb_std, self.bb_std)[2],
            close
        )

        # Bollinger Band Width
        self.bbw = self.I(
            lambda u, l, m: (u - l) / m,
            self.bb_upper, self.bb_lower, self.bb_middle
        )

        # ATR
        self.atr = self.I(talib.ATR, high, low, close, self.atr_period)
        self.atr_ma = self.I(talib.SMA, self.atr, self.atr_ma_period)

        # RSI
        self.rsi = self.I(talib.RSI, close, self.rsi_period)

        # VIX proxy
        vix_series = build_vix_proxy(pd.DataFrame({'Close': close}))
        self.vix = self.I(lambda x: x.values, vix_series)

        # VIX rolling percentile
        def vix_pct(x):
            s = pd.Series(x)
            return s.rolling(self.vix_lookback).apply(
                lambda w: (w.iloc[-1] >= np.nanpercentile(w, self.vix_pct_thresh * 100)).astype(float),
                raw=False
            )
        self.vix_pct = self.I(lambda x: vix_pct(x).values, self.vix)

        # VIX 3-bar pivot high (lower high on last bar)
        def vix_peak(x):
            s = pd.Series(x)
            prev = s.shift(1)
            prev2 = s.shift(2)
            # local peak: bar-2 was a peak and bar-1 < bar-2
            peak = (prev2 > prev) & (prev2 > s)
            return peak.astype(float).values
        self.vix_peak = self.I(lambda x: vix_peak(x), self.vix)

        # BBW rolling low (bottom quartile)
        def bbw_low(x):
            s = pd.Series(x)
            return s.rolling(self.bbw_lookback).apply(
                lambda w: (w.iloc[-1] <= np.nanpercentile(w, 25)).astype(float),
                raw=False
            )
        self.bbw_low = self.I(lambda x: bbw_low(x).values, self.bbw)

        print("🌙 Moon Dev: Indicators ready 🚀")

    def next(self):
        price = self.data.Close[-1]

        # Skip if not enough data
        if len(self.data) < max(self.vix_lookback, self.bbw_lookback, self.atr_ma_period) + 5:
            return

        # ─── Entry Logic ───────────────────────────────────────────────────
        if not self.position:
            vix_high = self.vix_pct[-1] >= 1.0
            vix_peak_ok = self.vix_peak[-1] == 1.0 or self.vix_peak[-2] == 1.0
            contraction = self.bbw_low[-1] == 1.0
            atr_below = self.atr[-1] < self.atr_ma[-1]
            rsi_confirm = self.rsi[-1] < 50

            if vix_high and vix_peak_ok and (contraction or atr_below) and rsi_confirm:
                # Risk-based position sizing
                risk_per_unit = self.atr[-1] * self.atr_stop_mult
                if risk_per_unit <= 0 or np.isnan(risk_per_unit):
                    return
                risk_amount = self.equity * self.risk_pct
                size = int(round(risk_amount / risk_per_unit))
                if size < 1:
                    size = 1

                stop_price = price + risk_per_unit
                target_price = self.bb_lower[-1]

                print(f"🌙 SHORT SIGNAL 🚀 | Price={price:.2f} | VIX={self.vix[-1]:.2f} | "
                      f"BBW={self.bbw[-1]:.4f} | RSI={self.rsi[-1]:.1f} | Size={size}")

                self.sell(size=size, sl=stop_price, tp=target_price)
                self.entry_bar = len(self.data)

        # ─── Exit Logic ────────────────────────────────────────────────────
        else:
            # Primary: lower BB touched (TP handles it)
            # Secondary: middle band touched
            if self.data.Low[-1] <= self.bb_middle[-1]:
                print(f"🌙 MID-BAND EXIT ✨ | Price={price:.2f}")
                self.position.close()
                return

            # Time stop
            if hasattr(self, 'entry_bar') and (len(self.data) - self.entry_bar) >= self.time_stop:
                print(f"🌙 TIME STOP ⏰ | Price={price:.2f}")
                self.position.close()


# ─── Run Backtest ──────────────────────────────────────────────────────────
print("🌙 Moon Dev: Launching VolatilityPinnacle backtest... 🚀✨")
bt = Backtest(
    data,
    VolatilityPinnacle,
    cash=1_000_000,
    commission=0.002,
    exclusive_orders=True,
)

stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙 Moon Dev: Backtest complete! ✨🚀")