import pandas as pd
import numpy as np
import talib
from backtesting import Backtest, Strategy

# ============================================================
# 🌙 Moon Dev's KineticDivergence Backtest 🌙
# ============================================================

DATA_PATH = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"

print("🌙 Moon Dev: Loading data from", DATA_PATH)
data = pd.read_csv(DATA_PATH)

# Clean columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Map to proper case
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

# Set datetime index
if 'datetime' in data.columns:
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')

data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
print("🌙 Moon Dev: Data loaded! Shape:", data.shape, "✨")


class KineticDivergence(Strategy):
    """
    KineticDivergence: Keltner Channel + RSI Divergence hybrid strategy.
    🌙 Dual confirmation: channel break + momentum divergence
    """

    # Keltner params
    ema_period = 20
    atr_keltner_period = 10
    atr_mult = 2.0

    # RSI
    rsi_period = 14
    rsi_ob = 70
    rsi_os = 30

    # ATR for exits
    atr_exit_period = 14
    tp_mult = 1.5
    sl_mult = 1.0
    time_stop = 20

    # Divergence window
    pivot_lookback = 3
    div_min_bars = 5
    div_max_bars = 15

    # Risk
    risk_pct = 0.01
    size_units = 1_000_000

    def init(self):
        print("🌙 Moon Dev: Initializing KineticDivergence indicators... ✨")
        close = self.data.Close
        high = self.data.High
        low = self.data.Low

        # Keltner Channel
        self.ema = self.I(talib.EMA, close, timeperiod=self.ema_period)
        self.atr_k = self.I(talib.ATR, high, low, close, timeperiod=self.atr_keltner_period)
        self.upper = self.I(lambda: self.ema + self.atr_mult * self.atr_k)
        self.lower = self.I(lambda: self.ema - self.atr_mult * self.atr_k)

        # RSI
        self.rsi = self.I(talib.RSI, close, timeperiod=self.rsi_period)

        # ATR for exits
        self.atr_x = self.I(talib.ATR, high, low, close, timeperiod=self.atr_exit_period)

        # ATR average to filter dead volatility
        self.atr_avg = self.I(talib.SMA, self.atr_x, timeperiod=20)

        # Pivot lows/highs (3-bar fractal)
        lb = self.pivot_lookback
        self.pivot_low = self.I(
            lambda: pd.Series(low).rolling(2 * lb + 1, center=True)
            .apply(lambda x: x[lb] if x[lb] == x.min() else np.nan, raw=True)
            .values
        )
        self.pivot_high = self.I(
            lambda: pd.Series(high).rolling(2 * lb + 1, center=True)
            .apply(lambda x: x[lb] if x[lb] == x.max() else np.nan, raw=True)
            .values
        )

        print("🌙 Moon Dev: Indicators ready! 🚀")

    def _find_bullish_divergence(self, i):
        """Price lower low + RSI higher low, RSI < OS at pivot."""
        lb = self.pivot_lookback
        # Need at least a few pivots to compare
        recent = []
        for j in range(max(0, i - self.div_max_bars - lb), i - lb + 1):
            pl = self.pivot_low[j]
            if not np.isnan(pl):
                recent.append(j)
        if len(recent) < 2:
            return False

        last = recent[-1]
        prev = recent[-2]
        gap = last - prev
        if gap < self.div_min_bars or gap > self.div_max_bars:
            return False

        # Price lower low
        if self.data.Low[last] >= self.data.Low[prev]:
            return False
        # RSI higher low
        if self.rsi[last] <= self.rsi[prev]:
            return False
        # RSI oversold at pivot
        if self.rsi[last] >= self.rsi_os:
            return False
        return True

    def _find_bearish_divergence(self, i):
        """Price higher high + RSI lower high, RSI > OB at pivot."""
        lb = self.pivot_lookback
        recent = []
        for j in range(max(0, i - self.div_max_bars - lb), i - lb + 1):
            ph = self.pivot_high[j]
            if not np.isnan(ph):
                recent.append(j)
        if len(recent) < 2:
            return False

        last = recent[-1]
        prev = recent[-2]
        gap = last - prev
        if gap < self.div_min_bars or gap > self.div_max_bars:
            return False

        # Price higher high
        if self.data.High[last] <= self.data.High[prev]:
            return False
        # RSI lower high
        if self.rsi[last] >= self.rsi[prev]:
            return False
        # RSI overbought at pivot
        if self.rsi[last] <= self.rsi_ob:
            return False
        return True

    def next(self):
        i = len(self.data) - 1
        if i < 30:
            return

        # Volatility filter: skip if ATR below its average
        if self.atr_x[-1] < self.atr_avg[-1]:
            return

        price = self.data.Close[-1]

        # ===== LONG =====
        if not self.position:
            # Price closed below lower band
            if self.data.Close[-1] < self.lower[-1]:
                if self._find_bullish_divergence(i):
                    # Confirmation: first bullish close
                    if self.data.Close[-1] > self.data.Open[-1]:
                        atr = self.atr_x[-1]
                        sl = price - self.sl_mult * atr
                        tp = price + self.tp_mult * atr
                        risk_per_unit = price - sl
                        if risk_per_unit > 0:
                            equity = self.equity
                            risk_dollars = equity * self.risk_pct
                            size = int(round(risk_dollars / risk_per_unit))
                            if size > 0:
                                print(f"🌙🚀 Moon Dev LONG! price={price:.2f} SL={sl:.2f} TP={tp:.2f} size={size}")
                                self.buy(size=size, sl=sl, tp=tp)
                                self.entry_bar = i

            # ===== SHORT =====
            elif self.data.Close[-1] > self.upper[-1]:
                if self._find_bearish_divergence(i):
                    if self.data.Close[-1] < self.data.Open[-1]:
                        atr = self.atr_x[-1]
                        sl = price + self.sl_mult * atr
                        tp = price - self.tp_mult * atr
                        risk_per_unit = sl - price
                        if risk_per_unit > 0:
                            equity = self.equity
                            risk_dollars = equity * self.risk_pct
                            size = int(round(risk_dollars / risk_per_unit))
                            if size > 0:
                                print(f"🌙🚀 Moon Dev SHORT! price={price:.2f} SL={sl:.2f} TP={tp:.2f} size={size}")
                                self.sell(size=size, sl=sl, tp=tp)
                                self.entry_bar = i

        else:
            # Time stop
            if hasattr(self, 'entry_bar') and (i - self.entry_bar) >= self.time_stop:
                print(f"🌙⏰ Moon Dev TIME STOP at bar {i}")
                self.position.close()


print("🌙 Moon Dev: Running KineticDivergence backtest... 🚀✨")
bt = Backtest(
    data,
    KineticDivergence,
    cash=1_000_000,
    commission=0.001,
    exclusive_orders=False
)

stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙 Moon Dev: Backtest complete! ✨🚀")