import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# ============================================================
# 🌙 MOON DEV'S VWAP CHANDE FRACTAL BACKTEST 🚀
# ============================================================

def FRAMA(high, low, close, period=10):
    """Fractal Adaptive Moving Average"""
    high = pd.Series(high)
    low = pd.Series(low)
    close = pd.Series(close)
    n = period
    half = n // 2

    # Fractal dimension calculation
    hh1 = high.rolling(half).max()
    ll1 = low.rolling(half).min()
    hh2 = high.rolling(half).max()
    ll2 = low.rolling(half).min()

    n1 = (hh1 - ll1) / half
    n2 = (high.rolling(half).max().shift(half) - low.rolling(half).min().shift(half)) / half
    n3 = (high.rolling(n).max() - low.rolling(n).min()) / n

    # Avoid division by zero
    dim = np.where((n1 + n2) > 0, (np.log(n1 + n2) - np.log(n3)) / np.log(2), 0.0)
    dim = pd.Series(dim, index=close.index).fillna(0.0).clip(1.0, 2.0)

    alpha = np.exp(-4.6 * (dim - 1.0))
    alpha = alpha.clip(0.01, 1.0).fillna(0.01)

    frama = np.zeros(len(close))
    frama[:] = np.nan
    if len(close) > n:
        frama[n] = close.iloc[n]
        for i in range(n + 1, len(close)):
            frama[i] = alpha.iloc[i] * close.iloc[i] + (1 - alpha.iloc[i]) * frama[i - 1]
    return pd.Series(frama, index=close.index)


def CMO(close, period=14):
    """Chande Momentum Oscillator"""
    close = pd.Series(close)
    diff = close.diff()
    gains = diff.clip(lower=0)
    losses = -diff.clip(upper=0)
    sum_gains = gains.rolling(period).sum()
    sum_losses = losses.rolling(period).sum()
    denom = sum_gains + sum_losses
    cmo = 100 * (sum_gains - sum_losses) / denom.replace(0, np.nan)
    return cmo


def VWAP(high, low, close, volume, period):
    """Rolling VWAP over N bars using typical price"""
    high = pd.Series(high)
    low = pd.Series(low)
    close = pd.Series(close)
    volume = pd.Series(volume)
    tp = (high + low + close) / 3.0
    pv = tp * volume
    vwap = pv.rolling(period).sum() / volume.rolling(period).sum()
    return vwap


class VWAPChandeFractal(Strategy):
    vwap_fast = 5
    vwap_slow = 50
    cmo_period = 14
    cmo_threshold = 50
    frama_period = 10
    risk_pct = 0.02
    stop_pct = 0.05

    def init(self):
        print("🌙 Moon Dev initializing VWAP Chande Fractal strategy... ✨")

        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        # VWAPs
        self.vwap_fast = self.I(VWAP, high, low, close, volume, self.vwap_fast,
                                name=f"VWAP_{self.vwap_fast}")
        self.vwap_slow = self.I(VWAP, high, low, close, volume, self.vwap_slow,
                                name=f"VWAP_{self.vwap_slow}")

        # CMO
        self.cmo = self.I(CMO, close, self.cmo_period, name="CMO")

        # FRAMA
        self.frama = self.I(FRAMA, high, low, close, self.frama_period,
                            name="FRAMA")

        self.below_frama_count = 0
        print("🚀 Moon Dev indicators loaded! Ready for launch...")

    def next(self):
        price = self.data.Close[-1]

        # ---- EXIT LOGIC ----
        if self.position:
            if price < self.frama[-1]:
                self.below_frama_count += 1
            else:
                self.below_frama_count = 0

            # Hard stop loss
            if self.trades and price < self.trades[-1].entry_price * (1 - self.stop_pct):
                print(f"🛑 Moon Dev STOP LOSS triggered at {price:.2f}")
                self.position.close()
                self.below_frama_count = 0
                return

            if self.below_frama_count >= 2:
                print(f"🌙 Moon Dev EXIT: Close {price:.2f} below FRAMA {self.frama[-1]:.2f} for 2 bars 💫")
                self.position.close()
                self.below_frama_count = 0
            return

        # ---- ENTRY LOGIC ----
        # Need enough history
        if len(self.data) < max(self.vwap_slow, self.cmo_period, self.frama_period) + 2:
            return

        # VWAP crossover: fast crosses above slow (no backtesting.lib used ✅)
        vwap_cross = (self.vwap_fast[-2] <= self.vwap_slow[-2]) and (self.vwap_fast[-1] > self.vwap_slow[-1])
        cmo_ok = self.cmo[-1] > self.cmo_threshold

        if vwap_cross and cmo_ok:
            # Position sizing: risk-based
            equity = self.equity
            risk_amount = equity * self.risk_pct
            stop_price = price * (1 - self.stop_pct)
            risk_per_unit = price - stop_price
            if risk_per_unit <= 0:
                return
            size = int(round(risk_amount / risk_per_unit))
            if size < 1:
                size = 1
            # Cap size to equity
            max_size = int(equity / price)
            if size > max_size:
                size = max_size
            if size < 1:
                return

            print(f"🚀 Moon Dev LONG SIGNAL! VWAP cross + CMO={self.cmo[-1]:.2f} | "
                  f"Price={price:.2f} | Size={size} 🌙")
            self.buy(size=size)


# ============================================================
# DATA LOADING
# ============================================================
data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
print("🌙 Loading Moon Dev data...")
data = pd.read_csv(data_path)

# Clean columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Ensure proper column mapping
data = data.rename(columns={
    'datetime': 'Date',
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

data['Date'] = pd.to_datetime(data['Date'])
data = data.set_index('Date')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

print(f"✨ Data loaded: {len(data)} bars from {data.index[0]} to {data.index[-1]}")

# ============================================================
# RUN BACKTEST
# ============================================================
bt = Backtest(data, VWAPChandeFractal, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)