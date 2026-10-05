import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# ============================================================
# 🌙 MOON DEV - KINETIC DIVERGENCE STRATEGY 🌙
# ============================================================

print("🌙 Moon Dev: Loading BTC-USD 15m data...")

data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Ensure proper column mapping
data = data.rename(columns={
    'datetime': 'datetime',
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

# Parse datetime and set index
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

print(f"✨ Moon Dev: Data loaded - {len(data)} bars")
print(f"🚀 Moon Dev: Date range: {data.index[0]} to {data.index[-1]}")


class KineticDivergence(Strategy):
    """
    🌙 KineticDivergence Strategy 🌙
    RSI divergence + volatility expansion + implied/realized vol spread
    """

    # Strategy parameters
    rsi_period = 9
    vol_period = 30
    atr_period = 14
    divergence_lookback = 5
    vol_threshold_pct = 0.75  # top quartile
    implied_realized_spread = 0.0005
    risk_pct = 0.01
    tp_atr_mult = 1.0
    sl_atr_mult = 1.0
    time_stop_bars = 4  # ~1 hour on 15m bars (proxy for time stop)

    def init(self):
        print("🌙 Moon Dev: Initializing KineticDivergence indicators...")

        close = pd.Series(self.data.Close, index=self.data.index)
        high = pd.Series(self.data.High, index=self.data.index)
        low = pd.Series(self.data.Low, index=self.data.index)
        volume = pd.Series(self.data.Volume, index=self.data.index)

        # RSI
        self.rsi = self.I(talib.RSI, close, timeperiod=self.rsi_period)

        # ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        # Log returns
        log_ret = np.log(close / close.shift(1)).fillna(0)

        # Realized volatility (rolling std of log returns)
        realized_vol = log_ret.rolling(self.vol_period).std().fillna(0)
        self.realized_vol = self.I(lambda x: x, realized_vol.values)

        # Implied vol proxy: EWMA of squared returns (annualized-ish)
        ewma_var = (log_ret ** 2).ewm(span=self.vol_period, adjust=False).mean()
        implied_vol = np.sqrt(ewma_var).fillna(0)
        self.implied_vol = self.I(lambda x: x, implied_vol.values)

        # Volatility of volatility
        vov = realized_vol.rolling(self.vol_period).std().fillna(0)
        self.vov = self.I(lambda x: x, vov.values)

        # Rolling max of realized vol for quartile detection
        vol_max = realized_vol.rolling(self.vol_period * 2).quantile(self.vol_threshold_pct).fillna(0)
        self.vol_quartile = self.I(lambda x: x, vol_max.values)

        # Volume imbalance proxy (using volume vs its average)
        vol_ma = volume.rolling(20).mean().fillna(0)
        self.vol_ma = self.I(lambda x: x, vol_ma.values)

        # VWAP
        typical = (high + low + close) / 3
        vwap = (typical * volume).rolling(20).sum() / volume.rolling(20).sum()
        vwap = vwap.fillna(close)
        self.vwap = self.I(lambda x: x, vwap.values)

        print("✨ Moon Dev: All indicators initialized!")

    def _find_swing_lows(self, arr, lookback):
        """Find recent swing lows in price."""
        lows = []
        for i in range(1, lookback):
            if arr[-i] < arr[-i - 1] and arr[-i] < arr[-i + 1]:
                lows.append((i, arr[-i]))
        return lows

    def _find_swing_highs(self, arr, lookback):
        """Find recent swing highs in price."""
        highs = []
        for i in range(1, lookback):
            if arr[-i] > arr[-i - 1] and arr[-i] > arr[-i + 1]:
                highs.append((i, arr[-i]))
        return highs

    def next(self):
        if len(self.data) < self.vol_period * 2 + 10:
            return

        price = self.data.Close[-1]
        rsi_now = self.rsi[-1]
        rsi_prev = self.rsi[-2]
        atr_now = self.atr[-1]

        if np.isnan(atr_now) or np.isnan(rsi_now) or atr_now <= 0:
            return

        realized = self.realized_vol[-1]
        implied = self.implied_vol[-1]
        vol_q = self.vol_quartile[-1]

        if np.isnan(realized) or np.isnan(implied) or np.isnan(vol_q):
            return

        # --- Volatility Regime Filter ---
        vol_expansion = realized > vol_q and realized > 0
        implied_realized = implied - realized
        spread_ok = implied_realized > self.implied_realized_spread

        # --- RSI Divergence Detection ---
        lookback = self.divergence_lookback
        close_arr = np.array(self.data.Close[-lookback - 2:])

        # Bullish divergence: price lower low, RSI higher low
        bullish_div = False
        bearish_div = False

        price_lows = self._find_swing_lows(close_arr, lookback)
        price_highs = self._find_swing_highs(close_arr, lookback)

        # Simple divergence check: compare recent price extreme vs RSI
        if len(price_lows) >= 2:
            # Most recent low vs previous low
            recent_low = min(close_arr[-lookback:])
            older_low = min(close_arr[:-lookback]) if len(close_arr) > lookback else close_arr[0]
            rsi_recent = self.rsi[-1]
            rsi_older = self.rsi[-lookback]
            if recent_low < older_low and rsi_recent > rsi_older:
                bullish_div = True

        if len(price_highs) >= 2:
            recent_high = max(close_arr[-lookback:])
            older_high = max(close_arr[:-lookback]) if len(close_arr) > lookback else close_arr[0]
            rsi_recent = self.rsi[-1]
            rsi_older = self.rsi[-lookback]
            if recent_high > older_high and rsi_recent < rsi_older:
                bearish_div = True

        # Volume imbalance proxy
        vol_ma = self.vol_ma[-1]
        volume_ok = not np.isnan(vol_ma) and self.data.Volume[-1] > vol_ma * 0.8

        # --- Entry Logic ---
        if not self.position:
            # LONG: bullish divergence + vol expansion + implied > realized
            if bullish_div and vol_expansion and spread_ok and volume_ok and rsi_now < 50:
                # Position sizing based on risk
                risk_amount = self.equity * self.risk_pct
                stop_dist = atr_now * self.sl_atr_mult
                if stop_dist > 0:
                    size = int(round(risk_amount / stop_dist))
                    if size > 0:
                        self.buy(size=size)
                        self.entry_price = price
                        self.entry_bar = len(self.data)
                        self.tp = price + atr_now * self.tp_atr_mult
                        self.sl = price - atr_now * self.sl_atr_mult
                        print(f"🚀 Moon Dev LONG! Price={price:.2f} RSI={rsi_now:.2f} Size={size} TP={self.tp:.2f} SL={self.sl:.2f}")

            # SHORT: bearish divergence + vol expansion + implied > realized
            elif bearish_div and vol_expansion and spread_ok and volume_ok and rsi_now > 50:
                risk_amount = self.equity * self.risk_pct
                stop_dist = atr_now * self.sl_atr_mult
                if stop_dist > 0:
                    size = int(round(risk_amount / stop_dist))
                    if size > 0:
                        self.sell(size=size)
                        self.entry_price = price
                        self.entry_bar = len(self.data)
                        self.tp = price - atr_now * self.tp_atr_mult
                        self.sl = price + atr_now * self.sl_atr_mult
                        print(f"🚀 Moon Dev SHORT! Price={price:.2f} RSI={rsi_now:.2f} Size={size} TP={self.tp:.2f} SL={self.sl:.2f}")

        # --- Exit Logic ---
        else:
            bars_in_trade = len(self.data) - self.entry_bar

            if self.position.is_long:
                # Take profit
                if price >= self.tp:
                    self.position.close()
                    print(f"✨ Moon Dev TP HIT (LONG)! Price={price:.2f}")
                # Stop loss
                elif price <= self.sl:
                    self.position.close()
                    print(f"🛑 Moon Dev SL HIT (LONG)! Price={price:.2f}")
                # RSI crosses through 50
                elif rsi_prev < 50 and rsi_now >= 50:
                    self.position.close()
                    print(f"🌙 Moon Dev RSI Cross Exit (LONG)! RSI={rsi_now:.2f}")
                # Time stop
                elif bars_in_trade >= self.time_stop_bars:
                    self.position.close()
                    print(f"⏰ Moon Dev Time Stop (LONG)! Bars={bars_in_trade}")

            elif self.position.is_short:
                if price <= self.tp:
                    self.position.close()
                    print(f"✨ Moon Dev TP HIT (SHORT)! Price={price:.2f}")
                elif price >= self.sl:
                    self.position.close()
                    print(f"🛑 Moon Dev SL HIT (SHORT)! Price={price:.2f}")
                elif rsi_prev > 50 and rsi_now <= 50:
                    self.position.close()
                    print(f"🌙 Moon Dev RSI Cross Exit (SHORT)! RSI={rsi_now:.2f}")
                elif bars_in_trade >= self.time_stop_bars:
                    self.position.close()
                    print(f"⏰ Moon Dev Time Stop (SHORT)! Bars={bars_in_trade}")


print("🌙 Moon Dev: Starting backtest...")

bt = Backtest(
    data,
    KineticDivergence,
    cash=1_000_000,
    commission=0.0002
)

stats = bt.run()
print(stats)
print(stats._strategy)