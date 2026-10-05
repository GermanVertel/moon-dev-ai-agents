import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# ============================================================
# 🌙 Moon Dev's LiquiditySentinel Backtest 🚀
# ============================================================

print("🌙 Moon Dev: Loading data from the lunar data vault...")

data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper case mapping
data.columns = [c.capitalize() for c in data.columns]

# Ensure datetime index
if 'Datetime' in data.columns:
    data['Datetime'] = pd.to_datetime(data['Datetime'])
    data = data.set_index('Datetime')
elif 'Date' in data.columns:
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.set_index('Date')

# Keep only required columns
data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

print(f"🌙 Moon Dev: Data loaded successfully! Shape: {data.shape} ✨")
print(f"🚀 First few rows:\n{data.head()}")


class LiquiditySentinel(Strategy):
    """
    🌙 LiquiditySentinel Strategy
    - Liquidity filter (proxy via dollar volume for single-asset)
    - VIX spread z-score proxy: using realized vol term structure proxy
    - Momentum-contrarian overlay
    """

    # --- Tunable parameters ---
    liquidity_window = 20
    momentum_window = 10
    sma_window = 50
    atr_window = 14
    spread_window = 20

    momentum_threshold = 0.02   # 2%
    zscore_threshold = 1.0
    atr_tp_mult = 2.5
    atr_sl_mult = 1.2

    risk_per_trade = 0.01       # 1% portfolio risk
    max_positions = 5
    time_stop_bars = 15

    def init(self):
        print("🌙 Moon Dev: Initializing LiquiditySentinel indicators... ✨")

        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        # --- Liquidity proxy: 20-day avg dollar volume ---
        dollar_vol = close * volume
        self.dollar_vol_ma = self.I(
            lambda s: pd.Series(s).rolling(self.liquidity_window).mean().values,
            dollar_vol,
            name="DollarVolMA"
        )

        # --- Trend filter: 50-day SMA ---
        self.sma50 = self.I(talib.SMA, close, timeperiod=self.sma_window, name="SMA50")

        # --- Momentum: 10-day ROC ---
        self.momentum = self.I(talib.ROC, close, timeperiod=self.momentum_window, name="MOM10")

        # --- ATR(14) ---
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_window, name="ATR14")

        # --- VIX spread proxy ---
        # Since we only have BTC data, we construct a proxy for "institutional
        # sentiment spread" using short-term vs long-term realized volatility.
        # This mirrors the term-structure logic: short-vol vs long-vol spread.
        returns = pd.Series(close).pct_change()

        short_vol = returns.rolling(5).std() * np.sqrt(252)
        long_vol = returns.rolling(20).std() * np.sqrt(252)

        # Spread = short_vol - long_vol (like front - second month)
        self.vix_spread = self.I(
            lambda s: s.values,
            (short_vol - long_vol).fillna(0),
            name="VIXSpreadProxy"
        )

        # Z-score of spread over 20 days
        def zscore_fn(s):
            s = pd.Series(s)
            mean = s.rolling(self.spread_window).mean()
            std = s.rolling(self.spread_window).std()
            return ((s - mean) / std).fillna(0).values

        self.spread_z = self.I(zscore_fn, self.vix_spread, name="SpreadZ")

        # Track entries for time stop
        self.entry_bar = None
        self.entry_price = None
        self.entry_atr = None
        self.trade_direction = None

        print("🌙 Moon Dev: All indicators ready! 🚀")

    def next(self):
        price = self.data.Close[-1]
        sma = self.sma50[-1]
        mom = self.momentum[-1]
        z = self.spread_z[-1]
        atr = self.atr[-1]

        # Skip if indicators not ready
        if np.isnan(sma) or np.isnan(mom) or np.isnan(z) or np.isnan(atr):
            return

        # --- Handle open position exits ---
        if self.position:
            self._manage_exit(price, z, mom)
            return

        # --- Entry logic ---
        long_signal = (z > self.zscore_threshold) and (mom > self.momentum_threshold * 100) and (price > sma)
        short_signal = (z < -self.zscore_threshold) and (mom < -self.momentum_threshold * 100) and (price < sma)

        if long_signal:
            print(f"🌙 Moon Dev: 🟢 LONG signal! Z={z:.2f} MOM={mom:.2f}% Price={price:.2f} SMA={sma:.2f} ✨")
            self._enter('long', price, atr)
        elif short_signal:
            print(f"🌙 Moon Dev: 🔴 SHORT signal! Z={z:.2f} MOM={mom:.2f}% Price={price:.2f} SMA={sma:.2f} ✨")
            self._enter('short', price, atr)

    def _enter(self, direction, price, atr):
        # Risk-based position sizing
        risk_amount = self.equity * self.risk_per_trade
        stop_distance = atr * self.atr_sl_mult

        if stop_distance <= 0:
            return

        # Size in units (integer)
        size = int(round(risk_amount / stop_distance))
        if size < 1:
            size = 1

        if direction == 'long':
            self.buy(size=size)
            self.entry_price = price
            self.entry_atr = atr
            self.trade_direction = 'long'
            self.entry_bar = len(self.data)
            print(f"🚀 Moon Dev: Entered LONG size={size} @ {price:.2f} | SL={price - stop_distance:.2f} TP={price + atr*self.atr_tp_mult:.2f}")
        else:
            self.sell(size=size)
            self.entry_price = price
            self.entry_atr = atr
            self.trade_direction = 'short'
            self.entry_bar = len(self.data)
            print(f"🚀 Moon Dev: Entered SHORT size={size} @ {price:.2f} | SL={price + stop_distance:.2f} TP={price - atr*self.atr_tp_mult:.2f}")

    def _manage_exit(self, price, z, mom):
        if self.entry_price is None or self.entry_atr is None:
            return

        tp_dist = self.entry_atr * self.atr_tp_mult
        sl_dist = self.entry_atr * self.atr_sl_mult
        bars_held = len(self.data) - self.entry_bar

        if self.trade_direction == 'long':
            # Profit target
            if price >= self.entry_price + tp_dist:
                print(f"🌙 Moon Dev: 🎯 LONG TP hit @ {price:.2f} 🚀")
                self.position.close()
                self._reset()
                return
            # Stop loss
            if price <= self.entry_price - sl_dist:
                print(f"🌙 Moon Dev: 🛑 LONG SL hit @ {price:.2f} 💥")
                self.position.close()
                self._reset()
                return
            # Signal decay
            if z < 0:
                print(f"🌙 Moon Dev: 📉 LONG signal decay (Z={z:.2f}) — exiting ✨")
                self.position.close()
                self._reset()
                return
            # Momentum reversal
            if mom < 0:
                print(f"🌙 Moon Dev: 🔄 LONG momentum reversal (MOM={mom:.2f}) — exiting ✨")
                self.position.close()
                self._reset()
                return

        elif self.trade_direction == 'short':
            if price <= self.entry_price - tp_dist:
                print(f"🌙 Moon Dev: 🎯 SHORT TP hit @ {price:.2f} 🚀")
                self.position.close()
                self._reset()
                return
            if price >= self.entry_price + sl_dist:
                print(f"🌙 Moon Dev: 🛑 SHORT SL hit @ {price:.2f} 💥")
                self.position.close()
                self._reset()
                return
            if z > 0:
                print(f"🌙 Moon Dev: 📈 SHORT signal decay (Z={z:.2f}) — exiting ✨")
                self.position.close()
                self._reset()
                return
            if mom > 0:
                print(f"🌙 Moon Dev: 🔄 SHORT momentum reversal (MOM={mom:.2f}) — exiting ✨")
                self.position.close()
                self._reset()
                return

        # Time stop
        if bars_held >= self.time_stop_bars:
            print(f"🌙 Moon Dev: ⏰ Time stop hit after {bars_held} bars — exiting ✨")
            self.position.close()
            self._reset()

    def _reset(self):
        self.entry_price = None
        self.entry_atr = None
        self.trade_direction = None
        self.entry_bar = None


print("🌙 Moon Dev: Launching backtest engine... 🚀✨")

bt = Backtest(
    data,
    LiquiditySentinel,
    cash=1_000_000,
    commission=0.0002,
    exclusive_orders=True,
)

stats = bt.run()

print("\n" + "=" * 60)
print("🌙 Moon Dev's LiquiditySentinel — FINAL STATS 🚀")
print("=" * 60)
print(stats)
print("=" * 60)
print(stats._strategy)
print("=" * 60)