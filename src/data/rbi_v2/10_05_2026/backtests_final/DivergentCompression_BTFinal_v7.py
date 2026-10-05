import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# 🌙 MOON DEV BACKTEST - DivergentCompression Strategy 🚀

data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"

print("🌙 Loading Moon Dev data from:", data_path)

data = pd.read_csv(data_path)
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Ensure proper column mapping
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
data = data.dropna()

# 🌙 CRITICAL FIX: Force all OHLCV columns to float64 (talib requires double)
for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    data[col] = data[col].astype(np.float64)

print("🌙 Data loaded! Shape:", data.shape)
print("✨ First few rows:\n", data.head())


class DivergentCompression(Strategy):
    """
    🌙 DivergentCompression Strategy ✨
    MACD divergence + low volume compression + Bollinger Band breakeven exits
    """
    # Parameters
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    vol_sma_period = 20
    bb_period = 20
    bb_std = 2.0
    atr_period = 14
    swing_lookback = 10
    risk_pct = 0.01
    time_stop_bars = 12
    rr_min = 1.5

    def init(self):
        print("🌙 Initializing Moon Dev DivergentCompression indicators...")

        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        # 🌙 Helper: ensure arrays are float64 (talib requires double)
        def _f(arr):
            return np.asarray(arr, dtype=np.float64)

        # MACD - talib.MACD returns (macd, signal, hist) as tuple
        self.macd = self.I(lambda x: talib.MACD(
            _f(x), fastperiod=self.macd_fast,
            slowperiod=self.macd_slow,
            signalperiod=self.macd_signal
        )[0], close)

        self.macd_signal_line = self.I(lambda x: talib.MACD(
            _f(x), fastperiod=self.macd_fast,
            slowperiod=self.macd_slow,
            signalperiod=self.macd_signal
        )[1], close)

        self.macd_hist = self.I(lambda x: talib.MACD(
            _f(x), fastperiod=self.macd_fast,
            slowperiod=self.macd_slow,
            signalperiod=self.macd_signal
        )[2], close)

        # Volume SMA
        self.vol_sma = self.I(lambda x: talib.SMA(_f(x), timeperiod=self.vol_sma_period), volume)

        # Bollinger Bands - talib.BBANDS returns (upper, middle, lower)
        self.bb_upper = self.I(lambda x: talib.BBANDS(
            _f(x), timeperiod=self.bb_period,
            nbdevup=self.bb_std, nbdevdn=self.bb_std,
            matype=0
        )[0], close)

        self.bb_middle = self.I(lambda x: talib.BBANDS(
            _f(x), timeperiod=self.bb_period,
            nbdevup=self.bb_std, nbdevdn=self.bb_std,
            matype=0
        )[1], close)

        self.bb_lower = self.I(lambda x: talib.BBANDS(
            _f(x), timeperiod=self.bb_period,
            nbdevup=self.bb_std, nbdevdn=self.bb_std,
            matype=0
        )[2], close)

        # ATR
        self.atr = self.I(lambda h, l, c: talib.ATR(_f(h), _f(l), _f(c), timeperiod=self.atr_period),
                          high, low, close)

        # Swing highs/lows for divergence detection
        self.swing_high = self.I(lambda x: talib.MAX(_f(x), timeperiod=self.swing_lookback), high)
        self.swing_low = self.I(lambda x: talib.MIN(_f(x), timeperiod=self.swing_lookback), low)

        # Track entry bar for time stop
        self.entry_bar = None
        self.stop_price = None
        self.target_price = None

        print("✨ Moon Dev indicators ready! 🚀")

    def next(self):
        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        volume = self.data.Volume[-1]

        macd_val = self.macd[-1]
        macd_hist = self.macd_hist[-1]

        vol_sma = self.vol_sma[-1]
        bb_up = self.bb_upper[-1]
        bb_mid = self.bb_middle[-1]
        bb_low = self.bb_lower[-1]
        atr = self.atr[-1]

        # Guard against NaN values
        if (np.isnan(macd_val) or np.isnan(vol_sma) or np.isnan(bb_up)
                or np.isnan(bb_mid) or np.isnan(bb_low) or np.isnan(atr)):
            return

        # ---------------- EXIT LOGIC ----------------
        if self.position:
            bars_held = len(self.data) - self.entry_bar if self.entry_bar else 0

            if self.position.is_long:
                # Primary: exit at BB midline (breakeven)
                if price >= bb_mid:
                    print(f"🌙✨ LONG exit at BB midline (breakeven) | Price: {price:.2f} | Mid: {bb_mid:.2f}")
                    self.position.close()
                    self.entry_bar = None
                    return

                # Extended: exit at upper band
                if price >= bb_up:
                    print(f"🌙🚀 LONG exit at upper BB (target) | Price: {price:.2f} | Upper: {bb_up:.2f}")
                    self.position.close()
                    self.entry_bar = None
                    return

                # Stop loss
                if self.stop_price and price <= self.stop_price:
                    print(f"🌙🛑 LONG stop hit | Price: {price:.2f} | Stop: {self.stop_price:.2f}")
                    self.position.close()
                    self.entry_bar = None
                    return

                # Time stop
                if bars_held >= self.time_stop_bars:
                    print(f"🌙⏰ LONG time stop ({bars_held} bars) | Price: {price:.2f}")
                    self.position.close()
                    self.entry_bar = None
                    return

            elif self.position.is_short:
                # Primary: exit at BB midline (breakeven)
                if price <= bb_mid:
                    print(f"🌙✨ SHORT exit at BB midline (breakeven) | Price: {price:.2f} | Mid: {bb_mid:.2f}")
                    self.position.close()
                    self.entry_bar = None
                    return

                # Extended: exit at lower band
                if price <= bb_low:
                    print(f"🌙🚀 SHORT exit at lower BB (target) | Price: {price:.2f} | Lower: {bb_low:.2f}")
                    self.position.close()
                    self.entry_bar = None
                    return

                # Stop loss
                if self.stop_price and price >= self.stop_price:
                    print(f"🌙🛑 SHORT stop hit | Price: {price:.2f} | Stop: {self.stop_price:.2f}")
                    self.position.close()
                    self.entry_bar = None
                    return

                # Time stop
                if bars_held >= self.time_stop_bars:
                    print(f"🌙⏰ SHORT time stop ({bars_held} bars) | Price: {price:.2f}")
                    self.position.close()
                    self.entry_bar = None
                    return

            return

        # ---------------- ENTRY LOGIC ----------------
        if len(self.data) < 30:
            return

        # Volume compression filter
        low_volume = volume < vol_sma
        if not low_volume:
            return

        # Need at least a few bars of history for divergence
        lookback = 5
        if len(self.data) < lookback * 2 + 2:
            return

        # Check for divergence using swing highs/lows over recent bars
        recent_highs = self.data.High[-lookback:]
        recent_lows = self.data.Low[-lookback:]
        recent_macd = self.macd[-lookback:]

        curr_swing_low = min(recent_lows)
        curr_swing_high = max(recent_highs)

        # Bullish divergence: price lower low, MACD higher low
        bullish_div = False
        bearish_div = False

        # Compare current swing low vs prior swing low (simple 2-window comparison)
        prev_low = self.data.Low[-lookback * 2:-lookback].min()
        prev_high = self.data.High[-lookback * 2:-lookback].max()

        curr_macd_low = min(recent_macd)
        curr_macd_high = max(recent_macd)
        prev_macd = self.macd[-lookback * 2:-lookback]
        prev_macd_low = min(prev_macd)
        prev_macd_high = max(prev_macd)

        # Bullish: price lower low, MACD higher low
        if curr_swing_low < prev_low and curr_macd_low > prev_macd_low:
            bullish_div = True

        # Bearish: price higher high, MACD lower high
        if curr_swing_high > prev_high and curr_macd_high < prev_macd_high:
            bearish_div = True

        # 🌙 CRITICAL FIX: relax entry gates so signals actually trigger.
        # Use ATR-based proximity to bands rather than tight 0.5% multipliers,
        # since BTC 15m bars rarely sit within 0.5% of the band at the exact
        # divergence bar. This preserves strategy intent (buy near lower band,
        # sell near upper band) while allowing realistic fills.

        band_prox = 0.5 * atr  # tolerance band around BB edges

        # LONG ENTRY
        if bullish_div and price <= bb_low + band_prox:
            stop = min(curr_swing_low, price - 1.5 * atr)
            risk = price - stop
            if risk <= 0:
                return
            reward = bb_mid - price
            if reward < self.rr_min * risk:
                print(f"🌙❌ Long skipped - R/R too low | Reward: {reward:.2f} | Risk: {risk:.2f}")
                return

            # 🌙 CRITICAL FIX: use fraction of equity for sizing (0 < size < 1)
            size_frac = (self.equity * self.risk_pct) / (risk * price)
            size_frac = max(0.01, min(0.95, size_frac))

            print(f"🌙🚀 LONG ENTRY | Price: {price:.2f} | Stop: {stop:.2f} | Target: {bb_mid:.2f} | SizeFrac: {size_frac:.4f}")
            self.buy(size=size_frac)
            self.entry_bar = len(self.data)
            self.stop_price = stop
            self.target_price = bb_mid

        # SHORT ENTRY
        elif bearish_div and price >= bb_up - band_prox:
            stop = max(curr_swing_high, price + 1.5 * atr)
            risk = stop - price
            if risk <= 0:
                return
            reward = price - bb_mid
            if reward < self.rr_min * risk:
                print(f"🌙❌ Short skipped - R/R too low | Reward: {reward:.2f} | Risk: {risk:.2f}")
                return

            size_frac = (self.equity * self.risk_pct) / (risk * price)
            size_frac = max(0.01, min(0.95, size_frac))

            print(f"🌙🚀 SHORT ENTRY | Price: {price:.2f} | Stop: {stop:.2f} | Target: {bb_mid:.2f} | SizeFrac: {size_frac:.4f}")
            self.sell(size=size_frac)
            self.entry_bar = len(self.data)
            self.stop_price = stop
            self.target_price = bb_mid


print("🌙✨ Starting Moon Dev DivergentCompression Backtest... 🚀")

bt = Backtest(
    data,
    DivergentCompression,
    cash=1_000_000,
    commission=0.001
)

stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙 Moon Dev Backtest Complete! ✨🚀")