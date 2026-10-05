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

        # MACD
        self.macd, self.macd_signal_line, self.macd_hist = self.I(
            talib.MACD, close,
            fastperiod=self.macd_fast,
            slowperiod=self.macd_slow,
            signalperiod=self.macd_signal
        )

        # Volume SMA
        self.vol_sma = self.I(talib.SMA, volume, timeperiod=self.vol_sma_period)

        # Bollinger Bands
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS, close,
            timeperiod=self.bb_period,
            nbdevup=self.bb_std,
            nbdevdn=self.bb_std,
            matype=0
        )

        # ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        # Swing highs/lows for divergence detection
        self.swing_high = self.I(talib.MAX, high, timeperiod=self.swing_lookback)
        self.swing_low = self.I(talib.MIN, low, timeperiod=self.swing_lookback)

        # Track entry bar for time stop
        self.entry_bar = None
        self.entry_price = None
        self.stop_price = None
        self.target_price = None

        print("✨ Moon Dev indicators ready! 🚀")

    def next(self):
        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        volume = self.data.Volume[-1]

        macd_val = self.macd[-1]
        macd_prev = self.macd[-2] if len(self.macd) > 2 else macd_val
        macd_hist = self.macd_hist[-1]

        vol_sma = self.vol_sma[-1]
        bb_up = self.bb_upper[-1]
        bb_mid = self.bb_middle[-1]
        bb_low = self.bb_lower[-1]
        atr = self.atr[-1]

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
        if len(self.data) < lookback + 2:
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
        prev_low = self.data.Low[-lookback * 2:-lookback].min() if len(self.data) >= lookback * 2 else None
        prev_high = self.data.High[-lookback * 2:-lookback].max() if len(self.data) >= lookback * 2 else None

        if prev_low is not None and prev_high is not None:
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

        # LONG ENTRY
        if bullish_div and price <= bb_low * 1.005:
            stop = min(curr_swing_low, price - 1.5 * atr)
            risk = price - stop
            if risk <= 0:
                return
            reward = bb_mid - price
            if reward < self.rr_min * risk:
                print(f"🌙❌ Long skipped - R/R too low | Reward: {reward:.2f} | Risk: {risk:.2f}")
                return

            size = int(round((self.equity * self.risk_pct) / risk))
            if size <= 0:
                size = 1
            # Cap size to prevent over-leverage
            max_size = int(self.equity / price)
            size = min(size, max_size)
            if size <= 0:
                return

            print(f"🌙🚀 LONG ENTRY | Price: {price:.2f} | Stop: {stop:.2f} | Target: {bb_mid:.2f} | Size: {size}")
            self.buy(size=size)
            self.entry_bar = len(self.data)
            self.entry_price = price
            self.stop_price = stop
            self.target_price = bb_mid

        # SHORT ENTRY
        elif bearish_div and price >= bb_up * 0.995:
            stop = max(curr_swing_high, price + 1.5 * atr)
            risk = stop - price
            if risk <= 0:
                return
            reward = price - bb_mid
            if reward < self.rr_min * risk:
                print(f"🌙❌ Short skipped - R/R too low | Reward: {reward:.2f} | Risk: {risk:.2f}")
                return

            size = int(round((self.equity * self.risk_pct) / risk))
            if size <= 0:
                size = 1
            max_size = int(self.equity / price)
            size = min(size, max_size)
            if size <= 0:
                return

            print(f"🌙🚀 SHORT ENTRY | Price: {price:.2f} | Stop: {stop:.2f} | Target: {bb_mid:.2f} | Size: {size}")
            self.sell(size=size)
            self.entry_bar = len(self.data)
            self.entry_price = price
            self.stop_price = stop
            self.target_price = bb_mid


print("🌙✨ Starting Moon Dev DivergentCompression Backtest... 🚀")

bt = Backtest(
    data,
    DivergentCompression,
    cash=1_000_000,
    commission=0.001,
    exclusive=False
)

stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙 Moon Dev Backtest Complete! ✨🚀")