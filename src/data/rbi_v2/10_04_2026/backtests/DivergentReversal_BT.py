import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

print("🌙✨ Moon Dev Backtest AI - DivergentReversal Strategy ✨🌙")

# Load and clean data
data_path = '/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv'
data = pd.read_csv(data_path)

print("🚀 Loading BTC-USD 15m data...")
print(f"📊 Raw data shape: {data.shape}")

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper column mapping
data.columns = [col.capitalize() for col in data.columns]

# Ensure datetime index
if 'Datetime' in data.columns:
    data['Datetime'] = pd.to_datetime(data['Datetime'])
    data.set_index('Datetime', inplace=True)
    data.index.name = 'Datetime'

print(f"✅ Cleaned data shape: {data.shape}")
print(f"📋 Columns: {list(data.columns)}")


class DivergentReversal(Strategy):
    """
    🌙 DivergentReversal Strategy 🌙
    Long-only contrarian intraday setup combining:
    - MACD bullish divergence at support
    - MACD histogram bullish crossover
    - Price action support confirmation
    """

    # Strategy parameters
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    swing_lookback = 20
    support_lookback = 30
    atr_period = 14
    atr_buffer = 0.5
    rr_ratio = 2.0
    risk_pct = 0.01  # 1% risk per trade
    time_stop_bars = 15

    def init(self):
        print("🌙 Initializing DivergentReversal indicators...")

        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)

        # MACD
        self.macd_line, self.macd_signal_line, self.macd_hist = self.I(
            talib.MACD, close,
            fastperiod=self.macd_fast,
            slowperiod=self.macd_slow,
            signalperiod=self.macd_signal
        )

        # ATR for stop sizing
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        # Swing lows / support
        self.swing_low = self.I(talib.MIN, low, timeperiod=self.swing_lookback)

        # Prior swing low for divergence detection (longer lookback)
        self.prior_swing_low = self.I(talib.MIN, low, timeperiod=self.support_lookback)

        # Track entry bar for time stop
        self.entry_bar = None
        self.entry_price = None
        self.stop_price = None
        self.tp_price = None

        print("✨ Indicators ready! MACD, ATR, Swing Lows loaded 🚀")

    def next(self):
        price = self.data.Close[-1]

        # Manage open position
        if self.position:
            bars_held = len(self.data) - 1 - self.entry_bar

            # Time stop
            if bars_held >= self.time_stop_bars:
                print(f"⏰ Time stop hit after {bars_held} bars - exiting at {price:.2f}")
                self.position.close()
                return

            # MACD histogram flip negative exit
            if self.macd_hist[-1] < 0 and self.macd_hist[-2] >= 0:
                print(f"📉 MACD histogram flipped negative - exiting at {price:.2f}")
                self.position.close()
                return

            return  # Only one position at a time

        # Need enough data
        if len(self.data) < self.support_lookback + 5:
            return

        # === ENTRY LOGIC ===

        # Step 1: Detect bullish divergence
        # Price makes lower low, MACD makes higher low
        current_low = self.data.Low[-1]
        prev_low = self.data.Low[-2]
        prior_low = self.prior_swing_low[-1]

        # Price near recent support zone
        near_support = current_low <= prior_low * 1.005

        # MACD divergence: price lower low, MACD higher low
        price_lower_low = current_low < self.data.Low[-5]
        macd_higher_low = self.macd_line[-1] > self.macd_line[-5]

        divergence = price_lower_low and macd_higher_low and near_support

        # Step 2: Histogram confirmation - bullish crossover or shrinking negatives
        hist_cross = self.macd_hist[-1] > 0 and self.macd_hist[-2] <= 0
        hist_shrinking = (self.macd_hist[-1] > self.macd_hist[-2] and
                          self.macd_hist[-2] > self.macd_hist[-3] and
                          self.macd_hist[-1] < 0)
        hist_confirm = hist_cross or hist_shrinking

        # Step 3: Price action confirmation - bullish candle
        bullish_candle = self.data.Close[-1] > self.data.Open[-1]
        closes_above_support = self.data.Close[-1] > prior_low * 0.998

        # Bullish engulfing
        engulfing = (self.data.Close[-1] > self.data.Open[-1] and
                     self.data.Close[-2] < self.data.Open[-2] and
                     self.data.Close[-1] > self.data.Open[-2])

        price_action = bullish_candle and closes_above_support

        # Combined signal
        if divergence and hist_confirm and price_action:
            # Stop loss below divergence low minus ATR buffer
            stop = current_low - self.atr_buffer * self.atr[-1]
            risk = price - stop

            if risk <= 0:
                return

            # Take profit at 1:2 R:R
            tp = price + self.rr_ratio * risk

            # Position sizing: risk 1% of equity
            equity = self.equity
            risk_amount = equity * self.risk_pct
            position_size = risk_amount / risk
            position_size = int(round(position_size))

            if position_size < 1:
                return

            print(f"🌙✨ DIVERGENT REVERSAL SIGNAL DETECTED! ✨🌙")
            print(f"   💰 Entry: {price:.2f}")
            print(f"   🛑 Stop: {stop:.2f} (risk: {risk:.2f})")
            print(f"   🎯 Target: {tp:.2f} (R:R 1:{self.rr_ratio})")
            print(f"   📊 MACD: {self.macd_line[-1]:.4f} | Hist: {self.macd_hist[-1]:.4f}")
            print(f"   📦 Size: {position_size} units 🚀")

            self.buy(size=position_size, sl=stop, tp=tp)
            self.entry_bar = len(self.data) - 1
            self.entry_price = price
            self.stop_price = stop
            self.tp_price = tp


print("🚀 Starting backtest...")
bt = Backtest(
    data,
    DivergentReversal,
    cash=1_000_000,
    commission=0.002,
    exclusive_orders=True
)

stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙✨ Backtest complete! ✨🌙")