import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# ============================================================
# 🌙 Moon Dev's OscillatorBandExit Backtest 🌙
# ============================================================

data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"

print("🌙 Moon Dev loading data from:", data_path)
data = pd.read_csv(data_path)

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper column mapping
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume',
}, inplace=True)

data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

print("✨ Data loaded. Shape:", data.shape)
print("🚀 Columns:", list(data.columns))


class OscillatorBandExit(Strategy):
    # Tunable params
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    bb_period = 20
    bb_std = 2.0
    ema_short = 5
    atr_period = 14
    atr_mult = 2.0
    arm_window = 5          # bars to keep exit_armed before reset
    time_stop_bars = 10     # max bars armed before forced exit
    entry_size = 0.99       # Moon Dev size — fraction of equity 🌙

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low

        # MACD
        self.macd, self.macd_signal, self.macd_hist = self.I(
            talib.MACD, close,
            fastperiod=self.macd_fast,
            slowperiod=self.macd_slow,
            signalperiod=self.macd_signal
        )

        # Bollinger Bands
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS, close,
            timeperiod=self.bb_period,
            nbdevup=self.bb_std,
            nbdevdn=self.bb_std,
            matype=0
        )

        # Short EMA for pullback confirmation
        self.ema5 = self.I(talib.EMA, close, timeperiod=self.ema_short)

        # ATR for trailing stop
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        # State
        self.exit_armed = False
        self.armed_bar = 0
        self.entry_price = 0.0
        self.trailing_stop = 0.0
        self.prev_low = None

        print("🌙✨ Indicators initialized. MACD, BB, EMA5, ATR ready! 🚀")

    def next(self):
        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]

        # Handle open position
        if self.position:
            # Update trailing stop (ATR-based)
            new_stop = price - self.atr_mult * self.atr[-1]
            if new_stop > self.trailing_stop:
                self.trailing_stop = new_stop

            # Check trailing stop hit
            if low <= self.trailing_stop:
                print(f"🛑🌙 Trailing stop hit at {self.trailing_stop:.2f} | price={price:.2f}")
                self.position.close()
                self._reset_state()
                return

            # ---- Arming logic ----
            macd_cross_up = (
                self.macd[-1] > self.macd_signal[-1]
                and self.macd[-2] <= self.macd_signal[-2]
            )
            bb_touch = high >= self.bb_upper[-1]

            if not self.exit_armed and macd_cross_up and bb_touch:
                self.exit_armed = True
                self.armed_bar = len(self.data)
                print(f"⚡🌙 EXIT ARMED! MACD cross up + BB touch at price={price:.2f}")
                return

            # ---- If armed, look for pullback confirmation ----
            if self.exit_armed:
                bars_armed = len(self.data) - self.armed_bar

                # Pullback confirmation: close below prev bar low OR close below 5-EMA
                pullback = False
                if self.prev_low is not None and price < self.prev_low:
                    pullback = True
                if price < self.ema5[-1]:
                    pullback = True

                # Momentum exhaustion: MACD histogram declining 2+ bars
                hist_declining = (
                    self.macd_hist[-1] < self.macd_hist[-2]
                    and self.macd_hist[-2] < self.macd_hist[-3]
                )

                if pullback:
                    print(f"✅🌙 PULLBACK CONFIRMED — EXITING LONG at {price:.2f} | bars_armed={bars_armed}")
                    self.position.close()
                    self._reset_state()
                    return

                if hist_declining and high >= self.bb_upper[-1]:
                    print(f"⚠️🌙 Momentum exhaustion — EXITING LONG at {price:.2f}")
                    self.position.close()
                    self._reset_state()
                    return

                if bars_armed >= self.time_stop_bars:
                    print(f"⏰🌙 Time stop reached ({bars_armed} bars) — EXITING at {price:.2f}")
                    self.position.close()
                    self._reset_state()
                    return

                if bars_armed >= self.arm_window:
                    print(f"🔄🌙 Arming window expired ({bars_armed} bars) — resetting exit_armed")
                    self.exit_armed = False

        else:
            # No position — enter a long (prerequisite for this exit strategy)
            # Simple entry: MACD cross up + price above middle band (trend bias)
            macd_cross_up = (
                self.macd[-1] > self.macd_signal[-1]
                and self.macd[-2] <= self.macd_signal[-2]
            )
            if macd_cross_up and price > self.bb_middle[-1]:
                self.buy(size=self.entry_size)
                self.entry_price = price
                self.trailing_stop = price - self.atr_mult * self.atr[-1]
                print(f"🚀🌙 LONG ENTRY at {price:.2f} | size={self.entry_size} | trailing_stop={self.trailing_stop:.2f}")

        # Track previous low
        self.prev_low = low

    def _reset_state(self):
        self.exit_armed = False
        self.armed_bar = 0
        self.trailing_stop = 0.0
        print("🌙✨ State reset — ready for next opportunity 🚀")


print("🌙🚀 Starting Moon Dev OscillatorBandExit backtest...")
bt = Backtest(
    data,
    OscillatorBandExit,
    cash=1_000_000,
    commission=0.001
)

stats = bt.run()
print(stats)
print(stats._strategy)