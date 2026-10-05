import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# ============================================================
# 🌙 MOON DEV'S FIBONACCI PULSE STRATEGY 🌙
# ============================================================

class FibonacciPulse(Strategy):
    # Strategy parameters
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    signal_ema_period = 26
    fib_lookback = 50          # lookback for swing high/low detection
    vol_ma_period = 20
    atr_period = 14
    atr_ma_period = 20
    atr_multiple_threshold = 1.5
    confluence_window = 5      # bars to check for confluence
    risk_pct = 0.02            # 2% risk per trade
    atr_stop_mult = 2.0        # 2x ATR stop loss
    atr_tp_mult = 3.0          # 3x ATR take profit

    def init(self):
        print("🌙✨ Moon Dev: Initializing FibonacciPulse Strategy... ✨🌙")
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        # MACD
        self.macd, self.macd_signal, self.macd_hist = self.I(
            talib.MACD, close,
            fastperiod=self.macd_fast,
            slowperiod=self.macd_slow,
            signalperiod=self.macd_signal
        )
        print("🚀 Moon Dev: MACD calculated")

        # 26-period EMA of MACD signal line
        self.macd_signal_ema = self.I(
            talib.EMA, self.macd_signal, timeperiod=self.signal_ema_period
        )
        print("🚀 Moon Dev: MACD Signal EMA calculated")

        # Volume MA
        self.vol_ma = self.I(talib.SMA, volume, timeperiod=self.vol_ma_period)
        print("🚀 Moon Dev: Volume MA calculated")

        # ATR and ATR MA
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)
        self.atr_ma = self.I(talib.SMA, self.atr, timeperiod=self.atr_ma_period)
        print("🚀 Moon Dev: ATR and ATR MA calculated")

        # Swing high/low for Fibonacci
        self.swing_high = self.I(talib.MAX, high, timeperiod=self.fib_lookback)
        self.swing_low = self.I(talib.MIN, low, timeperiod=self.fib_lookback)
        print("🚀 Moon Dev: Swing High/Low calculated")

        # Track entry bar for confluence window
        self.long_signal_bar = -1
        self.short_signal_bar = -1

    def next(self):
        # Ensure enough data
        if len(self.data) < self.fib_lookback + self.atr_ma_period + 10:
            return

        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        vol = self.data.Volume[-1]

        # Current indicator values
        macd_val = self.macd[-1]
        macd_sig = self.macd_signal[-1]
        macd_sig_ema = self.macd_signal_ema[-1]
        vol_ma = self.vol_ma[-1]
        atr = self.atr[-1]
        atr_ma = self.atr_ma[-1]
        swing_high = self.swing_high[-1]
        swing_low = self.swing_low[-1]

        if any(np.isnan([macd_val, macd_sig, macd_sig_ema, vol_ma, atr, atr_ma,
                         swing_high, swing_low])):
            return

        # Previous MACD signal for cross detection
        prev_macd_sig = self.macd_signal[-2]
        prev_macd_sig_ema = self.macd_signal_ema[-2]

        # Fibonacci levels
        swing_range = swing_high - swing_low
        if swing_range <= 0:
            return
        fib_382 = swing_high - 0.382 * swing_range
        fib_618 = swing_high - 0.618 * swing_range

        # Conditions
        vol_expansion = vol > vol_ma
        atr_multiple = atr / atr_ma if atr_ma > 0 else 0
        vol_expansion_atr = atr_multiple > self.atr_multiple_threshold

        # MACD signal cross above/below its 26 EMA
        cross_above = prev_macd_sig <= prev_macd_sig_ema and macd_sig > macd_sig_ema
        cross_below = prev_macd_sig >= prev_macd_sig_ema and macd_sig < macd_sig_ema

        # Fibonacci proximity (within 0.5% of 0.382 level)
        fib_tolerance = swing_range * 0.005
        near_fib_382 = abs(price - fib_382) <= fib_tolerance

        # Track signal bars for confluence window
        if cross_above:
            self.long_signal_bar = len(self.data) - 1
        if cross_below:
            self.short_signal_bar = len(self.data) - 1

        current_bar = len(self.data) - 1
        long_confluence = (
            self.long_signal_bar >= 0 and
            (current_bar - self.long_signal_bar) <= self.confluence_window
        )
        short_confluence = (
            self.short_signal_bar >= 0 and
            (current_bar - self.short_signal_bar) <= self.confluence_window
        )

        # ================= ENTRY LOGIC =================
        if not self.position:
            # LONG ENTRY
            if (long_confluence and near_fib_382 and vol_expansion and
                    vol_expansion_atr and price > swing_low):
                stop_price = swing_low - 0.5 * atr
                risk = price - stop_price
                if risk > 0:
                    risk_amount = self.equity * self.risk_pct
                    position_size = int(round(risk_amount / risk))
                    if position_size > 0:
                        print(f"🌙🚀 Moon Dev LONG SIGNAL! Price={price:.2f} "
                              f"Fib382={fib_382:.2f} ATR_mult={atr_multiple:.2f} "
                              f"Vol={vol:.2f} Size={position_size}")
                        self.buy(size=position_size)
                        self.long_signal_bar = -1

            # SHORT ENTRY
            elif (short_confluence and near_fib_382 and vol_expansion and
                  vol_expansion_atr and price < swing_high):
                stop_price = swing_high + 0.5 * atr
                risk = stop_price - price
                if risk > 0:
                    risk_amount = self.equity * self.risk_pct
                    position_size = int(round(risk_amount / risk))
                    if position_size > 0:
                        print(f"🌙🔻 Moon Dev SHORT SIGNAL! Price={price:.2f} "
                              f"Fib382={fib_382:.2f} ATR_mult={atr_multiple:.2f} "
                              f"Vol={vol:.2f} Size={position_size}")
                        self.sell(size=position_size)
                        self.short_signal_bar = -1

        # ================= EXIT LOGIC =================
        else:
            # Exit on ATR contraction
            if atr_multiple < 1.0:
                print(f"🌙⚠️ Moon Dev EXIT: ATR contraction ({atr_multiple:.2f})")
                self.position.close()
                return

            # Exit on MACD signal reversal
            if self.position.is_long and cross_below:
                print(f"🌙⚠️ Moon Dev EXIT LONG: MACD signal reversal")
                self.position.close()
                return
            if self.position.is_short and cross_above:
                print(f"🌙⚠️ Moon Dev EXIT SHORT: MACD signal reversal")
                self.position.close()
                return

            # ATR trailing stop / target management
            entry_price = self.position.entry_price
            if self.position.is_long:
                # Take profit at 3x ATR
                if price >= entry_price + self.atr_tp_mult * atr:
                    print(f"🌙✅ Moon Dev TP HIT LONG at {price:.2f}")
                    self.position.close()
                    return
                # Stop loss at 2x ATR
                if price <= entry_price - self.atr_stop_mult * atr:
                    print(f"🌙🛑 Moon Dev SL HIT LONG at {price:.2f}")
                    self.position.close()
                    return
            else:
                if price <= entry_price - self.atr_tp_mult * atr:
                    print(f"🌙✅ Moon Dev TP HIT SHORT at {price:.2f}")
                    self.position.close()
                    return
                if price >= entry_price + self.atr_stop_mult * atr:
                    print(f"🌙🛑 Moon Dev SL HIT SHORT at {price:.2f}")
                    self.position.close()
                    return


# ============================================================
# 🌙 DATA LOADING & BACKTEST EXECUTION 🌙
# ============================================================

print("🌙✨ Moon Dev: Loading BTC-USD 15m data... ✨🌙")
data = pd.read_csv(
    "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
)

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Rename to proper case for backtesting.py
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

# Set datetime index
if 'datetime' in data.columns:
    data['datetime'] = pd.to_datetime(data['datetime'])
    data.set_index('datetime', inplace=True)

print(f"🌙✨ Moon Dev: Data loaded! Shape={data.shape} ✨🌙")
print(data.head())

# Run backtest
bt = Backtest(
    data,
    FibonacciPulse,
    cash=1_000_000,
    commission=0.002,
    exclusive_orders=True
)

print("🌙🚀 Moon Dev: Running backtest... 🚀🌙")
stats = bt.run()
print(stats)
print(stats._strategy)