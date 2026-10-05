import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Load data
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper column mapping
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

print("🌙✨ Moon Dev: Data loaded and cleaned! Shape:", data.shape)
print("🚀 Moon Dev: Columns:", list(data.columns))


class FractalVolatilitySync(Strategy):
    """
    FractalVolatilitySync Strategy 🌙
    Combines Bill Williams Fractals + RSI + ATR + EMA50
    """
    # Parameters
    rsi_period = 14
    atr_period = 14
    atr_ma_period = 20
    ema_period = 50
    risk_pct = 0.02  # 2% risk per trade
    atr_sl_mult = 1.5
    atr_tp_mult = 3.0
    trail_trigger_mult = 1.5
    trail_mult = 1.0
    max_positions = 3

    def init(self):
        # 🌙 Moon Dev: Compute all indicators with self.I() — NO backtesting.lib!
        close = self.data.Close
        high = self.data.High
        low = self.data.Low

        self.rsi = self.I(talib.RSI, close, timeperiod=self.rsi_period, name='RSI')
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period, name='ATR')
        self.atr_ma = self.I(talib.SMA, self.atr, timeperiod=self.atr_ma_period, name='ATR_MA')
        self.ema = self.I(talib.EMA, close, timeperiod=self.ema_period, name='EMA50')

        # 🌙 Moon Dev: Fractal detection using rolling max/min (Bill Williams 5-bar pattern)
        # Bullish fractal: middle bar has lowest low, 2 bars on each side have higher lows
        # Bearish fractal: middle bar has highest high, 2 bars on each side have lower highs
        # Use talib.MIN/MAX over 5-bar window and compare center bar
        self.roll_min_low = self.I(talib.MIN, low, timeperiod=5, name='RollMinLow')
        self.roll_max_high = self.I(talib.MAX, high, timeperiod=5, name='RollMaxHigh')

        # Store raw arrays for fractal detection
        self.low_arr = low
        self.high_arr = high

        # Track trade state
        self.entry_price = None
        self.entry_atr = None
        self.trade_direction = 0  # 1 long, -1 short
        self.trail_active = False
        self.trail_stop = None

        print("🌙✨ Moon Dev: Indicators initialized! RSI, ATR, ATR_MA, EMA50 ready 🚀")

    def _is_bullish_fractal(self, i):
        """Check if bar i-2 is a bullish fractal (confirmed at bar i)."""
        if i < 5:
            return False
        c = i - 2  # center bar
        lows = self.low_arr
        # Bullish fractal: center low is lowest of 5 bars
        if (lows[c] < lows[c-1] and lows[c] < lows[c-2] and
                lows[c] < lows[c+1] and lows[c] < lows[c+2]):
            return True
        return False

    def _is_bearish_fractal(self, i):
        """Check if bar i-2 is a bearish fractal (confirmed at bar i)."""
        if i < 5:
            return False
        c = i - 2
        highs = self.high_arr
        if (highs[c] > highs[c-1] and highs[c] > highs[c-2] and
                highs[c] > highs[c+1] and highs[c] > highs[c+2]):
            return True
        return False

    def _bullish_fractal_recent(self, i, window=2):
        """Check for bullish fractal within confirmation window."""
        for k in range(1, window + 2):
            if self._is_bullish_fractal(i - k + 1):
                return True
        return False

    def _bearish_fractal_recent(self, i, window=2):
        for k in range(1, window + 2):
            if self._is_bearish_fractal(i - k + 1):
                return True
        return False

    def next(self):
        i = len(self.data) - 1

        if i < 60:
            return

        price = self.data.Close[-1]
        rsi = self.rsi[-1]
        rsi_prev = self.rsi[-2]
        atr = self.atr[-1]
        atr_ma = self.atr_ma[-1]
        ema = self.ema[-1]

        if np.isnan(atr) or np.isnan(atr_ma) or np.isnan(ema) or np.isnan(rsi):
            return

        # 🌙 Moon Dev: ATR expansion filter
        atr_expanding = atr > atr_ma

        # 🌙 Moon Dev: Manage existing trades (trailing stop)
        if self.position:
            entry = self.trades[0].entry_price if self.trades else None
            if self.trades:
                t = self.trades[0]
                # Trailing stop logic
                if t.is_long:
                    profit = price - t.entry_price
                    if not self.trail_active and profit >= self.trail_trigger_mult * self.entry_atr:
                        self.trail_active = True
                        self.trail_stop = price - self.trail_mult * self.entry_atr
                        print(f"🌙✨ Moon Dev: Trailing stop ACTIVATED (LONG) at {self.trail_stop:.2f} 🚀")
                    elif self.trail_active:
                        new_stop = price - self.trail_mult * self.entry_atr
                        if new_stop > self.trail_stop:
                            self.trail_stop = new_stop
                            print(f"🌙 Moon Dev: Trailing stop updated (LONG) → {self.trail_stop:.2f}")
                        if price <= self.trail_stop:
                            print(f"🌙 Moon Dev: Trailing stop HIT (LONG) at {price:.2f} 💫")
                            t.close()
                            self._reset_trade_state()
                            return
                else:
                    profit = t.entry_price - price
                    if not self.trail_active and profit >= self.trail_trigger_mult * self.entry_atr:
                        self.trail_active = True
                        self.trail_stop = price + self.trail_mult * self.entry_atr
                        print(f"🌙✨ Moon Dev: Trailing stop ACTIVATED (SHORT) at {self.trail_stop:.2f} 🚀")
                    elif self.trail_active:
                        new_stop = price + self.trail_mult * self.entry_atr
                        if new_stop < self.trail_stop:
                            self.trail_stop = new_stop
                            print(f"🌙 Moon Dev: Trailing stop updated (SHORT) → {self.trail_stop:.2f}")
                        if price >= self.trail_stop:
                            print(f"🌙 Moon Dev: Trailing stop HIT (SHORT) at {price:.2f} 💫")
                            t.close()
                            self._reset_trade_state()
                            return

            # Signal exit: RSI crossing against position
            if t.is_long and rsi > 70:
                print(f"🌙 Moon Dev: RSI overbought exit LONG at {price:.2f} 🚀")
                t.close()
                self._reset_trade_state()
                return
            if t.is_short and rsi < 30:
                print(f"🌙 Moon Dev: RSI oversold exit SHORT at {price:.2f} 🚀")
                t.close()
                self._reset_trade_state()
                return

        # 🌙 Moon Dev: Entry logic
        if len(self.trades) >= self.max_positions:
            return

        # Long entry conditions
        bull_fractal = self._bullish_fractal_recent(i, window=2)
        rsi_long = (rsi > 30 and rsi > rsi_prev) or (50 <= rsi <= 70)
        trend_long = price > ema

        if bull_fractal and rsi_long and atr_expanding and trend_long and not self.position:
            # Position sizing: risk 2% of equity, stop = 1.5*ATR
            risk_amount = self.equity * self.risk_pct
            stop_dist = self.atr_sl_mult * atr
            if stop_dist <= 0:
                return
            size = int(round(risk_amount / stop_dist))
            if size < 1:
                size = 1

            sl = price - stop_dist
            tp = price + self.atr_tp_mult * atr
            print(f"🌙✨ Moon Dev: LONG SIGNAL 🚀 | Price={price:.2f} RSI={rsi:.2f} ATR={atr:.2f} "
                  f"EMA={ema:.2f} SL={sl:.2f} TP={tp:.2f} Size={size}")
            self.buy(size=size, sl=sl, tp=tp)
            self.entry_price = price
            self.entry_atr = atr
            self.trade_direction = 1
            self.trail_active = False
            self.trail_stop = None

        # Short entry conditions
        elif not self.position:
            bear_fractal = self._bearish_fractal_recent(i, window=2)
            rsi_short = (rsi < 70 and rsi < rsi_prev) or (30 <= rsi <= 50)
            trend_short = price < ema

            if bear_fractal and rsi_short and atr_expanding and trend_short:
                risk_amount = self.equity * self.risk_pct
                stop_dist = self.atr_sl_mult * atr
                if stop_dist <= 0:
                    return
                size = int(round(risk_amount / stop_dist))
                if size < 1:
                    size = 1

                sl = price + stop_dist
                tp = price - self.atr_tp_mult * atr
                print(f"🌙✨ Moon Dev: SHORT SIGNAL 🚀 | Price={price:.2f} RSI={rsi:.2f} ATR={atr:.2f} "
                      f"EMA={ema:.2f} SL={sl:.2f} TP={tp:.2f} Size={size}")
                self.sell(size=size, sl=sl, tp=tp)
                self.entry_price = price
                self.entry_atr = atr
                self.trade_direction = -1
                self.trail_active = False
                self.trail_stop = None

    def _reset_trade_state(self):
        self.entry_price = None
        self.entry_atr = None
        self.trade_direction = 0
        self.trail_active = False
        self.trail_stop = None


# 🌙 Moon Dev: Run the backtest
print("🌙🚀 Moon Dev: Starting FractalVolatilitySync backtest...")
bt = Backtest(
    data,
    FractalVolatilitySync,
    cash=1_000_000,
    commission=0.002,
    exclusive=False
)

stats = bt.run()
print("🌙✨ Moon Dev: Backtest complete! Printing full stats...")
print(stats)
print(stats._strategy)