import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# 🌙 Moon Dev's FiboPulseTrail Backtest 🚀

DATA_PATH = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"

# Load data
print("🌙 Moon Dev: Loading BTC-USD 15m data... ✨")
data = pd.read_csv(DATA_PATH)
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Map columns to backtesting.py format
data = data.rename(columns={
    'datetime': 'Datetime',
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

if 'Datetime' in data.columns:
    data['Datetime'] = pd.to_datetime(data['Datetime'])
    data = data.set_index('Datetime')

data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
print(f"🚀 Moon Dev: Data loaded with {len(data)} bars! ✨")


class FiboPulseTrail(Strategy):
    # Strategy parameters
    swing_lookback = 50
    ema_fast_period = 50
    ema_slow_period = 200
    adx_period = 14
    adx_threshold = 25
    atr_period = 14
    atr_buffer = 0.5
    risk_pct = 0.02  # 2% risk per trade
    fib_tolerance = 0.005  # 0.5% tolerance around fib levels

    def init(self):
        print("🌙 Moon Dev: Initializing FiboPulseTrail indicators... ✨")

        close = self.data.Close
        high = self.data.High
        low = self.data.Low

        # EMAs for trend filter
        self.ema_fast = self.I(talib.EMA, close, timeperiod=self.ema_fast_period, name='EMA50')
        self.ema_slow = self.I(talib.EMA, close, timeperiod=self.ema_slow_period, name='EMA200')

        # ADX for trend strength
        self.adx = self.I(talib.ADX, high, low, close, timeperiod=self.adx_period, name='ADX')
        self.plus_di = self.I(talib.PLUS_DI, high, low, close, timeperiod=self.adx_period, name='+DI')
        self.minus_di = self.I(talib.MINUS_DI, high, low, close, timeperiod=self.adx_period, name='-DI')

        # MACD oscillator
        self.macd, self.macd_signal, self.macd_hist = self.I(
            talib.MACD, close, fastperiod=12, slowperiod=26, signalperiod=9,
            name='MACD'
        )

        # ATR for stops
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period, name='ATR')

        # Swing high/low for Fibonacci anchoring
        self.swing_high = self.I(talib.MAX, high, timeperiod=self.swing_lookback, name='SwingHigh')
        self.swing_low = self.I(talib.MIN, low, timeperiod=self.swing_lookback, name='SwingLow')

        # Track trade state
        self.entry_price = None
        self.stop_price = None
        self.tp_price = None
        self.trailing_active = False

        print("🚀 Moon Dev: Indicators ready! Let's ride the FiboPulse! 🌙")

    def next(self):
        price = self.data.Close[-1]

        # Skip if indicators not ready
        if np.isnan(self.ema_slow[-1]) or np.isnan(self.adx[-1]) or np.isnan(self.atr[-1]):
            return

        # Get current values
        ema_fast = self.ema_fast[-1]
        ema_slow = self.ema_slow[-1]
        ema_fast_prev = self.ema_fast[-2] if len(self.ema_fast) > 1 else ema_fast
        adx = self.adx[-1]
        plus_di = self.plus_di[-1]
        minus_di = self.minus_di[-1]
        macd_hist = self.macd_hist[-1]
        macd_hist_prev = self.macd_hist[-2] if len(self.macd_hist) > 1 else macd_hist
        atr = self.atr[-1]

        # Swing high/low
        sh = self.swing_high[-1]
        sl = self.swing_low[-1]
        fib_range = sh - sl

        if fib_range <= 0:
            return

        # Fibonacci retracement levels (for uptrend pullbacks)
        fib_382 = sh - 0.382 * fib_range
        fib_500 = sh - 0.500 * fib_range
        fib_618 = sh - 0.618 * fib_range

        # For downtrend rallies (inverse)
        fib_382_up = sl + 0.382 * fib_range
        fib_500_up = sl + 0.500 * fib_range
        fib_618_up = sl + 0.618 * fib_range

        # Trend filters
        uptrend = price > ema_slow and ema_fast > ema_fast_prev
        downtrend = price < ema_slow and ema_fast < ema_fast_prev
        strong_trend = adx > self.adx_threshold

        # Fib touch detection (within tolerance)
        tol = self.fib_tolerance * fib_range
        near_fib_long = (
            abs(price - fib_382) < tol or
            abs(price - fib_500) < tol or
            abs(price - fib_618) < tol
        )
        near_fib_short = (
            abs(price - fib_382_up) < tol or
            abs(price - fib_500_up) < tol or
            abs(price - fib_618_up) < tol
        )

        # Oscillator signal: MACD histogram turning positive/negative
        macd_bull_cross = macd_hist_prev <= 0 and macd_hist > 0
        macd_bear_cross = macd_hist_prev >= 0 and macd_hist < 0

        # Manage existing position
        if self.position:
            self._manage_position(price, atr, ema_fast, macd_hist, macd_hist_prev)
            return

        # === LONG ENTRY ===
        if uptrend and strong_trend and plus_di > minus_di and near_fib_long and macd_bull_cross:
            print(f"🌙 Moon Dev LONG SIGNAL! Price={price:.2f} | ADX={adx:.1f} | MACD Hist={macd_hist:.4f} 🚀")
            stop = fib_618 - self.atr_buffer * atr
            risk = price - stop
            if risk > 0:
                # Use fractional sizing (percentage of equity) - valid for backtesting.py
                size_frac = min(self.risk_pct * price / risk, 0.95)
                if size_frac > 0:
                    self.buy(size=size_frac)
                    self.entry_price = price
                    self.stop_price = stop
                    self.tp_price = sh  # target previous swing high
                    self.trailing_active = False
                    print(f"✨ Moon Dev: Entered LONG size={size_frac:.4f} | Stop={stop:.2f} | TP={sh:.2f} 🌙")

        # === SHORT ENTRY ===
        elif downtrend and strong_trend and minus_di > plus_di and near_fib_short and macd_bear_cross:
            print(f"🌙 Moon Dev SHORT SIGNAL! Price={price:.2f} | ADX={adx:.1f} | MACD Hist={macd_hist:.4f} 🚀")
            stop = fib_618_up + self.atr_buffer * atr
            risk = stop - price
            if risk > 0:
                size_frac = min(self.risk_pct * price / risk, 0.95)
                if size_frac > 0:
                    self.sell(size=size_frac)
                    self.entry_price = price
                    self.stop_price = stop
                    self.tp_price = sl  # target previous swing low
                    self.trailing_active = False
                    print(f"✨ Moon Dev: Entered SHORT size={size_frac:.4f} | Stop={stop:.2f} | TP={sl:.2f} 🌙")

    def _manage_position(self, price, atr, ema_fast, macd_hist, macd_hist_prev):
        """Manage exits: stop loss, take profit, trailing stop, oscillator exit."""
        if self.position.is_long:
            # Trailing stop activation: 1x ATR in profit
            if not self.trailing_active and price >= self.entry_price + atr:
                self.trailing_active = True
                print(f"🌙 Moon Dev: Trailing stop ACTIVATED for LONG at {price:.2f} ✨")

            if self.trailing_active:
                # Chandelier-style: 2x ATR from highest close
                trail_stop = price - 2 * atr
                if trail_stop > self.stop_price:
                    self.stop_price = trail_stop

            # Stop loss
            if price <= self.stop_price:
                print(f"🌙 Moon Dev: LONG STOP hit at {price:.2f} 💥")
                self.position.close()
                return

            # Take profit
            if price >= self.tp_price:
                print(f"🌙 Moon Dev: LONG TAKE PROFIT hit at {price:.2f} 🎯")
                self.position.close()
                return

            # Oscillator exit: MACD bearish cross
            if macd_hist_prev >= 0 and macd_hist < 0:
                print(f"🌙 Moon Dev: LONG closed on MACD bearish cross at {price:.2f} 🌙")
                self.position.close()
                return

        elif self.position.is_short:
            if not self.trailing_active and price <= self.entry_price - atr:
                self.trailing_active = True
                print(f"🌙 Moon Dev: Trailing stop ACTIVATED for SHORT at {price:.2f} ✨")

            if self.trailing_active:
                trail_stop = price + 2 * atr
                if trail_stop < self.stop_price:
                    self.stop_price = trail_stop

            if price >= self.stop_price:
                print(f"🌙 Moon Dev: SHORT STOP hit at {price:.2f} 💥")
                self.position.close()
                return

            if price <= self.tp_price:
                print(f"🌙 Moon Dev: SHORT TAKE PROFIT hit at {price:.2f} 🎯")
                self.position.close()
                return

            if macd_hist_prev <= 0 and macd_hist > 0:
                print(f"🌙 Moon Dev: SHORT closed on MACD bullish cross at {price:.2f} 🌙")
                self.position.close()
                return


print("🌙 Moon Dev: Starting FiboPulseTrail backtest... 🚀✨")
bt = Backtest(data, FiboPulseTrail, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙 Moon Dev: Backtest complete! ✨🚀")