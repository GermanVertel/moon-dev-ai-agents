import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# 🌙 Moon Dev's FiboMomentumSync Backtest 🌙
print("🚀 Initializing FiboMomentumSync Strategy...")

# Load data
data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper column mapping - rename by name, not by position
rename_map = {}
for col in data.columns:
    if col == 'datetime' or col == 'date' or col == 'time':
        rename_map[col] = 'datetime'
    elif col == 'open':
        rename_map[col] = 'Open'
    elif col == 'high':
        rename_map[col] = 'High'
    elif col == 'low':
        rename_map[col] = 'Low'
    elif col == 'close':
        rename_map[col] = 'Close'
    elif col == 'volume':
        rename_map[col] = 'Volume'
data = data.rename(columns=rename_map)

# Set datetime index
if 'datetime' in data.columns:
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')
elif not isinstance(data.index, pd.DatetimeIndex):
    data.index = pd.to_datetime(data.index)

# Keep only OHLCV
data = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
data = data.dropna()

print(f"✨ Data loaded: {len(data)} bars from {data.index[0]} to {data.index[-1]}")
print(f"🌙 Columns: {list(data.columns)}")


class FiboMomentumSync(Strategy):
    # Strategy parameters
    ema_fast = 50
    ema_slow = 200
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    atr_period = 14
    swing_lookback = 50
    risk_pct = 0.02
    time_stop_bars = 20

    def init(self):
        print("🌙 Initializing indicators...")

        # Trend EMAs
        self.ema50 = self.I(talib.EMA, self.data.Close, timeperiod=self.ema_fast)
        self.ema200 = self.I(talib.EMA, self.data.Close, timeperiod=self.ema_slow)

        # MACD - wrap each output separately for clean indexing
        def macd_line(close):
            m, s, h = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
            return m

        def macd_signal_fn(close):
            m, s, h = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
            return s

        def macd_hist_fn(close):
            m, s, h = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
            return h

        self.macd = self.I(macd_line, self.data.Close)
        self.macd_signal_line = self.I(macd_signal_fn, self.data.Close)
        self.macd_hist = self.I(macd_hist_fn, self.data.Close)

        # ATR for stops
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close,
                          timeperiod=self.atr_period)

        # Swing high/low
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=self.swing_lookback)
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=self.swing_lookback)

        # Track entry info
        self.entry_bar = 0
        self.entry_price_val = 0
        self.stop_price = 0
        self.tp1_price = 0
        self.tp2_price = 0
        self.tp1_hit = False

        print("✨ All indicators initialized!")

    def next(self):
        # Skip if not enough data
        if len(self.data) < self.ema_slow + 10:
            return

        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]

        ema50 = self.ema50[-1]
        ema200 = self.ema200[-1]
        macd = self.macd[-1]
        macd_sig = self.macd_signal_line[-1]
        macd_hist = self.macd_hist[-1]
        macd_hist_prev = self.macd_hist[-2]
        macd_prev = self.macd[-2]
        macd_sig_prev = self.macd_signal_line[-2]
        atr = self.atr[-1]

        swing_high = self.swing_high[-1]
        swing_low = self.swing_low[-1]

        # ============ MANAGE OPEN POSITION ============
        if self.position:
            bars_in_trade = len(self.data) - self.entry_bar

            # Time stop
            if bars_in_trade >= self.time_stop_bars and not self.tp1_hit:
                print(f"⏰ Time stop hit after {bars_in_trade} bars - closing at {price:.2f}")
                self.position.close()
                return

            if self.position.is_long:
                # Check TP1
                if not self.tp1_hit and high >= self.tp1_price:
                    self.tp1_hit = True
                    print(f"🎯 TP1 hit (long) at {self.tp1_price:.2f} - moving SL to breakeven")
                    self.stop_price = self.entry_price_val

                # Check TP2
                if self.tp1_hit and high >= self.tp2_price:
                    print(f"🚀 TP2 hit (long) at {self.tp2_price:.2f} - closing position!")
                    self.position.close()
                    return

                # Trailing stop with ATR after TP1
                if self.tp1_hit:
                    trail_stop = price - 2 * atr
                    if trail_stop > self.stop_price:
                        self.stop_price = trail_stop

                # Check stop loss
                if low <= self.stop_price:
                    print(f"🛑 Stop loss hit (long) at {self.stop_price:.2f}")
                    self.position.close()
                    return

                # Momentum exit: MACD crosses back against trade
                if not self.tp1_hit and macd < macd_sig and macd_prev >= macd_sig_prev:
                    print(f"📉 MACD bearish cross - momentum exit (long)")
                    self.position.close()
                    return

            elif self.position.is_short:
                # Check TP1
                if not self.tp1_hit and low <= self.tp1_price:
                    self.tp1_hit = True
                    print(f"🎯 TP1 hit (short) at {self.tp1_price:.2f} - moving SL to breakeven")
                    self.stop_price = self.entry_price_val

                # Check TP2
                if self.tp1_hit and low <= self.tp2_price:
                    print(f"🚀 TP2 hit (short) at {self.tp2_price:.2f} - closing position!")
                    self.position.close()
                    return

                # Trailing stop with ATR after TP1
                if self.tp1_hit:
                    trail_stop = price + 2 * atr
                    if trail_stop < self.stop_price:
                        self.stop_price = trail_stop

                # Check stop loss
                if high >= self.stop_price:
                    print(f"🛑 Stop loss hit (short) at {self.stop_price:.2f}")
                    self.position.close()
                    return

                # Momentum exit
                if not self.tp1_hit and macd > macd_sig and macd_prev <= macd_sig_prev:
                    print(f"📈 MACD bullish cross - momentum exit (short)")
                    self.position.close()
                    return
            return

        # ============ ENTRY LOGIC ============

        # Trend identification
        uptrend = price > ema200 and ema50 > ema200
        downtrend = price < ema200 and ema50 < ema200

        if not uptrend and not downtrend:
            return

        # Calculate Fibonacci levels
        swing_range = swing_high - swing_low
        if swing_range <= 0:
            return

        if uptrend:
            # Fib retracement levels for long (from swing low to swing high)
            fib_382 = swing_high - 0.382 * swing_range
            fib_500 = swing_high - 0.500 * swing_range
            fib_618 = swing_high - 0.618 * swing_range
            fib_786 = swing_high - 0.786 * swing_range

            # Price in golden zone (38.2% - 61.8%)
            in_fib_zone = fib_618 <= price <= fib_382

            if not in_fib_zone:
                return

            # MACD confirmation
            macd_bull_cross = macd > macd_sig and macd_prev <= macd_sig_prev
            macd_hist_flip = macd_hist > 0 and macd_hist_prev <= 0
            macd_bullish_div = (low < self.data.Low[-2] and macd > macd_prev)

            macd_confirm = macd_bull_cross or macd_hist_flip or macd_bullish_div

            if not macd_confirm:
                return

            # Candlestick confirmation: close above prior candle's high
            candle_confirm = price > self.data.High[-2]

            if not candle_confirm:
                return

            # Calculate stops and targets
            stop_price = fib_786 - 0.5 * atr
            risk = price - stop_price

            if risk <= 0:
                return

            tp1_price = swing_high
            tp2_price = swing_high + 0.272 * swing_range  # 127.2% extension

            # Risk-reward check (min 1:2 for TP1)
            reward1 = tp1_price - price
            if reward1 / risk < 2.0:
                print(f"⚠️ R:R too low ({reward1/risk:.2f}) - skipping long")
                return

            # Position sizing: risk 2% of equity, expressed as fraction of equity
            equity = self.equity
            risk_amount = equity * self.risk_pct
            position_units = risk_amount / risk

            # Convert to fraction of equity for backtesting.py
            position_fraction = (position_units * price) / equity

            if position_fraction <= 0:
                return
            if position_fraction > 0.95:
                position_fraction = 0.95

            print(f"🌙 LONG ENTRY | Price: {price:.2f} | Fib Zone: {fib_618:.2f}-{fib_382:.2f}")
            print(f"   SL: {stop_price:.2f} | TP1: {tp1_price:.2f} | TP2: {tp2_price:.2f}")
            print(f"   Size frac: {position_fraction:.4f} | R:R = {reward1/risk:.2f}")

            self.buy(size=position_fraction)
            self.entry_bar = len(self.data)
            self.entry_price_val = price
            self.stop_price = stop_price
            self.tp1_price = tp1_price
            self.tp2_price = tp2_price
            self.tp1_hit = False

        elif downtrend:
            # Fib retracement levels for short (from swing high to swing low)
            fib_382 = swing_low + 0.382 * swing_range
            fib_500 = swing_low + 0.500 * swing_range
            fib_618 = swing_low + 0.618 * swing_range
            fib_786 = swing_low + 0.786 * swing_range

            # Price in golden zone
            in_fib_zone = fib_382 <= price <= fib_618

            if not in_fib_zone:
                return

            # MACD confirmation
            macd_bear_cross = macd < macd_sig and macd_prev >= macd_sig_prev
            macd_hist_flip = macd_hist < 0 and macd_hist_prev >= 0
            macd_bearish_div = (high > self.data.High[-2] and macd < macd_prev)

            macd_confirm = macd_bear_cross or macd_hist_flip or macd_bearish_div

            if not macd_confirm:
                return

            # Candlestick confirmation: close below prior candle's low
            candle_confirm = price < self.data.Low[-2]

            if not candle_confirm:
                return

            # Calculate stops and targets
            stop_price = fib_786 + 0.5 * atr
            risk = stop_price - price

            if risk <= 0:
                return

            tp1_price = swing_low
            tp2_price = swing_low - 0.272 * swing_range  # 127.2% extension

            # Risk-reward check
            reward1 = price - tp1_price
            if reward1 / risk < 2.0:
                print(f"⚠️ R:R too low ({reward1/risk:.2f}) - skipping short")
                return

            # Position sizing
            equity = self.equity
            risk_amount = equity * self.risk_pct
            position_units = risk_amount / risk
            position_fraction = (position_units * price) / equity

            if position_fraction <= 0:
                return
            if position_fraction > 0.95:
                position_fraction = 0.95

            print(f"🌙 SHORT ENTRY | Price: {price:.2f} | Fib Zone: {fib_382:.2f}-{fib_618:.2f}")
            print(f"   SL: {stop_price:.2f} | TP1: {tp1_price:.2f} | TP2: {tp2_price:.2f}")
            print(f"   Size frac: {position_fraction:.4f} | R:R = {reward1/risk:.2f}")

            self.sell(size=position_fraction)
            self.entry_bar = len(self.data)
            self.entry_price_val = price
            self.stop_price = stop_price
            self.tp1_price = tp1_price
            self.tp2_price = tp2_price
            self.tp1_hit = False


print("🚀 Running backtest...")
bt = Backtest(data, FiboMomentumSync, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)