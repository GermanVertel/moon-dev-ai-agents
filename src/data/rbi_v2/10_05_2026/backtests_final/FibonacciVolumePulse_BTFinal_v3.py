import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

print("🌙✨ Moon Dev Backtest AI initializing Fibonacci VolumePulse...")
print("🚀 Loading cosmic data from the Moon Dev data vault...")

data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
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
    'volume': 'Volume'
}, inplace=True)

# Set datetime index
if 'datetime' in data.columns:
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')

print(f"🌙 Data loaded: {len(data)} rows of lunar price action ✨")
print(f"🚀 Columns: {list(data.columns)}")


class FibonacciVolumePulse(Strategy):
    # Strategy parameters
    ema_fast = 50
    ema_slow = 200
    atr_period = 14
    vol_sma_period = 20
    vol_multiplier = 1.5
    swing_lookback = 30
    risk_per_trade = 0.01  # 1% risk
    rr_target = 2.0  # 2:1 R:R minimum
    max_bars_in_trade = 50  # time-based exit

    def init(self):
        print("🌙✨ Initializing Moon Dev indicators...")
        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        volume = pd.Series(self.data.Volume)
        open_ = pd.Series(self.data.Open)

        # Trend filter EMAs
        self.ema_fast_line = self.I(talib.EMA, close, timeperiod=self.ema_fast)
        self.ema_slow_line = self.I(talib.EMA, close, timeperiod=self.ema_slow)

        # Volume baseline
        self.vol_sma = self.I(talib.SMA, volume, timeperiod=self.vol_sma_period)

        # ATR for trailing stops and volatility filter
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        # Swing high/low detection
        self.swing_high = self.I(talib.MAX, high, timeperiod=self.swing_lookback)
        self.swing_low = self.I(talib.MIN, low, timeperiod=self.swing_lookback)

        # Candlestick patterns
        self.hammer = self.I(talib.CDLHAMMER, open_, high, low, close)
        self.engulfing = self.I(talib.CDLENGULFING, open_, high, low, close)
        self.shooting_star = self.I(talib.CDLSHOOTINGSTAR, open_, high, low, close)

        # RSI filter to avoid extremes
        self.rsi = self.I(talib.RSI, close, timeperiod=14)

        # ATR percentile for volatility filter
        self.atr_pct = self.I(lambda x: pd.Series(x).rolling(100).rank(pct=True).values, self.atr)

        print("🚀 Moon Dev indicators ready for launch!")

    def next(self):
        # Skip if already in a position
        if self.position:
            self._manage_position()
            return

        # Need enough data
        if len(self.data) < self.swing_lookback + 5:
            return

        price = self.data.Close[-1]
        vol = self.data.Volume[-1]
        vol_avg = self.vol_sma[-1]

        # Volatility filter - skip extreme volatility
        if not np.isnan(self.atr_pct[-1]) and self.atr_pct[-1] > 0.95:
            return

        # Volume spike check
        if np.isnan(vol_avg) or vol_avg == 0:
            return
        vol_spike = vol > (vol_avg * self.vol_multiplier)

        # Trend determination
        ema_f = self.ema_fast_line[-1]
        ema_s = self.ema_slow_line[-1]
        if np.isnan(ema_f) or np.isnan(ema_s):
            return

        uptrend = price > ema_s and ema_f > ema_s
        downtrend = price < ema_s and ema_f < ema_s

        # Fibonacci grid from recent swing
        sh = self.swing_high[-1]
        sl = self.swing_low[-1]
        if np.isnan(sh) or np.isnan(sl) or sh <= sl:
            return

        swing_range = sh - sl

        # Fib levels
        fib_382 = sh - swing_range * 0.382
        fib_618 = sh - swing_range * 0.618
        fib_786 = sh - swing_range * 0.786

        # Bullish reversal candle detection
        bullish_candle = (self.hammer[-1] != 0) or (self.engulfing[-1] > 0)
        bearish_candle = (self.shooting_star[-1] != 0) or (self.engulfing[-1] < 0)

        # RSI filters
        rsi_ok_long = not np.isnan(self.rsi[-1]) and self.rsi[-1] < 70
        rsi_ok_short = not np.isnan(self.rsi[-1]) and self.rsi[-1] > 30

        # LONG entry
        if uptrend and vol_spike and bullish_candle and rsi_ok_long:
            in_fib_zone = fib_618 <= price <= fib_382
            if in_fib_zone:
                entry = price
                stop = min(self.data.Low[-1], fib_786)
                risk = entry - stop
                if risk > 0:
                    reward = swing_range * 0.618  # target toward prior swing high
                    if reward / risk >= self.rr_target:
                        # Position sizing: risk 1% of equity as fraction
                        equity = self.equity
                        risk_amount = equity * self.risk_per_trade
                        position_size = int(round(risk_amount / risk))
                        if position_size > 0:
                            # Cap size to available cash/price
                            max_units = int(equity / entry)
                            position_size = min(position_size, max_units)
                            if position_size > 0:
                                print(f"🌙✨ LONG SIGNAL! Entry: {entry:.2f} | Stop: {stop:.2f} | Risk: {risk:.2f} | Size: {position_size} 🚀")
                                self.buy(size=position_size, sl=stop, tp=entry + reward)

        # SHORT entry
        if downtrend and vol_spike and bearish_candle and rsi_ok_short:
            # For downtrend, fib levels computed from swing low up
            fib_618_s = sl + swing_range * 0.618
            fib_382_s = sl + swing_range * 0.382
            in_fib_zone = fib_382_s <= price <= fib_618_s
            if in_fib_zone:
                entry = price
                stop = max(self.data.High[-1], sl + swing_range * 0.786)
                risk = stop - entry
                if risk > 0:
                    reward = swing_range * 0.618
                    if reward / risk >= self.rr_target:
                        equity = self.equity
                        risk_amount = equity * self.risk_per_trade
                        position_size = int(round(risk_amount / risk))
                        if position_size > 0:
                            max_units = int(equity / entry)
                            position_size = min(position_size, max_units)
                            if position_size > 0:
                                print(f"🌙🔻 SHORT SIGNAL! Entry: {entry:.2f} | Stop: {stop:.2f} | Risk: {risk:.2f} | Size: {position_size} 🚀")
                                self.sell(size=position_size, sl=stop, tp=entry - reward)

    def _manage_position(self):
        # Time-based exit
        if self.position:
            # Trail stop using ATR once in profit
            atr_val = self.atr[-1]
            if np.isnan(atr_val):
                return

            # Use trades list to get entry price (Position has no .entry_price)
            try:
                entry_price = self.trades[-1].entry_price
            except (IndexError, AttributeError):
                entry_price = None

            if self.position.is_long:
                if entry_price is not None and self.data.Close[-1] > entry_price:
                    new_sl = self.data.Close[-1] - 1.5 * atr_val
                    try:
                        current_sl = self.trades[-1].sl
                    except (IndexError, AttributeError):
                        current_sl = None
                    if current_sl is None or new_sl > current_sl:
                        self.trades[-1].sl = new_sl
            elif self.position.is_short:
                if entry_price is not None and self.data.Close[-1] < entry_price:
                    new_sl = self.data.Close[-1] + 1.5 * atr_val
                    try:
                        current_sl = self.trades[-1].sl
                    except (IndexError, AttributeError):
                        current_sl = None
                    if current_sl is None or new_sl < current_sl:
                        self.trades[-1].sl = new_sl


print("🌙✨ Launching Moon Dev Fibonacci VolumePulse Backtest... 🚀")
bt = Backtest(data, FibonacciVolumePulse, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙✨ Moon Dev backtest complete! To the moon! 🚀")