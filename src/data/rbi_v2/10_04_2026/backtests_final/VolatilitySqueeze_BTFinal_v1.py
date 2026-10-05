import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# ==================== DATA LOADING ====================
data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
print("🌙 Moon Dev loading data from:", data_path)

data = pd.read_csv(data_path)

# Clean column names
data.columns = data.columns.str.strip().str.lower()
print("✨ Columns after cleaning:", list(data.columns))

# Drop unnamed columns
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Rename columns properly
data = data.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})

# Set datetime as index
data = data.set_index(pd.to_datetime(data['datetime']))
data = data.drop(columns=['datetime'])

print("🚀 Final columns:", list(data.columns))
print("🌙 Data shape:", data.shape)
print(data.head())

# ==================== STRATEGY ====================
class VolatilitySqueeze(Strategy):
    # Bollinger Bands
    bb_period = 20
    bb_std = 2.0
    # ATR
    atr_period = 14
    # RSI
    rsi_period = 14
    # Squeeze lookback (how far back to check for BB width low)
    squeeze_lookback = 100
    # Squeeze percentile threshold (BB width must be in bottom X% of lookback)
    squeeze_pct = 20
    # Breakout confirmation: price must exceed band by this ATR multiple
    breakout_atr_mult = 0.1
    # Stop loss ATR multiple
    stop_atr_mult = 1.5
    # Take profit ATR multiple
    tp_atr_mult = 3.0
    # Trailing stop ATR multiple
    trail_atr_mult = 2.0
    # Risk per trade (fraction of equity)
    risk_pct = 0.02

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low

        print("🌙✨ Initializing Volatility Squeeze indicators...")

        # Bollinger Bands
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS, close, timeperiod=self.bb_period,
            nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0
        )

        # BB Width
        self.bb_width = self.I(
            lambda u, m, l: (u - l) / m,
            self.bb_upper, self.bb_middle, self.bb_lower
        )

        # ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        # RSI
        self.rsi = self.I(talib.RSI, close, timeperiod=self.rsi_period)

        # Volume SMA for confirmation
        self.vol_sma = self.I(talib.SMA, self.data.Volume, timeperiod=20)

        # Track our own trailing stop levels (Position has no .sl attribute)
        self.current_sl = None
        self.current_tp = None

        print("🚀 Indicators ready! Moon Dev power activated.")

    def next(self):
        # Skip if not enough data
        if len(self.data) < self.squeeze_lookback + 5:
            return

        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        atr = self.atr[-1]
        rsi = self.rsi[-1]
        bb_upper = self.bb_upper[-1]
        bb_lower = self.bb_lower[-1]
        bb_width = self.bb_width[-1]

        if np.isnan(atr) or np.isnan(rsi) or np.isnan(bb_width):
            return

        # ---- Squeeze detection: BB width in bottom percentile of lookback ----
        recent_widths = self.bb_width[-self.squeeze_lookback:]
        if len(recent_widths) < self.squeeze_lookback:
            return
        threshold = np.percentile(recent_widths, self.squeeze_pct)
        in_squeeze = bb_width <= threshold

        # ---- Manage open position ----
        if self.position:
            # Trailing stop based on ATR (track manually since Position has no .sl)
            if self.position.is_long:
                new_sl = price - self.trail_atr_mult * atr
                if self.current_sl is None or new_sl > self.current_sl:
                    self.current_sl = new_sl
                    # If price falls below our trailing SL, exit
                    if price <= new_sl:
                        print(f"🌙📈 Trailing SL hit at {new_sl:.2f}, closing long")
                        self.position.close()
                        self.current_sl = None
            elif self.position.is_short:
                new_sl = price + self.trail_atr_mult * atr
                if self.current_sl is None or new_sl < self.current_sl:
                    self.current_sl = new_sl
                    if price >= new_sl:
                        print(f"🌙📉 Trailing SL hit at {new_sl:.2f}, closing short")
                        self.position.close()
                        self.current_sl = None
            return

        # ---- Long Entry ----
        breakout_long = price > (bb_upper + self.breakout_atr_mult * atr)
        if in_squeeze and breakout_long and rsi > 50:
            sl_price = low - self.stop_atr_mult * atr
            tp_price = price + self.tp_atr_mult * atr
            risk_per_unit = price - sl_price
            if risk_per_unit <= 0:
                return
            risk_amount = self.equity * self.risk_pct
            size = int(round(risk_amount / risk_per_unit))
            if size < 1:
                return
            # Cap size to available margin (cash-based)
            max_size = int(self.equity * 0.95 / price)
            if max_size < 1:
                return
            size = min(size, max_size)
            print(f"🌙🚀 LONG SQUEEZE BREAKOUT! Price={price:.2f} BB_Upper={bb_upper:.2f} "
                  f"RSI={rsi:.2f} ATR={atr:.2f} Size={size}")
            self.buy(size=size, sl=sl_price, tp=tp_price)
            self.current_sl = sl_price

        # ---- Short Entry ----
        breakout_short = price < (bb_lower - self.breakout_atr_mult * atr)
        if in_squeeze and breakout_short and rsi < 50:
            sl_price = high + self.stop_atr_mult * atr
            tp_price = price - self.tp_atr_mult * atr
            risk_per_unit = sl_price - price
            if risk_per_unit <= 0:
                return
            risk_amount = self.equity * self.risk_pct
            size = int(round(risk_amount / risk_per_unit))
            if size < 1:
                return
            max_size = int(self.equity * 0.95 / price)
            if max_size < 1:
                return
            size = min(size, max_size)
            print(f"🌙🔻 SHORT SQUEEZE BREAKOUT! Price={price:.2f} BB_Lower={bb_lower:.2f} "
                  f"RSI={rsi:.2f} ATR={atr:.2f} Size={size}")
            self.sell(size=size, sl=sl_price, tp=tp_price)
            self.current_sl = sl_price


# ==================== RUN BACKTEST ====================
print("🌙✨ Starting Moon Dev Volatility Squeeze Backtest...")
bt = Backtest(data, VolatilitySqueeze, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙🚀 Moon Dev Backtest Complete!")