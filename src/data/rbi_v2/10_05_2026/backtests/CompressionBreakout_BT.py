import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# ============================================================
# 🌙 MOON DEV BACKTEST ENGINE — CompressionBreakout 🌙
# ============================================================

DATA_PATH = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"

print("🌙 [MOON DEV] Loading cosmic data from:", DATA_PATH)

data = pd.read_csv(DATA_PATH)

# --- Clean column names ---
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# --- Ensure proper column mapping ---
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

# --- Set datetime index ---
if 'datetime' in data.columns:
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')

data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

print("🚀 [MOON DEV] Data prepared. Shape:", data.shape)
print("✨ [MOON DEV] First rows:\n", data.head())


class CompressionBreakout(Strategy):
    """
    🌙 CompressionBreakout Strategy
    - Detect volatility squeeze via narrowest BBWidth over 20 bars
    - Enter long on close > UpperBand with volume + prior-high confirmation
    - Exit via 2x ATR trailing stop
    """

    # --- Strategy parameters ---
    bb_period = 20
    bb_std = 2.0
    squeeze_lookback = 20
    atr_period = 14
    atr_mult = 2.0
    vol_period = 20
    risk_pct = 0.01  # 1% equity risk per trade

    def init(self):
        print("🌙 [MOON DEV] Initializing CompressionBreakout indicators...")

        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        # --- Bollinger Bands ---
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS, close,
            timeperiod=self.bb_period,
            nbdevup=self.bb_std,
            nbdevdn=self.bb_std,
            matype=0
        )

        # --- Band Width = (Upper - Lower) / Middle ---
        def compute_bbwidth(upper, middle, lower):
            upper = np.asarray(upper, dtype=float)
            middle = np.asarray(middle, dtype=float)
            lower = np.asarray(lower, dtype=float)
            with np.errstate(divide='ignore', invalid='ignore'):
                width = (upper - lower) / middle
            return width

        self.bb_width = self.I(compute_bbwidth, self.bb_upper, self.bb_middle, self.bb_lower,
                               name="BBWidth")

        # --- Rolling min of BBWidth over squeeze_lookback ---
        def rolling_min(arr, window):
            s = pd.Series(arr)
            return s.rolling(window).min().values

        self.bb_width_min = self.I(rolling_min, self.bb_width, self.squeeze_lookback,
                                   name="BBWidthMin")

        # --- ATR ---
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period, name="ATR")

        # --- Volume SMA ---
        self.vol_sma = self.I(talib.SMA, volume, timeperiod=self.vol_period, name="VolSMA")

        # --- Trailing stop state ---
        self.trailing_stop = None
        self.entry_price = None

        print("✨ [MOON DEV] Indicators ready. Squeeze lookback:", self.squeeze_lookback)

    def next(self):
        price = self.data.Close[-1]
        high_prev = self.data.High[-2] if len(self.data.High) > 1 else np.nan

        # --- Manage open position (trailing stop) ---
        if self.position:
            atr_val = self.atr[-1]
            if not np.isnan(atr_val):
                # Ratchet stop upward only
                new_stop = price - (self.atr_mult * atr_val)
                if self.trailing_stop is None or new_stop > self.trailing_stop:
                    self.trailing_stop = new_stop
                    print(f"🌙 [MOON DEV] Trailing stop raised to {self.trailing_stop:.2f}")

                # Exit if price closes below trailing stop
                if price < self.trailing_stop:
                    print(f"🚨 [MOON DEV] Trailing stop HIT at {price:.2f} (stop={self.trailing_stop:.2f}) — exiting!")
                    self.position.close()
                    self.trailing_stop = None
                    self.entry_price = None
                    return

        # --- Check for new entry ---
        if not self.position:
            bb_width = self.bb_width[-1]
            bb_width_min = self.bb_width_min[-1]
            upper = self.bb_upper[-1]
            vol = self.data.Volume[-1]
            vol_sma = self.vol_sma[-1]

            # Guard against NaNs
            if (np.isnan(bb_width) or np.isnan(bb_width_min) or
                np.isnan(upper) or np.isnan(vol_sma) or
                np.isnan(high_prev)):
                return

            squeeze = bb_width <= bb_width_min + 1e-12
            breakout = price > upper
            vol_confirm = vol > vol_sma
            momentum_confirm = price > high_prev

            if squeeze and breakout and vol_confirm and momentum_confirm:
                atr_val = self.atr[-1]
                if np.isnan(atr_val) or atr_val <= 0:
                    return

                entry_price = price
                initial_stop = entry_price - (self.atr_mult * atr_val)
                risk_per_unit = entry_price - initial_stop

                if risk_per_unit <= 0:
                    return

                # --- Position sizing ---
                equity = self.equity
                risk_amount = equity * self.risk_pct
                size = int(round(risk_amount / risk_per_unit))

                if size < 1:
                    print("🌙 [MOON DEV] Calculated size < 1, skipping trade.")
                    return

                print(f"🚀 [MOON DEV] SQUEEZE BREAKOUT! Entry={entry_price:.2f}, "
                      f"Stop={initial_stop:.2f}, Size={size}, ATR={atr_val:.2f}")

                self.buy(size=size)
                self.entry_price = entry_price
                self.trailing_stop = initial_stop


# ============================================================
# 🌙 RUN BACKTEST
# ============================================================
print("🌙 [MOON DEV] Launching CompressionBreakout backtest...")

bt = Backtest(
    data,
    CompressionBreakout,
    cash=1_000_000,
    commission=0.0002,
    exclusive=False
)

stats = bt.run()
print(stats)
print(stats._strategy)