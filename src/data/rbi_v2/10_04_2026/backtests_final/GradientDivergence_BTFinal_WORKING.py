import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# ============================================================
# 🌙 Moon Dev's GradientDivergence Backtest 🌙
# ============================================================

data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"

print("🌙✨ Loading Moon Dev data from orbit...")
data = pd.read_csv(data_path, parse_dates=True, index_col=0)

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper case mapping
data.columns = [col.capitalize() for col in data.columns]

# Ensure required columns
required = ['Open', 'High', 'Low', 'Close', 'Volume']
data = data[[c for c in required if c in data.columns]]
data = data.dropna()

print(f"🚀 Moon Dev data ready! Shape: {data.shape}")
print(f"📊 Columns: {list(data.columns)}")


class GradientDivergence(Strategy):
    # Strategy parameters
    bb_period = 20
    bb_std = 2.0
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    atr_period = 14
    slope_lookback = 5
    entry_threshold = 0.15
    exit_threshold = 0.05
    macd_exit_threshold = 0.0
    divergence_min = 0.05
    risk_pct = 0.01
    stop_atr_mult = 2.0
    min_atr_floor = 0.0  # filter dead markets

    def init(self):
        print("🌙 Initializing GradientDivergence indicators...")
        close = self.data.Close
        high = self.data.High
        low = self.data.Low

        # Bollinger Bands
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS, close,
            timeperiod=self.bb_period,
            nbdevup=self.bb_std,
            nbdevdn=self.bb_std,
            matype=0
        )

        # MACD
        self.macd, self.macd_signal, self.macd_hist = self.I(
            talib.MACD, close,
            fastperiod=self.macd_fast,
            slowperiod=self.macd_slow,
            signalperiod=self.macd_signal
        )

        # ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        print("✨ Moon Dev indicators initialized!")

    def next(self):
        # Need enough bars
        if len(self.data) < max(self.bb_period, self.macd_slow, self.atr_period) + self.slope_lookback + 2:
            return

        n = self.slope_lookback
        atr_now = self.atr[-1]

        # ATR floor filter
        if atr_now <= self.min_atr_floor or atr_now <= 0:
            return

        # Upper band slope normalized by ATR
        bb_upper_now = self.bb_upper[-1]
        bb_upper_prev = self.bb_upper[-1 - n]
        bb_slope_atr = (bb_upper_now - bb_upper_prev) / (n * atr_now)

        # Lower band slope (for shorts)
        bb_lower_now = self.bb_lower[-1]
        bb_lower_prev = self.bb_lower[-1 - n]
        bb_lower_slope_atr = (bb_lower_now - bb_lower_prev) / (n * atr_now)

        # MACD slope normalized by ATR
        macd_now = self.macd[-1]
        macd_prev = self.macd[-1 - n]
        macd_slope_atr = (macd_now - macd_prev) / (n * atr_now)

        divergence = bb_slope_atr - macd_slope_atr

        price = self.data.Close[-1]

        # ==================== ENTRY LOGIC ====================
        if not self.position:
            # Long entry
            if (bb_slope_atr > self.entry_threshold and
                macd_slope_atr > 0 and
                divergence > self.divergence_min):

                stop_price = price - self.stop_atr_mult * atr_now
                risk_per_unit = price - stop_price
                if risk_per_unit <= 0:
                    return

                risk_amount = self.equity * self.risk_pct
                size = int(round(risk_amount / risk_per_unit))
                if size < 1:
                    size = 1

                print(f"🌙🚀 LONG SIGNAL! Price={price:.2f} BB_slope_ATR={bb_slope_atr:.4f} "
                      f"MACD_slope_ATR={macd_slope_atr:.4f} Div={divergence:.4f} Size={size}")
                self.buy(size=size, sl=stop_price)

            # Short entry
            elif (bb_lower_slope_atr < -self.entry_threshold and
                  macd_slope_atr < 0 and
                  (-divergence) > self.divergence_min):

                stop_price = price + self.stop_atr_mult * atr_now
                risk_per_unit = stop_price - price
                if risk_per_unit <= 0:
                    return

                risk_amount = self.equity * self.risk_pct
                size = int(round(risk_amount / risk_per_unit))
                if size < 1:
                    size = 1

                print(f"🌙🔻 SHORT SIGNAL! Price={price:.2f} BB_lower_slope_ATR={bb_lower_slope_atr:.4f} "
                      f"MACD_slope_ATR={macd_slope_atr:.4f} Div={-divergence:.4f} Size={size}")
                self.sell(size=size, sl=stop_price)

        # ==================== EXIT LOGIC ====================
        else:
            # Long exit: both slopes decelerate
            if self.position.is_long:
                if (bb_slope_atr < self.exit_threshold and
                    macd_slope_atr < self.macd_exit_threshold):
                    print(f"🌙✨ EXIT LONG: BB_slope_ATR={bb_slope_atr:.4f} MACD_slope_ATR={macd_slope_atr:.4f}")
                    self.position.close()

            # Short exit: both slopes rise above negative thresholds
            elif self.position.is_short:
                if (bb_lower_slope_atr > -self.exit_threshold and
                    macd_slope_atr > -self.macd_exit_threshold):
                    print(f"🌙✨ EXIT SHORT: BB_lower_slope_ATR={bb_lower_slope_atr:.4f} MACD_slope_ATR={macd_slope_atr:.4f}")
                    self.position.close()


print("🌙🚀 Launching Moon Dev GradientDivergence Backtest...")
bt = Backtest(
    data,
    GradientDivergence,
    cash=1_000_000,
    commission=0.001,
    exclusive_orders=True
)

stats = bt.run()
print(stats)
print(stats._strategy)