import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# ============================================================
# 🌙 Moon Dev's Elastic Reversion Backtest 🚀
# ============================================================

DATA_PATH = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"

print("🌙 Moon Dev: Loading cosmic data from the stars... ✨")

# Load and clean data
data = pd.read_csv(DATA_PATH)
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

data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

print(f"🌙 Moon Dev: Data loaded with {len(data)} bars of pure galactic energy 🚀")


class ElasticReversion(Strategy):
    """
    🌙 Elastic Reversion Strategy 🌙
    Slow BB(50,2) regime filter + Fast BB(20,2) trigger + ATR(14) stop
    """
    # Strategy params
    bb_slow_period = 50
    bb_fast_period = 20
    bb_std = 2.0
    atr_period = 14
    atr_mult = 0.5
    max_bars_hold = 3
    risk_pct = 0.02  # 2% of equity risked per trade
    fixed_size = 1_000_000  # requested size

    def init(self):
        print("🌙 Moon Dev: Initializing indicators... ✨")

        close = self.data.Close
        high = self.data.High
        low = self.data.Low

        # --- Slow Bollinger Bands (50, 2) for regime filter ---
        self.bb_slow_upper, self.bb_slow_mid, self.bb_slow_lower = self.I(
            talib.BBANDS, close,
            timeperiod=self.bb_slow_period,
            nbdevup=self.bb_std,
            nbdevdn=self.bb_std,
            matype=0,
            name="BB_Slow"
        )

        # --- Fast Bollinger Bands (20, 2) for trigger + exit ---
        self.bb_fast_upper, self.bb_fast_mid, self.bb_fast_lower = self.I(
            talib.BBANDS, close,
            timeperiod=self.bb_fast_period,
            nbdevup=self.bb_std,
            nbdevdn=self.bb_std,
            matype=0,
            name="BB_Fast"
        )

        # --- ATR(14) for stop-loss ---
        self.atr = self.I(talib.ATR, high, low, close,
                          timeperiod=self.atr_period, name="ATR")

        # Track entry state
        self.entry_bar = None
        self.entry_price = None
        self.stop_price = None

        print("🌙 Moon Dev: Indicators ready! Let's ride the reversion wave 🚀")

    def next(self):
        price = self.data.Close[-1]

        # ---------------- ENTRY LOGIC ----------------
        if not self.position:
            # Condition A: price < lower band of slow BB(50,2)
            cond_a = (
                not np.isnan(self.bb_slow_lower[-1]) and
                price < self.bb_slow_lower[-1]
            )
            # Condition B: close < lower band of fast BB(20,2)
            cond_b = (
                not np.isnan(self.bb_fast_lower[-1]) and
                price < self.bb_fast_lower[-1]
            )

            if cond_a and cond_b:
                atr_val = self.atr[-1]
                if not np.isnan(atr_val) and atr_val > 0:
                    # Risk-based sizing
                    equity = self.equity
                    risk_amount = equity * self.risk_pct
                    stop_distance = self.atr_mult * atr_val
                    raw_size = risk_amount / stop_distance if stop_distance > 0 else 0

                    # Cap by fixed requested size and make integer
                    size = int(round(min(raw_size, self.fixed_size)))
                    if size <= 0:
                        size = 1

                    # 🚀 Moon Dev FIX: Cap size by available cash so order actually fills
                    max_affordable = int(equity / price) if price > 0 else 0
                    if max_affordable < 1:
                        print("🌙 Moon Dev: Not enough cash for even 1 unit, skipping... 💸")
                        return
                    if size > max_affordable:
                        size = max_affordable

                    self.stop_price = price - stop_distance
                    self.entry_bar = len(self.data) - 1
                    self.entry_price = price

                    print(f"🌙 Moon Dev LONG SIGNAL! 🚀 Price={price:.2f} | "
                          f"LB50={self.bb_slow_lower[-1]:.2f} | "
                          f"LB20={self.bb_fast_lower[-1]:.2f} | "
                          f"ATR={atr_val:.2f} | Stop={self.stop_price:.2f} | "
                          f"Size={size}")

                    self.buy(size=size)

        # ---------------- EXIT LOGIC ----------------
        else:
            bars_held = len(self.data) - 1 - self.entry_bar

            # Stop-loss check (intrabar low)
            if self.data.Low[-1] <= self.stop_price:
                print(f"🌙 Moon Dev STOP-LOSS HIT 💥 at {self.stop_price:.2f} "
                      f"(low={self.data.Low[-1]:.2f}) | bars_held={bars_held}")
                self.position.close()
                return

            # Target: fast upper band touch within 3 bars
            ub = self.bb_fast_upper[-1]
            if not np.isnan(ub) and self.data.High[-1] >= ub and bars_held <= self.max_bars_hold:
                print(f"🌙 Moon Dev TARGET HIT 🎯 at UB20={ub:.2f} "
                      f"(high={self.data.High[-1]:.2f}) | bars_held={bars_held}")
                self.position.close()
                return

            # Time-based exit at 3 bars
            if bars_held >= self.max_bars_hold:
                print(f"🌙 Moon Dev TIME EXIT ⏰ at close={price:.2f} | bars_held={bars_held}")
                self.position.close()
                return


print("🌙 Moon Dev: Launching the backtest rocket... 🚀✨")

bt = Backtest(
    data,
    ElasticReversion,
    cash=1_000_000,
    commission=0.002,
    exclusive_orders=True
)

stats = bt.run()
print(stats)
print(stats._strategy)