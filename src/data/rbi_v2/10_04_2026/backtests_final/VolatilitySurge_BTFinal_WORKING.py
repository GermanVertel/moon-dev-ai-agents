import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# 🌙 Moon Dev's VolatilitySurge Backtest 🌙

print("🚀 Moon Dev loading data from the lunar vault...")

data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper column mapping
data = data.rename(columns={
    'datetime': 'datetime',
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')

print(f"✨ Data loaded: {len(data)} bars of lunar price action")
print(f"🌙 Columns: {list(data.columns)}")


class VolatilitySurge(Strategy):
    # Strategy parameters
    bb_period = 20
    bb_std = 2
    atr_period = 14
    rsi_period = 50
    rsi_threshold = 50
    atr_stop_mult = 1.5
    risk_pct = 0.02  # 2% risk per trade
    time_stop_bars = 20

    def init(self):
        print("🌙 Initializing VolatilitySurge indicators...")

        close = self.data.Close
        high = self.data.High
        low = self.data.Low

        # Bollinger Bands
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS, close, timeperiod=self.bb_period,
            nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0,
            name=['BB_Upper', 'BB_Middle', 'BB_Lower']
        )

        # ATR
        self.atr = self.I(talib.ATR, high, low, close,
                          timeperiod=self.atr_period, name='ATR')

        # RSI(50)
        self.rsi = self.I(talib.RSI, close, timeperiod=self.rsi_period, name='RSI')

        # BB Width
        self.bb_width = self.I(lambda u, l: u - l, self.bb_upper, self.bb_lower,
                               name='BB_Width')

        # Track trade state
        self.entry_bar = None
        self.entry_price = None
        self.stop_price = None

        print("✨ Indicators ready — BB, ATR, RSI(50) all loaded 🌙")

    def next(self):
        price = self.data.Close[-1]

        # If in a position, check exit conditions
        if self.position:
            bars_held = len(self.data) - self.entry_bar

            # Exit 1: RSI crosses below 50 (momentum reversal)
            if self.rsi[-2] >= self.rsi_threshold and self.rsi[-1] < self.rsi_threshold:
                print(f"🌙 RSI(50) reversal exit at {price:.2f} | RSI={self.rsi[-1]:.2f}")
                self.position.close()
                self.entry_bar = None
                return

            # Exit 2: RSI bearish crossover from above 50
            if len(self.rsi) > 1 and self.rsi[-2] > self.rsi_threshold and self.rsi[-1] < self.rsi_threshold:
                print(f"🌙 RSI(50) cross-down exit at {price:.2f}")
                self.position.close()
                self.entry_bar = None
                return

            # Exit 3: Time stop
            if bars_held >= self.time_stop_bars:
                print(f"⏰ Time stop hit ({bars_held} bars) at {price:.2f}")
                self.position.close()
                self.entry_bar = None
                return

            return

        # Entry logic (long only, one position at a time)
        if len(self.rsi) < 2 or len(self.atr) < 1:
            return

        bb_width_now = self.bb_width[-1]
        atr_now = self.atr[-1]
        bb_upper_now = self.bb_upper[-1]
        bb_middle_now = self.bb_middle[-1]
        rsi_now = self.rsi[-1]

        # Entry conditions
        cond_vol_surge = bb_width_now > atr_now
        cond_above_basis = price > bb_middle_now
        cond_breakout = price > bb_upper_now
        cond_rsi_bull = rsi_now > self.rsi_threshold

        if cond_vol_surge and cond_above_basis and cond_breakout and cond_rsi_bull:
            # ATR-based stop
            stop_price = price - (self.atr_stop_mult * atr_now)
            risk_per_unit = price - stop_price

            if risk_per_unit <= 0:
                return

            # Position sizing: risk 2% of equity
            equity = self.equity
            risk_amount = equity * self.risk_pct
            position_size = int(round(risk_amount / risk_per_unit))

            if position_size < 1:
                position_size = 1

            print(f"🚀 MOON DEV LONG SIGNAL 🚀 | Price={price:.2f} | "
                  f"BB_Width={bb_width_now:.2f} > ATR={atr_now:.2f} | "
                  f"RSI={rsi_now:.2f} | Size={position_size} | Stop={stop_price:.2f}")

            self.buy(size=position_size, sl=stop_price)
            self.entry_bar = len(self.data)
            self.entry_price = price
            self.stop_price = stop_price


print("🌙 Running Moon Dev's VolatilitySurge backtest...")

bt = Backtest(
    data,
    VolatilitySurge,
    cash=1_000_000,
    commission=0.002,
    exclusive_orders=True
)

stats = bt.run()
print(stats)
print(stats._strategy)