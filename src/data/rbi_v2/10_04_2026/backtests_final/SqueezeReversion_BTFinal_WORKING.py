import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# ============================================================
# 🌙 MOON DEV - SqueezeReversion Strategy 🌙
# ============================================================

print("🌙✨ Moon Dev Backtest AI initializing SqueezeReversion...")
print("🚀 Loading data from the lunar data vault...")

# Load data
data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Ensure proper column mapping
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

print(f"🌙 Data loaded: {len(data)} rows from {data.index[0]} to {data.index[-1]}")
print(f"✨ Columns: {list(data.columns)}")

# ============================================================
# Strategy Class
# ============================================================
class SqueezeReversion(Strategy):
    # Bollinger Band parameters
    bb_period = 20
    bb_std = 2.0
    # Bandwidth SMA period
    bw_sma_period = 20
    # ATR period
    atr_period = 14
    # Squeeze confirmation bars
    squeeze_bars = 3
    # Risk per trade (fraction of equity)
    risk_pct = 0.02
    # Stop loss ATR multiplier
    atr_stop_mult = 2.0
    # Max holding period
    max_hold_bars = 12

    def init(self):
        print("🌙 Initializing SqueezeReversion indicators...")
        close = self.data.Close
        high = self.data.High
        low = self.data.Low

        # Bollinger Bands
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS, close, timeperiod=self.bb_period,
            nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0
        )

        # Bandwidth = (Upper - Lower) / Middle
        def bandwidth(upper, middle, lower):
            return (upper - lower) / middle

        self.bw = self.I(bandwidth, self.bb_upper, self.bb_middle, self.bb_lower)

        # Bandwidth SMA (volatility baseline)
        self.bw_sma = self.I(talib.SMA, self.bw, timeperiod=self.bw_sma_period)

        # ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        # Track bars in position
        self.bars_in_position = 0
        self.entry_price = 0.0
        self.stop_price = 0.0

        print("✨ Indicators ready: BB(20,2), BW, BW_SMA(20), ATR(14)")

    def next(self):
        # Skip if indicators not ready
        if len(self.data) < self.bb_period + self.bw_sma_period:
            return
        if np.isnan(self.bw_sma[-1]) or np.isnan(self.atr[-1]):
            return

        price = self.data.Close[-1]
        upper = self.bb_upper[-1]
        lower = self.bb_lower[-1]
        middle = self.bb_middle[-1]
        bw = self.bw[-1]
        bw_sma = self.bw_sma[-1]
        atr = self.atr[-1]

        # ---------- MANAGE OPEN POSITION ----------
        if self.position:
            self.bars_in_position += 1
            is_short = self.position.is_short

            if is_short:
                # Primary exit: Bandwidth expands back to/above SMA
                if bw >= bw_sma:
                    print(f"🌙✨ EXIT SHORT [BW reversion] @ {price:.2f} | BW={bw:.5f} >= SMA={bw_sma:.5f}")
                    self.position.close()
                    self.bars_in_position = 0
                    return

                # Secondary exit: Stop loss (2x ATR above entry)
                if price >= self.stop_price:
                    print(f"🌙🛑 STOP LOSS HIT @ {price:.2f} | Stop={self.stop_price:.2f}")
                    self.position.close()
                    self.bars_in_position = 0
                    return

                # Time-based exit
                if self.bars_in_position >= self.max_hold_bars:
                    print(f"🌙⏰ TIME EXIT @ {price:.2f} | Held {self.bars_in_position} bars")
                    self.position.close()
                    self.bars_in_position = 0
                    return

            return

        # ---------- ENTRY LOGIC ----------
        # Confirm squeeze for at least N consecutive bars
        if len(self.bw) < self.squeeze_bars + 1:
            return

        squeeze_confirmed = True
        for i in range(1, self.squeeze_bars + 1):
            if np.isnan(self.bw[-i]) or np.isnan(self.bw_sma[-i]):
                squeeze_confirmed = False
                break
            if self.bw[-i] >= self.bw_sma[-i]:
                squeeze_confirmed = False
                break

        if not squeeze_confirmed:
            return

        # Breakout trigger: close above upper band (for short entry)
        # Optional filter: price above middle band (overextension)
        if price > upper and price > middle:
            # Calculate stop: 2x ATR above entry
            stop_price = price + (self.atr_stop_mult * atr)
            risk_per_unit = stop_price - price

            if risk_per_unit <= 0:
                return

            # Position sizing: risk 2% of equity
            equity = self.equity
            risk_amount = equity * self.risk_pct
            position_size = risk_amount / risk_per_unit
            position_size = int(round(position_size))

            if position_size <= 0:
                return

            self.entry_price = price
            self.stop_price = stop_price
            self.bars_in_position = 0

            print(f"🌙🚀 SHORT ENTRY @ {price:.2f} | Upper={upper:.2f} Middle={middle:.2f} "
                  f"| BW={bw:.5f} SMA={bw_sma:.5f} | ATR={atr:.2f} | Size={position_size}")
            self.sell(size=position_size)


# ============================================================
# Run Backtest
# ============================================================
print("🌙 Running Moon Dev SqueezeReversion backtest...")
bt = Backtest(data, SqueezeReversion, cash=1000000, commission=0.002, exclusive_orders=True)
stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙✨ Backtest complete! Moon Dev out. 🚀")