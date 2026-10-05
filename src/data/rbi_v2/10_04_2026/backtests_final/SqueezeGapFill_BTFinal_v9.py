import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# ============================================================
# 🌙 Moon Dev's SqueezeGapFill Backtest 🌙
# ============================================================

print("🚀 Moon Dev loading data... ✨")

data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Map to backtesting.py required columns
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume',
})

data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
data = data.dropna()

print(f"🌙 Data loaded: {len(data)} bars ✨")


class SqueezeGapFill(Strategy):
    # Parameters
    rsi_period = 14
    atr_period = 14
    sma_fast = 20
    sma_slow = 50
    rsi_low = 25
    rsi_high = 40
    vol_mult = 1.5
    gap_pct = 0.03
    hard_stop_pct = 0.07
    tp1_pct = 0.10
    tp2_pct = 0.20
    trail_activate = 0.08
    trail_pct = 0.05
    time_stop_bars = 15
    risk_pct = 0.01

    def init(self):
        print("🌙 Initializing Moon Dev indicators... ✨")
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=self.rsi_period)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period)
        self.sma20 = self.I(talib.SMA, self.data.Close, timeperiod=self.sma_fast)
        self.sma50 = self.I(talib.SMA, self.data.Close, timeperiod=self.sma_slow)
        self.vol_avg = self.I(talib.SMA, self.data.Volume, timeperiod=20)
        self.high20 = self.I(talib.MAX, self.data.High, timeperiod=20)
        self.low20 = self.I(talib.MIN, self.data.Low, timeperiod=20)
        self._tp1_hit = False
        self._entry_bar = None
        print("🚀 Indicators ready!")

    def next(self):
        price = self.data.Close[-1]

        # Skip if indicators not ready
        if (np.isnan(self.rsi[-1]) or np.isnan(self.atr[-1]) or
                np.isnan(self.sma20[-1]) or np.isnan(self.sma50[-1]) or
                np.isnan(self.vol_avg[-1]) or len(self.data) < 55):
            return

        # ---- Manage open position ----
        if self.position:
            entry = self.trades[-1].entry_price
            pnl_pct = (price - entry) / entry
            bars_held = len(self.data) - self.trades[-1].entry_bar

            # Hard stop
            if pnl_pct <= -self.hard_stop_pct:
                print(f"🛑 Moon Dev HARD STOP hit at {price:.2f} ({pnl_pct*100:.2f}%)")
                self.position.close()
                self._tp1_hit = False
                return

            # Take profit 1 - scale out half
            if pnl_pct >= self.tp1_pct and not self._tp1_hit:
                print(f"💰 Moon Dev TP1 +10% hit at {price:.2f} — scaling out 50% ✨")
                self.sell(size=self.position.size * 0.5)
                self._tp1_hit = True

            # Take profit 2 - full exit
            if pnl_pct >= self.tp2_pct:
                print(f"🎯 Moon Dev TP2 +20% hit at {price:.2f} — full exit 🚀")
                self.position.close()
                self._tp1_hit = False
                return

            # Trailing stop after activation
            if pnl_pct >= self.trail_activate:
                trail_level = price * (1 - self.trail_pct)
                if self.data.Low[-1] <= trail_level:
                    print(f"📉 Moon Dev TRAILING STOP at {price:.2f} ({pnl_pct*100:.2f}%)")
                    self.position.close()
                    self._tp1_hit = False
                    return

            # Time stop
            if bars_held >= self.time_stop_bars:
                print(f"⏰ Moon Dev TIME STOP after {bars_held} bars ({pnl_pct*100:.2f}%)")
                self.position.close()
                self._tp1_hit = False
                return

            return

        # ---- Entry logic ----
        prev_close = self.data.Close[-2]
        today_open = self.data.Open[-1]

        # Gap-down detection (recent gap down >= 3%)
        gap_down = (prev_close - today_open) / prev_close >= self.gap_pct

        # Price below 20 & 50 SMA (discounted)
        below_smas = price < self.sma20[-1] and price < self.sma50[-1]

        # RSI oversold but not dead cat
        rsi_ok = self.rsi_low <= self.rsi[-1] <= self.rsi_high

        # Relative volume >= 1.5x
        vol_ok = self.data.Volume[-1] >= self.vol_mult * self.vol_avg[-1]

        # Close above prior bar's high
        breakout = price > self.data.High[-2]

        # Higher low formation (recent swing low)
        higher_low = self.data.Low[-1] > self.data.Low[-2]

        if gap_down and below_smas and rsi_ok and vol_ok and breakout and higher_low:
            # Risk-based position sizing — use fraction of equity for backtesting.py
            risk_per_unit = self.atr[-1] * 2
            if risk_per_unit <= 0:
                return
            equity = self.equity
            risk_amount = equity * self.risk_pct
            units = risk_amount / risk_per_unit
            size_fraction = min(units * price / equity, 0.95)
            if size_fraction <= 0:
                return

            print(f"🌙✨ SQUEEZE SIGNAL! Price={price:.2f} RSI={self.rsi[-1]:.1f} "
                  f"Vol={self.data.Volume[-1]:.0f} (avg {self.vol_avg[-1]:.0f}) "
                  f"SizeFrac={size_fraction:.4f} 🚀")
            self.buy(size=size_fraction)
            self._tp1_hit = False
            self._entry_bar = len(self.data)


print("🚀 Moon Dev running backtest... ✨🌙")
bt = Backtest(data, SqueezeGapFill, cash=1_000_000, commission=0.002)
stats = bt.run()
print(stats)
print(stats._strategy)