import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Load and clean data
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

print("🌙 Moon Dev Backtest Loading... ExhaustionDivergence Strategy ✨")
print(f"📊 Data shape: {data.shape}")
print(f"🚀 Starting backtest...")


class ExhaustionDivergence(Strategy):
    # Strategy parameters
    rsi_period = 14
    atr_period = 14
    ema_fast = 20
    ema_slow = 50
    swing_lookback = 20
    pivot_window = 5
    risk_pct = 0.01
    max_positions = 2
    rsi_overbought = 75
    time_stop_bars = 15
    tp1_mult = 1.5
    tp2_mult = 2.5
    tp3_mult = 4.0
    sl_mult = 1.5

    def init(self):
        print("🌙 Initializing indicators...")
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=self.rsi_period)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period)
        self.ema20 = self.I(talib.EMA, self.data.Close, timeperiod=self.ema_fast)
        self.ema50 = self.I(talib.EMA, self.data.Close, timeperiod=self.ema_slow)
        self.vol_sma = self.I(talib.SMA, self.data.Volume, timeperiod=20)

        # Swing highs/lows using talib MAX/MIN
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=self.swing_lookback)
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=self.swing_lookback)

        # Track trade state
        self.entry_bar = None
        self.stop_price = None
        self.tp1_price = None
        self.tp2_price = None
        self.tp3_price = None
        self.tp1_hit = False
        self.tp2_hit = False
        self.initial_size = 0
        print("✨ Indicators ready!")

    def next(self):
        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        volume = self.data.Volume[-1]

        # Manage existing position
        if self.position:
            bars_held = len(self.data) - self.entry_bar

            # Time stop
            if bars_held >= self.time_stop_bars and not self.tp1_hit:
                print(f"⏰ Time stop hit at bar {len(self.data)} | Price: {price:.2f}")
                self.position.close()
                self._reset_state()
                return

            # Check TP levels
            if not self.tp1_hit and high >= self.tp1_price:
                self.tp1_hit = True
                # Scale out 1/3
                close_size = int(round(self.initial_size / 3))
                if close_size > 0 and close_size < self.position.size:
                    self.position.close(portion=0.33)
                    print(f"🎯 TP1 hit (1.5x ATR) at {high:.2f} | Scaling out 33% 🌙")
                # Move stop to breakeven
                self.stop_price = max(self.stop_price, self.data.Close[-1])

            if self.tp1_hit and not self.tp2_hit and high >= self.tp2_price:
                self.tp2_hit = True
                if self.position.size > 0:
                    self.position.close(portion=0.5)
                    print(f"🎯 TP2 hit (2.5x ATR) at {high:.2f} | Scaling out 50% 🚀")

            if self.tp2_hit and high >= self.tp3_price:
                print(f"🎯 TP3 hit (4x ATR) at {high:.2f} | Closing remaining ✨")
                self.position.close()
                self._reset_state()
                return

            # Trailing stop after TP1
            if self.tp1_hit:
                trail = max(self.ema20[-1], self.data.Close[-1] - self.atr[-1])
                self.stop_price = max(self.stop_price, trail)

            # Stop loss check
            if low <= self.stop_price:
                print(f"🛑 Stop hit at {self.stop_price:.2f} | Price: {low:.2f}")
                self.position.close()
                self._reset_state()
                return

            # RSI convergence exit - RSI makes higher high with price
            if len(self.rsi) > 3 and self.rsi[-1] > 70 and self.rsi[-1] > self.rsi[-2] and self.rsi[-3] > self.rsi[-2]:
                print(f"📉 RSI convergence exit at {price:.2f} | RSI: {self.rsi[-1]:.2f}")
                self.position.close()
                self._reset_state()
                return

            return

        # Entry logic - no position
        if len(self.data) < max(self.ema_slow, self.swing_lookback + 5, 30):
            return

        # Max 2 concurrent positions check (we only trade 1 at a time in backtesting.py)
        # RSI overbought filter
        if self.rsi[-1] > self.rsi_overbought:
            return

        # Trend filter: price above rising EMA50
        if not (price > self.ema50[-1] and self.ema50[-1] > self.ema50[-2]):
            return

        # Detect Higher High in price
        lookback = self.swing_lookback
        recent_high = self.data.High[-1]
        prior_high_idx = -lookback
        prior_high = self.data.High[prior_high_idx]

        # Price makes higher high
        if recent_high <= prior_high:
            return

        # Find the prior swing high bar (highest in the lookback window before recent)
        window_highs = self.data.High[-lookback:-1]
        if len(window_highs) < 2:
            return
        prior_swing_idx = np.argmax(window_highs) + (-lookback)
        prior_swing_high = self.data.High[prior_swing_idx]

        # Confirm recent bar is a new high vs prior swing
        if recent_high <= prior_swing_high:
            return

        # RSI divergence: RSI at recent high is lower than RSI at prior swing high
        rsi_recent = self.rsi[-1]
        rsi_prior = self.rsi[prior_swing_idx]
        if rsi_recent >= rsi_prior:
            return

        # Volume filter: volume on recent high < volume on prior swing high
        vol_recent = self.data.Volume[-1]
        vol_prior = self.data.Volume[prior_swing_idx]
        if vol_recent >= vol_prior:
            return

        # Volume should not expand sharply (contradicts exhaustion)
        if vol_recent > self.vol_sma[-1] * 1.5:
            return

        # Entry trigger: bullish confirmation candle - close above prior candle high
        if not (self.data.Close[-1] > self.data.High[-2]):
            return

        # Calculate position size based on risk
        atr_val = self.atr[-1]
        stop_price = price - self.sl_mult * atr_val
        risk_per_unit = price - stop_price

        if risk_per_unit <= 0:
            return

        equity = self.equity
        risk_amount = equity * self.risk_pct
        position_size = int(round(risk_amount / risk_per_unit))

        if position_size <= 0:
            return

        # Cap position size to available equity
        max_size = int(equity / price)
        position_size = min(position_size, max_size)

        if position_size <= 0:
            return

        print(f"🌙✨ EXHAUSTION DIVERGENCE SIGNAL! ✨🌙")
        print(f"   Price: {price:.2f} | RSI: {rsi_recent:.2f} (prior: {rsi_prior:.2f})")
        print(f"   Volume: {vol_recent:.2f} (prior: {vol_prior:.2f})")
        print(f"   ATR: {atr_val:.2f} | Stop: {stop_price:.2f}")
        print(f"   Size: {position_size} | Risk: ${risk_amount:.2f}")

        self.buy(size=position_size)

        self.entry_bar = len(self.data)
        self.stop_price = stop_price
        self.tp1_price = price + self.tp1_mult * atr_val
        self.tp2_price = price + self.tp2_mult * atr_val
        self.tp3_price = price + self.tp3_mult * atr_val
        self.tp1_hit = False
        self.tp2_hit = False
        self.initial_size = position_size

    def _reset_state(self):
        self.entry_bar = None
        self.stop_price = None
        self.tp1_hit = False
        self.tp2_hit = False
        self.initial_size = 0


bt = Backtest(
    data,
    ExhaustionDivergence,
    cash=1_000_000,
    commission=0.002,
    exclusive_orders=True
)

stats = bt.run()
print(stats)
print(stats._strategy)