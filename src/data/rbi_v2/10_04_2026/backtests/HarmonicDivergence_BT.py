import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# ─────────────────────────────────────────────────────────────
# 🌙 Moon Dev's HarmonicDivergence Backtest 🌙
# ─────────────────────────────────────────────────────────────

data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"

print("🌙✨ Loading data from the Moon Dev vault...")
data = pd.read_csv(data_path)

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper column mapping
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')

print(f"🚀 Data loaded: {len(data)} rows from {data.index[0]} to {data.index[-1]}")


class HarmonicDivergence(Strategy):
    rsi_period = 3
    sma_period = 6
    atr_period = 14
    atr_avg_period = 10
    swing_lookback = 5
    risk_pct = 0.01
    atr_stop_mult = 1.5
    atr_trail_mult = 1.5
    rr_target = 2.0

    def init(self):
        print("🌙 Initializing Moon Dev indicators...")
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=self.rsi_period)
        self.sma = self.I(talib.SMA, self.data.Close, timeperiod=self.sma_period)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close,
                          timeperiod=self.atr_period)
        self.atr_avg = self.I(talib.SMA, self.atr, timeperiod=self.atr_avg_period)
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=self.swing_lookback)
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=self.swing_lookback)
        print("✨ Indicators ready, let's ride the moon! 🚀")

    def _bullish_divergence(self, i):
        """Price lower low + RSI higher low within last 3-5 bars."""
        if i < self.swing_lookback + 2:
            return False
        # Look back for a prior swing low
        for j in range(2, self.swing_lookback + 1):
            idx = i - j
            if idx < 1:
                continue
            # Current bar makes a lower low vs prior swing low
            prior_low = self.data.Low[idx]
            curr_low = self.data.Low[i]
            prior_rsi = self.rsi[idx]
            curr_rsi = self.rsi[i]
            if (curr_low < prior_low) and (curr_rsi > prior_rsi) and not np.isnan(prior_rsi):
                # RSI HL must be recent (within lookback)
                return True
        return False

    def _bearish_divergence(self, i):
        """Price higher high + RSI lower high at a swing high."""
        if i < self.swing_lookback + 2:
            return False
        for j in range(2, self.swing_lookback + 1):
            idx = i - j
            if idx < 1:
                continue
            prior_high = self.data.High[idx]
            curr_high = self.data.High[i]
            prior_rsi = self.rsi[idx]
            curr_rsi = self.rsi[i]
            if (curr_high > prior_high) and (curr_rsi < prior_rsi) and not np.isnan(prior_rsi):
                return True
        return False

    def next(self):
        i = len(self.data) - 1
        price = self.data.Close[-1]

        if np.isnan(self.rsi[-1]) or np.isnan(self.sma[-1]) or np.isnan(self.atr[-1]):
            return

        # ─── Position management ───────────────────────────────
        if self.position:
            entry = self.position.entry_price
            # Trail stop at 1.5 × ATR below highest close since entry
            highest_close = max(self.data.Close[-self.position.size * 0 + 1:]) if False else price
            # Simpler: use current high tracked manually
            if not hasattr(self, 'peak'):
                self.peak = price
            self.peak = max(self.peak, price)

            trail_stop = self.peak - self.atr_trail_mult * self.atr[-1]

            # Primary exit: bearish divergence
            if self._bearish_divergence(i):
                print(f"🌙 Bearish divergence detected — exiting long at {price:.2f}")
                self.position.close()
                self.peak = None
                return

            # RSI overbought then crosses below 50
            if len(self.rsi) > 2 and self.rsi[-2] > 70 and self.rsi[-1] < 50:
                print(f"🌙 RSI momentum fade — exiting long at {price:.2f}")
                self.position.close()
                self.peak = None
                return

            # Volatility spike: ATR > 1.5× avg
            if not np.isnan(self.atr_avg[-1]) and self.atr[-1] > 1.5 * self.atr_avg[-1]:
                print(f"🌙 ATR volatility spike — exiting long at {price:.2f}")
                self.position.close()
                self.peak = None
                return

            # Candle range > 2× ATR
            candle_range = self.data.High[-1] - self.data.Low[-1]
            if candle_range > 2 * self.atr[-1]:
                print(f"🌙 Candle range spike — exiting long at {price:.2f}")
                self.position.close()
                self.peak = None
                return

            # Trailing stop hit
            if price < trail_stop:
                print(f"🌙 Trailing stop hit — exiting long at {price:.2f}")
                self.position.close()
                self.peak = None
                return
            return

        # ─── Entry logic ───────────────────────────────────────
        # Step 1: bullish divergence
        if not self._bullish_divergence(i):
            return

        # Step 2: trend filter — close above 6-period SMA
        if price <= self.sma[-1]:
            return

        # Step 3: trigger — enter long
        swing_low = self.swing_low[-1]
        stop_price = min(swing_low, price - self.atr_stop_mult * self.atr[-1])
        risk_per_unit = price - stop_price
        if risk_per_unit <= 0:
            return

        equity = self.equity
        risk_amount = equity * self.risk_pct
        position_size = risk_amount / risk_per_unit
        position_size = int(round(position_size))
        if position_size < 1:
            position_size = 1

        # Cap by equity
        max_size = int(equity / price)
        if position_size > max_size:
            position_size = max_size

        take_profit = price + self.rr_target * risk_per_unit

        print(f"🚀🌙 BULLISH DIVERGENCE! Entering LONG at {price:.2f} | "
              f"size={position_size} | SL={stop_price:.2f} | TP={take_profit:.2f}")

        self.buy(size=position_size, sl=stop_price, tp=take_profit)
        self.peak = price


bt = Backtest(data, HarmonicDivergence, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)