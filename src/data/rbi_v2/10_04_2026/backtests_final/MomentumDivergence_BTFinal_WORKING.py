import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

print("🌙 Moon Dev's MomentumDivergence Backtest Loading... ✨🚀")

data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper column mapping
data = data.rename(columns={
    'datetime': 'Datetime',
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

if 'Datetime' in data.columns:
    data['Datetime'] = pd.to_datetime(data['Datetime'])
    data = data.set_index('Datetime')

data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
print(f"🌙 Data loaded: {len(data)} bars ✨")


class MomentumDivergence(Strategy):
    # Parameters
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    rsi_period = 14
    atr_period = 14
    vol_period = 20
    swing_lookback = 5
    risk_pct = 0.02
    rr_min = 1.5

    def init(self):
        print("🌙 Initializing Moon Dev indicators... ✨")
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        # MACD
        self.macd, self.macd_signal, self.macd_hist = self.I(
            talib.MACD, close,
            fastperiod=self.macd_fast,
            slowperiod=self.macd_slow,
            signalperiod=self.macd_signal
        )

        # RSI
        self.rsi = self.I(talib.RSI, close, timeperiod=self.rsi_period)

        # ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        # Swing highs (rolling max)
        self.swing_high = self.I(talib.MAX, high, timeperiod=self.swing_lookback)

        # Volume % change average
        vol_series = pd.Series(volume)
        vol_pct = vol_series.pct_change().abs() * 100
        vol_avg = vol_pct.rolling(self.vol_period).mean().fillna(0).values
        self.vol_avg = self.I(lambda x: x, vol_avg)

        # Track state
        self.entry_price = None
        self.tp_price = None
        self.sl_price = None
        self.swing_high_at_entry = None
        print("🌙 Indicators ready! 🚀")

    def next(self):
        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]

        # Manage open position
        if self.position:
            # Take profit
            if self.tp_price is not None and low <= self.tp_price:
                print(f"🌙✨ TP HIT at {self.tp_price:.2f} | Price: {price:.2f} 🚀")
                self.position.close()
                self.entry_price = None
                self.tp_price = None
                self.sl_price = None
                return

            # Stop loss
            if self.sl_price is not None and high >= self.sl_price:
                print(f"🌙💥 SL HIT at {self.sl_price:.2f} | Price: {price:.2f}")
                self.position.close()
                self.entry_price = None
                self.tp_price = None
                self.sl_price = None
                return

            # Invalidation: price breaks above swing high
            if self.swing_high_at_entry is not None and high > self.swing_high_at_entry:
                print(f"🌙⚠️ INVALIDATION - Price broke swing high {self.swing_high_at_entry:.2f}")
                self.position.close()
                self.entry_price = None
                self.tp_price = None
                self.sl_price = None
                return

            return

        # Need enough data
        if len(self.data) < max(self.macd_slow + self.macd_signal, self.vol_period, self.swing_lookback) + 10:
            return

        # Look for swing high with divergence
        lb = self.swing_lookback
        idx = len(self.data) - 1

        # Current bar should be a local swing high (lookback check)
        # Check if a swing high formed lb bars ago
        pivot_idx = idx - lb
        if pivot_idx < lb + 1:
            return

        pivot_high = self.data.High[pivot_idx]
        # Is it a swing high? Compare to surrounding bars
        window_highs = self.data.High[pivot_idx - lb:pivot_idx + lb + 1]
        if pivot_high != max(window_highs):
            return

        # Find previous swing high (before pivot)
        prev_pivot_idx = None
        for j in range(pivot_idx - lb - 1, lb, -1):
            if j - lb < 0:
                break
            wh = self.data.High[j - lb:j + lb + 1]
            if self.data.High[j] == max(wh):
                prev_pivot_idx = j
                break

        if prev_pivot_idx is None:
            return

        prev_high = self.data.High[prev_pivot_idx]

        # Bearish divergence: price higher high
        if pivot_high <= prev_high:
            return

        # MACD lower high at pivot
        macd_pivot = self.macd[pivot_idx]
        macd_prev = self.macd[prev_pivot_idx]
        if macd_pivot >= macd_prev:
            return

        # RSI lower high at pivot
        rsi_pivot = self.rsi[pivot_idx]
        rsi_prev = self.rsi[prev_pivot_idx]
        if rsi_pivot >= rsi_prev:
            return

        # Both divergences confirmed!
        print(f"🌙🔍 DUAL DIVERGENCE detected! Price HH: {prev_high:.2f}->{pivot_high:.2f} | MACD: {macd_prev:.4f}->{macd_pivot:.4f} | RSI: {rsi_prev:.2f}->{rsi_pivot:.2f}")

        # Entry confirmation: reversal candle - bearish close
        # Break of prior candle low or bearish candle
        prev_close = self.data.Close[-2]
        prev_open = self.data.Open[-2]
        curr_close = self.data.Close[-1]
        curr_open = self.data.Open[-1]

        bearish_engulf = (curr_close < curr_open) and (curr_open >= prev_close) and (curr_close <= prev_open)
        break_prior_low = curr_close < self.data.Low[-2]
        bearish_candle = curr_close < curr_open

        if not (bearish_engulf or break_prior_low or bearish_candle):
            return

        # ATR filter
        atr_val = self.atr[-1]
        if atr_val <= 0 or np.isnan(atr_val):
            return

        # ATR unusually low/high filter
        atr_series = pd.Series(self.atr)
        atr_mean = atr_series.rolling(50).mean().iloc[-1]
        if not np.isnan(atr_mean) and atr_mean > 0:
            if atr_val < atr_mean * 0.3 or atr_val > atr_mean * 3.0:
                print(f"🌙⚠️ ATR filter skip: ATR={atr_val:.2f} mean={atr_mean:.2f}")
                return

        # Volume-based TP
        vol_avg_pct = self.vol_avg[-1]
        if np.isnan(vol_avg_pct) or vol_avg_pct <= 0:
            return

        entry = price
        tp_distance = entry * (2 * vol_avg_pct / 100)
        sl_distance = atr_val

        # RR check
        if sl_distance <= 0:
            return
        rr = tp_distance / sl_distance
        if rr < self.rr_min:
            print(f"🌙⚠️ RR too low: {rr:.2f} < {self.rr_min}")
            return

        tp_price = entry - tp_distance
        sl_price = entry + sl_distance

        # Position sizing: risk 2% of equity
        equity = self.equity
        risk_amount = equity * self.risk_pct
        position_size = risk_amount / sl_distance
        position_size = int(round(position_size))

        if position_size <= 0:
            return

        # Cap by equity
        max_size = int(equity / entry)
        position_size = min(position_size, max_size)
        if position_size <= 0:
            return

        self.entry_price = entry
        self.tp_price = tp_price
        self.sl_price = sl_price
        self.swing_high_at_entry = pivot_high

        print(f"🌙🚀 SHORT ENTRY! Price: {entry:.2f} | TP: {tp_price:.2f} | SL: {sl_price:.2f} | Size: {position_size} | RR: {rr:.2f} ✨")
        self.sell(size=position_size)


bt = Backtest(data, MomentumDivergence, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)