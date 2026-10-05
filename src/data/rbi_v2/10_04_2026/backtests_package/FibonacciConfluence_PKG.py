import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# 🌙 Moon Dev Data Loading & Cleaning
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'datetime': 'Date',
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})
data['Date'] = pd.to_datetime(data['Date'])
data = data.set_index('Date')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
print("🌙✨ Data loaded successfully! Rows:", len(data), "🚀")


class FibonacciConfluence(Strategy):
    sma_period = 20
    rsi_period = 14
    atr_period = 14
    pivot_window = 5
    fib_tolerance_atr = 0.25
    risk_pct = 0.01
    stop_atr_mult = 1.5
    trail_atr_mult = 2.0
    rr_target = 2.0

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low

        self.sma = self.I(talib.SMA, close, timeperiod=self.sma_period)
        self.rsi = self.I(talib.RSI, close, timeperiod=self.rsi_period)
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        # 🌙 ATR moving average — compute via pandas to avoid nesting Indicator arrays
        atr_series = pd.Series(self.atr)
        atr_ma_vals = atr_series.rolling(20).mean().to_numpy()
        self.atr_ma = self.I(lambda: atr_ma_vals)

        # Fractal pivots (5-bar rolling extremes)
        self.pivot_high = self.I(talib.MAX, high, timeperiod=self.pivot_window)
        self.pivot_low = self.I(talib.MIN, low, timeperiod=self.pivot_window)

        # Track state
        self.swing_high = np.nan
        self.swing_low = np.nan
        self.fib_618 = np.nan
        self.entry_price = None
        self.stop_price = None
        self.tp_price = None
        self.rsi_peak_price = None
        self.rsi_peak_value = None
        self.rsi_peak_idx = None
        self.bars_since_cross = 0
        self.entry_bar = None

    def _update_swings(self, i):
        # Detect fresh fractal pivot at i-2 (confirmed)
        idx = i - 2
        if idx < self.pivot_window:
            return
        h = self.data.High[idx]
        l = self.data.Low[idx]
        if h == self.pivot_high[idx]:
            self.swing_high = h
        if l == self.pivot_low[idx]:
            self.swing_low = l
        if not np.isnan(self.swing_high) and not np.isnan(self.swing_low) and self.swing_high > self.swing_low:
            self.fib_618 = self.swing_high - 0.618 * (self.swing_high - self.swing_low)

    def _bearish_divergence(self, i):
        # Track RSI peaks; detect higher high in price with lower high in RSI
        if i < 3:
            return False
        # Look for local RSI peak at i-1
        if self.rsi[i-1] > self.rsi[i-2] and self.rsi[i-1] > self.rsi[i]:
            price_peak = self.data.High[i-1]
            rsi_peak = self.rsi[i-1]
            if self.rsi_peak_price is not None and self.rsi_peak_idx is not None:
                # need 3+ bar separation
                if (i-1) - self.rsi_peak_idx >= 3:
                    if price_peak > self.rsi_peak_price and rsi_peak < self.rsi_peak_value:
                        return True
            self.rsi_peak_price = price_peak
            self.rsi_peak_value = rsi_peak
            self.rsi_peak_idx = i-1
        return False

    def _bearish_reversal_candle(self, i):
        if i < 1:
            return False
        o, c = self.data.Open[i], self.data.Close[i]
        po, pc = self.data.Open[i-1], self.data.Close[i-1]
        # bearish engulfing
        if c < o and pc > po and c < po and o > pc:
            return True
        # close below prior low
        if c < self.data.Low[i-1]:
            return True
        # shooting star
        h, l = self.data.High[i], self.data.Low[i]
        rng = h - l
        if rng > 0 and (h - max(o, c)) > 0.6 * rng and (min(o, c) - l) < 0.2 * rng:
            return True
        return False

    def next(self):
        i = len(self.data) - 1
        if i < max(self.sma_period, self.rsi_period, self.atr_period, self.pivot_window) + 5:
            return

        self._update_swings(i)

        price = self.data.Close[i]
        low = self.data.Low[i]
        sma = self.sma[i]
        rsi = self.rsi[i]
        atr = self.atr[i]
        atr_ma = self.atr_ma[i]

        if np.isnan(sma) or np.isnan(rsi) or np.isnan(atr) or np.isnan(atr_ma):
            return

        # ============ MANAGE OPEN POSITION ============
        if self.position:
            # Trail stop
            new_stop = price - self.trail_atr_mult * atr
            if new_stop > self.stop_price:
                self.stop_price = new_stop
            # Move to breakeven after +1 ATR
            if price >= self.entry_price + atr and self.stop_price < self.entry_price:
                self.stop_price = self.entry_price

            # Hard invalidation: close below fib or SMA
            invalid = False
            if not np.isnan(self.fib_618) and price < self.fib_618:
                invalid = True
            if price < sma:
                invalid = True

            # Bearish RSI divergence exit
            div_exit = self._bearish_divergence(i) and self._bearish_reversal_candle(i)

            if invalid:
                print(f"🌙💥 HARD INVALIDATION EXIT at {price:.2f}")
                self.position.close()
                self._reset()
                return
            if div_exit:
                print(f"🌙🔻 BEARISH RSI DIVERGENCE EXIT at {price:.2f}")
                self.position.close()
                self._reset()
                return
            if price <= self.stop_price:
                print(f"🌙🛑 STOP HIT at {price:.2f} (stop={self.stop_price:.2f})")
                self.position.close()
                self._reset()
                return
            if price >= self.tp_price:
                print(f"🌙🎯 TAKE PROFIT at {price:.2f}")
                self.position.close()
                self._reset()
                return
            return

        # ============ ENTRY LOGIC ============
        # Skip low volatility
        if atr < atr_ma:
            return
        if np.isnan(self.fib_618):
            return
        # No trade zone: fib and SMA too far apart
        if abs(self.fib_618 - sma) > atr:
            return

        prev_price = self.data.Close[i-1]
        prev_sma = self.sma[i-1]

        # Momentum cross: fresh cross above SMA
        fresh_cross = prev_price <= prev_sma and price > sma
        reclaim = price > sma and prev_price <= prev_sma

        if fresh_cross or reclaim:
            self.bars_since_cross = 0
        else:
            self.bars_since_cross += 1

        # Confluence touch within 1-3 bars of cross
        if self.bars_since_cross > 3:
            return

        tolerance = self.fib_tolerance_atr * atr
        touched_fib = abs(low - self.fib_618) <= tolerance or (low <= self.fib_618 <= self.data.High[i])

        # RSI context
        rsi_ok = rsi > 40

        # Impulse bullish
        bullish_impulse = self.swing_high > self.swing_low

        if (fresh_cross or reclaim) and touched_fib and rsi_ok and bullish_impulse:
            entry = price
            stop = entry - self.stop_atr_mult * atr
            risk = entry - stop
            if risk <= 0:
                return
            tp = entry + self.rr_target * risk

            # Position sizing: risk 1% of equity
            equity = self.equity
            risk_amount = equity * self.risk_pct
            size = int(round(risk_amount / risk))
            if size < 1:
                size = 1

            self.entry_price = entry
            self.stop_price = stop
            self.tp_price = tp
            self.entry_bar = i
            self.rsi_peak_price = None
            self.rsi_peak_value = None
            self.rsi_peak_idx = None

            print(f"🌙🚀 LONG ENTRY @ {entry:.2f} | SMA={sma:.2f} RSI={rsi:.1f} ATR={atr:.2f} | Stop={stop:.2f} TP={tp:.2f} Size={size}")
            self.buy(size=size)

    def _reset(self):
        self.entry_price = None
        self.stop_price = None
        self.tp_price = None
        self.rsi_peak_price = None
        self.rsi_peak_value = None
        self.rsi_peak_idx = None
        self.bars_since_cross = 0


bt = Backtest(data, FibonacciConfluence, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)