import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy
import os

# Load data
data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Map columns
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

print(f"🌙 Moon Dev Data Loaded: {len(data)} bars ✨")
print(f"🚀 Columns: {list(data.columns)}")


class HarmonicStochastic(Strategy):
    # Harmonic detection params
    swing_window = 5
    fib_tolerance = 0.08

    # Stochastic
    stoch_k = 14
    stoch_d = 3
    stoch_smooth = 3

    # Volume
    vol_ma_period = 20
    vol_mult = 1.5

    # ATR
    atr_period = 14
    atr_mult_sl = 1.0

    # Risk
    risk_pct = 0.01
    max_hold_bars = 60

    def init(self):
        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        volume = pd.Series(self.data.Volume)

        # Stochastic
        self.stoch_k_line, self.stoch_d_line = self.I(
            talib.STOCH, high, low, close,
            fastk_period=self.stoch_k,
            slowk_period=self.stoch_smooth,
            slowk_matype=0,
            slowd_period=self.stoch_d,
            slowd_matype=0
        )

        # Volume MA
        self.vol_ma = self.I(talib.SMA, volume, timeperiod=self.vol_ma_period)

        # ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        # Swing highs/lows
        self.swing_high = self.I(talib.MAX, high, timeperiod=self.swing_window)
        self.swing_low = self.I(talib.MIN, low, timeperiod=self.swing_window)

        # Trackers
        self.entry_price = None
        self.stop_price = None
        self.tp1 = None
        self.tp2 = None
        self.tp3 = None
        self.bars_held = 0
        self.tp1_hit = False
        self.tp2_hit = False
        self.trade_dir = 0

        print("🌙✨ HarmonicStochastic indicators initialized 🚀")

    def _find_swings(self, lookback=40):
        """Find recent swing highs/lows using talib MAX/MIN arrays."""
        i = len(self.data) - 1
        if i < lookback:
            return None

        highs = np.array(self.data.High[-lookback:])
        lows = np.array(self.data.Low[-lookback:])

        # Identify pivot points (local extrema)
        pivots_high = []
        pivots_low = []
        for j in range(2, lookback - 2):
            if highs[j] > highs[j-1] and highs[j] > highs[j-2] and highs[j] > highs[j+1] and highs[j] > highs[j+2]:
                pivots_high.append((j, highs[j]))
            if lows[j] < lows[j-1] and lows[j] < lows[j-2] and lows[j] < lows[j+1] and lows[j] < lows[j+2]:
                pivots_low.append((j, lows[j]))

        return pivots_high, pivots_low

    def _check_bullish_butterfly(self):
        """Detect bullish Butterfly: X-A-B-C-D with D below X (extension)."""
        result = self._find_swings(60)
        if result is None:
            return None
        pivots_high, pivots_low = result

        if len(pivots_high) < 2 or len(pivots_low) < 3:
            return None

        # Bullish: X low, A high, B low, C high, D low (D < X)
        try:
            X = pivots_low[-3]
            A = pivots_high[-2]
            B = pivots_low[-2]
            C = pivots_high[-1]
            D = pivots_low[-1]
        except IndexError:
            return None

        # Ensure ordering: X < A, B between, C < A, D < X (butterfly)
        if not (X[0] < A[0] < B[0] < C[0] < D[0]):
            return None

        xa = A[1] - X[1]
        ab = A[1] - B[1]
        bc = C[1] - B[1]
        cd = C[1] - D[1]

        if xa <= 0 or ab <= 0 or bc <= 0 or cd <= 0:
            return None

        # B retracement of XA: 0.382 - 0.886
        b_ratio = ab / xa
        if not (0.382 <= b_ratio <= 0.886):
            return None

        # C retracement of AB: 0.382 - 0.886
        c_ratio = bc / ab
        if not (0.382 <= c_ratio <= 0.886):
            return None

        # D extension of XA: 1.27 or 1.618 (Butterfly defining feature)
        d_ext = (A[1] - D[1]) / xa
        if not (1.15 <= d_ext <= 1.75):
            return None

        # D retracement of BC: 1.618 - 2.618
        d_bc = cd / bc
        if not (1.4 <= d_bc <= 2.8):
            return None

        return {
            'X': X[1], 'A': A[1], 'B': B[1], 'C': C[1], 'D': D[1],
            'type': 'bullish'
        }

    def _check_bearish_butterfly(self):
        """Detect bearish Butterfly: X-A-B-C-D with D above X."""
        result = self._find_swings(60)
        if result is None:
            return None
        pivots_high, pivots_low = result

        if len(pivots_low) < 2 or len(pivots_high) < 3:
            return None

        try:
            X = pivots_high[-3]
            A = pivots_low[-2]
            B = pivots_high[-2]
            C = pivots_low[-1]
            D = pivots_high[-1]
        except IndexError:
            return None

        if not (X[0] < A[0] < B[0] < C[0] < D[0]):
            return None

        xa = X[1] - A[1]
        ab = B[1] - A[1]
        bc = B[1] - C[1]
        cd = D[1] - C[1]

        if xa <= 0 or ab <= 0 or bc <= 0 or cd <= 0:
            return None

        b_ratio = ab / xa
        if not (0.382 <= b_ratio <= 0.886):
            return None

        c_ratio = bc / ab
        if not (0.382 <= c_ratio <= 0.886):
            return None

        d_ext = (D[1] - A[1]) / xa
        if not (1.15 <= d_ext <= 1.75):
            return None

        d_bc = cd / bc
        if not (1.4 <= d_bc <= 2.8):
            return None

        return {
            'X': X[1], 'A': A[1], 'B': B[1], 'C': C[1], 'D': D[1],
            'type': 'bearish'
        }

    def next(self):
        i = len(self.data) - 1
        if i < 60:
            return

        price = self.data.Close[-1]
        vol = self.data.Volume[-1]
        vol_ma = self.vol_ma[-1]
        k = self.stoch_k_line[-1]
        d = self.stoch_d_line[-1]
        k_prev = self.stoch_k_line[-2]
        d_prev = self.stoch_d_line[-2]
        atr = self.atr[-1]
        sh = self.swing_high[-1]
        sl = self.swing_low[-1]

        if np.isnan(atr) or atr <= 0 or np.isnan(vol_ma) or vol_ma <= 0:
            return

        # Manage open trade
        if self.position:
            self.bars_held += 1
            if self.trade_dir == 1:
                # Move stop to breakeven after TP1
                if not self.tp1_hit and self.data.High[-1] >= self.tp1:
                    self.tp1_hit = True
                    self.stop_price = max(self.stop_price, self.entry_price)
                    print(f"🌙 TP1 hit LONG! Stop → breakeven 🚀")
                if not self.tp2_hit and self.data.High[-1] >= self.tp2:
                    self.tp2_hit = True
                    self.stop_price = max(self.stop_price, self.data.Close[-1] - 1.5 * atr)
                    print(f"🌙 TP2 hit LONG! Trailing stop 1.5x ATR 🚀")
                if self.data.High[-1] >= self.tp3:
                    print(f"🌙✨ TP3 hit LONG! Closing full position 💰")
                    self.position.close()
                    self.trade_dir = 0
                    return
                if self.data.Low[-1] <= self.stop_price:
                    print(f"🌙 Stop hit LONG at {self.stop_price:.2f} 💀")
                    self.position.close()
                    self.trade_dir = 0
                    return
            elif self.trade_dir == -1:
                if not self.tp1_hit and self.data.Low[-1] <= self.tp1:
                    self.tp1_hit = True
                    self.stop_price = min(self.stop_price, self.entry_price)
                    print(f"🌙 TP1 hit SHORT! Stop → breakeven 🚀")
                if not self.tp2_hit and self.data.Low[-1] <= self.tp2:
                    self.tp2_hit = True
                    self.stop_price = min(self.stop_price, self.data.Close[-1] + 1.5 * atr)
                    print(f"🌙 TP2 hit SHORT! Trailing stop 1.5x ATR 🚀")
                if self.data.Low[-1] <= self.tp3:
                    print(f"🌙✨ TP3 hit SHORT! Closing full position 💰")
                    self.position.close()
                    self.trade_dir = 0
                    return
                if self.data.High[-1] >= self.stop_price:
                    print(f"🌙 Stop hit SHORT at {self.stop_price:.2f} 💀")
                    self.position.close()
                    self.trade_dir = 0
                    return

            # Time stop
            if self.bars_held >= self.max_hold_bars:
                print(f"🌙 Time stop triggered after {self.bars_held} bars ⏰")
                self.position.close()
                self.trade_dir = 0
            return

        # Look for entries
        vol_surge = vol > self.vol_mult * vol_ma

        # Bullish Butterfly + oversold stoch + bullish cross + breakout above swing high
        bull = self._check_bullish_butterfly()
        if bull is not None:
            stoch_ok = k < 30 and k_prev < d_prev and k > d
            breakout_ok = price > sh and vol_surge
            if stoch_ok and breakout_ok:
                entry = price
                stop = bull['D'] - self.atr_mult_sl * atr
                risk = entry - stop
                if risk <= 0:
                    return
                cd = bull['C'] - bull['D']
                tp1 = bull['D'] + 0.382 * cd
                tp2 = bull['D'] + 0.618 * cd
                tp3 = bull['A']
                size = 0.95
                self.buy(size=size)
                self.entry_price = entry
                self.stop_price = stop
                self.tp1 = tp1
                self.tp2 = tp2
                self.tp3 = tp3
                self.trade_dir = 1
                self.bars_held = 0
                self.tp1_hit = False
                self.tp2_hit = False
                print(f"🌙🚀 BULLISH BUTTERFLY LONG! Entry={entry:.2f} SL={stop:.2f} TP1={tp1:.2f} TP2={tp2:.2f} TP3={tp3:.2f} ✨")
                return

        # Bearish Butterfly + overbought stoch + bearish cross + breakout below swing low
        bear = self._check_bearish_butterfly()
        if bear is not None:
            stoch_ok = k > 70 and k_prev > d_prev and k < d
            breakout_ok = price < sl and vol_surge
            if stoch_ok and breakout_ok:
                entry = price
                stop = bear['D'] + self.atr_mult_sl * atr
                risk = stop - entry
                if risk <= 0:
                    return
                cd = bear['D'] - bear['C']
                tp1 = bear['D'] - 0.382 * cd
                tp2 = bear['D'] - 0.618 * cd
                tp3 = bear['A']
                size = 0.95
                self.sell(size=size)
                self.entry_price = entry
                self.stop_price = stop
                self.tp1 = tp1
                self.tp2 = tp2
                self.tp3 = tp3
                self.trade_dir = -1
                self.bars_held = 0
                self.tp1_hit = False
                self.tp2_hit = False
                print(f"🌙🚀 BEARISH BUTTERFLY SHORT! Entry={entry:.2f} SL={stop:.2f} TP1={tp1:.2f} TP2={tp2:.2f} TP3={tp3:.2f} ✨")
                return


bt = Backtest(data, HarmonicStochastic, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)