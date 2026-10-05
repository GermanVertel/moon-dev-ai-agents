import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# 🌙 Moon Dev's HarmonicFiltration Backtest 🌙
print("🌙✨ Initializing Moon Dev's HarmonicFiltration Strategy...")
print("🚀 Loading BTC-USD 15m data...")

data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper case mapping
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

print(f"🌙 Data loaded: {len(data)} bars from {data.index[0]} to {data.index[-1]}")


class HarmonicFiltration(Strategy):
    """
    🌙 HarmonicFiltration Strategy 🌙
    Trades false breakouts at harmonic PRZ zones.
    """
    # Parameters
    swing_period = 5          # minimum 5-bar pivots
    atr_period = 14
    rsi_period = 14
    risk_pct = 0.01           # 1% risk per trade
    sl_atr_mult = 1.5
    prz_atr_tol = 0.5         # PRZ zone ±0.5 ATR
    wick_ratio = 0.6          # long rejection wick >60%
    max_positions = 2

    def init(self):
        print("🌙✨ Calculating indicators via TA-Lib...")
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period)
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=self.rsi_period)
        self.macd, self.macd_sig, self.macd_hist = self.I(
            talib.MACD, self.data.Close, 12, 26, 9
        )
        # Swing highs/lows using rolling max/min via talib
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=self.swing_period * 2)
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=self.swing_period * 2)
        # Volume SMA for breakout volume check
        self.vol_sma = self.I(talib.SMA, self.data.Volume, timeperiod=20)
        # Range boundary (prior N-bar high/low)
        self.range_high = self.I(talib.MAX, self.data.High, timeperiod=20)
        self.range_low = self.I(talib.MIN, self.data.Low, timeperiod=20)
        print("🚀 Indicators ready!")

    def _detect_harmonic_prz(self, i, direction):
        """
        Simplified harmonic PRZ detection using swing pivots.
        Look at last 5 pivots to find XABCD structure and check D retracement ratios.
        Returns (prz_price, prz_valid) or (None, False).
        """
        if i < 60:
            return None, False

        # Gather recent swing highs/lows as candidate pivots
        window = 40
        highs = self.data.High[i - window:i]
        lows = self.data.Low[i - window:i]

        # Simple pivot detection
        pivot_highs = []
        pivot_lows = []
        for k in range(2, len(highs) - 2):
            if highs[k] == max(highs[k-2:k+3]):
                pivot_highs.append((i - window + k, highs[k]))
            if lows[k] == min(lows[k-2:k+3]):
                pivot_lows.append((i - window + k, lows[k]))

        if direction == 'long':
            # For long: D should be a low (bullish PRZ)
            if len(pivot_lows) < 2 or len(pivot_highs) < 3:
                return None, False
            # Take last 5 alternating pivots (X,A,B,C,D)
            pivots = sorted(pivot_highs + pivot_lows, key=lambda x: x[0])
            if len(pivots) < 5:
                return None, False
            X, A, B, C, D = pivots[-5], pivots[-4], pivots[-3], pivots[-2], pivots[-1]
            # Ensure alternating
            # XA leg
            xa = abs(A[1] - X[1])
            if xa == 0:
                return None, False
            # AB retracement
            ab_ret = abs(B[1] - A[1]) / xa
            # AD retracement (D relative to XA)
            ad_ret = abs(D[1] - A[1]) / xa
            # Validate harmony ratios (loose tolerance)
            # Gartley/Bat/Butterfly/Crab/Shark D ranges
            valid_d = (0.70 <= ad_ret <= 1.75)
            valid_ab = (0.30 <= ab_ret <= 0.85)
            if valid_d and valid_ab:
                prz = D[1]
                return prz, True
        else:
            # For short: D should be a high (bearish PRZ)
            if len(pivot_highs) < 2 or len(pivot_lows) < 3:
                return None, False
            pivots = sorted(pivot_highs + pivot_lows, key=lambda x: x[0])
            if len(pivots) < 5:
                return None, False
            X, A, B, C, D = pivots[-5], pivots[-4], pivots[-3], pivots[-2], pivots[-1]
            xa = abs(A[1] - X[1])
            if xa == 0:
                return None, False
            ab_ret = abs(B[1] - A[1]) / xa
            ad_ret = abs(D[1] - A[1]) / xa
            valid_d = (0.70 <= ad_ret <= 1.75)
            valid_ab = (0.30 <= ab_ret <= 0.85)
            if valid_d and valid_ab:
                prz = D[1]
                return prz, True

        return None, False

    def _check_false_breakout(self, i, direction):
        """
        Check if bar i-1 was a false breakout and bar i closes back inside.
        """
        if i < 3:
            return False, None

        prev_high = self.range_high[i - 1]
        prev_low = self.range_low[i - 1]

        if direction == 'short':
            # Price broke above prior range high then closed back inside
            if self.data.High[i - 1] > prev_high and self.data.Close[i - 1] < prev_high:
                return True, self.data.High[i - 1]
            if self.data.High[i] > prev_high and self.data.Close[i] < prev_high:
                return True, self.data.High[i]
        else:
            if self.data.Low[i - 1] < prev_low and self.data.Close[i - 1] > prev_low:
                return True, self.data.Low[i - 1]
            if self.data.Low[i] < prev_low and self.data.Close[i] > prev_low:
                return True, self.data.Low[i]

        return False, None

    def _rejection_wick(self, i, direction):
        """Check for long rejection wick."""
        o, h, l, c = self.data.Open[i], self.data.High[i], self.data.Low[i], self.data.Close[i]
        rng = h - l
        if rng <= 0:
            return False
        if direction == 'short':
            upper_wick = h - max(o, c)
            return (upper_wick / rng) > self.wick_ratio
        else:
            lower_wick = min(o, c) - l
            return (lower_wick / rng) > self.wick_ratio

    def _divergence(self, i, direction):
        """RSI/MACD divergence at breakout extreme."""
        if i < 5:
            return False
        if direction == 'short':
            # Price higher high but RSI lower high
            if self.data.High[i] > self.data.High[i - 3] and self.rsi[i] < self.rsi[i - 3]:
                return True
        else:
            if self.data.Low[i] < self.data.Low[i - 3] and self.rsi[i] > self.rsi[i - 3]:
                return True
        return False

    def next(self):
        i = len(self.data) - 1
        if i < 60:
            return

        price = self.data.Close[i]
        atr = self.atr[i]
        if atr is None or np.isnan(atr) or atr <= 0:
            return

        # Skip if max positions reached
        if len(self.trades) >= self.max_positions:
            return

        # ==== SHORT SETUP ====
        if not self.position:
            fb_short, extreme_high = self._check_false_breakout(i, 'short')
            if fb_short and extreme_high:
                prz, prz_valid = self._detect_harmonic_prz(i, 'short')
                wick_ok = self._rejection_wick(i, 'short')
                div_ok = self._divergence(i, 'short')
                vol_ok = self.data.Volume[i] < self.vol_sma[i]  # declining volume
                # PRZ alignment
                prz_align = False
                if prz_valid and prz is not None:
                    prz_align = abs(extreme_high - prz) <= self.prz_atr_tol * atr

                if prz_align and wick_ok and vol_ok:
                    # Confluence score
                    score = 2 + int(div_ok) + int(vol_ok) + int(wick_ok)
                    if score >= 3:
                        sl = extreme_high + self.sl_atr_mult * atr
                        risk = sl - price
                        if risk <= 0:
                            return
                        risk_amount = self.equity * self.risk_pct
                        size = int(round(risk_amount / risk))
                        if size > 0:
                            print(f"🌙🔻 SHORT SIGNAL @ {price:.2f} | PRZ={prz:.2f} | SL={sl:.2f} | Size={size} | Div={div_ok} Wick={wick_ok} Vol={vol_ok}")
                            self.sell(size=size, sl=sl, tp=price - 2 * risk)

            # ==== LONG SETUP ====
            fb_long, extreme_low = self._check_false_breakout(i, 'long')
            if fb_long and extreme_low:
                prz, prz_valid = self._detect_harmonic_prz(i, 'long')
                wick_ok = self._rejection_wick(i, 'long')
                div_ok = self._divergence(i, 'long')
                vol_ok = self.data.Volume[i] < self.vol_sma[i]
                prz_align = False
                if prz_valid and prz is not None:
                    prz_align = abs(extreme_low - prz) <= self.prz_atr_tol * atr

                if prz_align and wick_ok and vol_ok:
                    score = 2 + int(div_ok) + int(vol_ok) + int(wick_ok)
                    if score >= 3:
                        sl = extreme_low - self.sl_atr_mult * atr
                        risk = price - sl
                        if risk <= 0:
                            return
                        risk_amount = self.equity * self.risk_pct
                        size = int(round(risk_amount / risk))
                        if size > 0:
                            print(f"🌙🟢 LONG SIGNAL @ {price:.2f} | PRZ={prz:.2f} | SL={sl:.2f} | Size={size} | Div={div_ok} Wick={wick_ok} Vol={vol_ok}")
                            self.buy(size=size, sl=sl, tp=price + 2 * risk)


print("🌙✨ Starting backtest...")
bt = Backtest(data, HarmonicFiltration, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙✨ Backtest complete! Moon Dev out! 🚀")