import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# 🌙 Moon Dev's VolatilityConvergence Backtest 🌙

def kst(close, r1=10, r2=15, r3=20, r4=30, s1=10, s2=10, s3=10, s4=15):
    """Know Sure Thing oscillator"""
    close = pd.Series(close)
    roc1 = (close / close.shift(r1) - 1) * 100
    roc2 = (close / close.shift(r2) - 1) * 100
    roc3 = (close / close.shift(r3) - 1) * 100
    roc4 = (close / close.shift(r4) - 1) * 100
    k = (roc1.rolling(s1).mean() * 1 +
         roc2.rolling(s2).mean() * 2 +
         roc3.rolling(s3).mean() * 3 +
         roc4.rolling(s4).mean() * 4)
    return k.values


class VolatilityConvergence(Strategy):
    # Strategy parameters
    n_high = 20
    proximity_atr = 1.0        # within X * ATR of rolling high
    atr_period = 14
    atr_osc_lookback = 100
    kst_signal_period = 9
    stop_atr_mult = 2.0
    exit_atr_mult = 1.5
    risk_pct = 0.01
    max_hold = 20

    def init(self):
        print("🌙✨ Moon Dev: Initializing VolatilityConvergence Strategy ✨🌙")
        close = self.data.Close
        high = self.data.High
        low = self.data.Low

        # Rolling High
        self.rolling_high = self.I(talib.MAX, high, timeperiod=self.n_high)

        # ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        # ATR%
        self.atr_pct = self.I(lambda a, c: a / c, self.atr, close)

        # ATR oscillator: z-score of ATR% over lookback
        def atr_zscore(arr):
            s = pd.Series(arr)
            mean = s.rolling(self.atr_osc_lookback).mean()
            std = s.rolling(self.atr_osc_lookback).std()
            return ((s - mean) / std).values

        self.atr_osc = self.I(atr_zscore, self.atr_pct)

        # KST
        self.kst_line = self.I(kst, close)
        self.kst_signal = self.I(talib.SMA, self.kst_line, timeperiod=self.kst_signal_period)

        # KST z-score for scale comparison
        def kst_zscore(arr):
            s = pd.Series(arr)
            mean = s.rolling(self.atr_osc_lookback).mean()
            std = s.rolling(self.atr_osc_lookback).std()
            return ((s - mean) / std).values

        self.kst_z = self.I(kst_zscore, self.kst_line)

        print("🌙 Moon Dev: Indicators ready — ATR, ATR%, ATR-Osc, KST, KST-Signal 🚀")

    def next(self):
        price = self.data.Close[-1]
        rh = self.rolling_high[-1]
        atr = self.atr[-1]
        atr_osc = self.atr_osc[-1]
        kst_val = self.kst_z[-1]
        kst_line = self.kst_line[-1]
        kst_sig = self.kst_signal[-1]

        # Skip if NaNs
        if np.isnan(rh) or np.isnan(atr) or np.isnan(atr_osc) or np.isnan(kst_val) or np.isnan(kst_line) or np.isnan(kst_sig):
            return

        if self.position:
            # Track entry info
            entry_price = self.trades[-1].entry_price
            entry_atr = self.trades[-1].atr_entry
            highest_close = max(self.trades[-1].highest_close, price)

            # Update highest close for trailing
            self.trades[-1].highest_close = highest_close

            # Primary exit: volatility reversion
            if atr >= self.exit_atr_mult * entry_atr:
                print(f"🌙 Moon Dev EXIT: Volatility reversion! ATR {atr:.2f} >= {self.exit_atr_mult}*{entry_atr:.2f} 💰")
                self.position.close()
                return

            # Trailing stop
            trail_stop = highest_close - self.stop_atr_mult * atr
            if price <= trail_stop:
                print(f"🌙 Moon Dev EXIT: Trailing stop hit at {price:.2f} (trail {trail_stop:.2f}) 🛑")
                self.position.close()
                return

            # Momentum exit
            if kst_line < kst_sig:
                print(f"🌙 Moon Dev EXIT: KST crossed below signal ({kst_line:.2f} < {kst_sig:.2f}) 📉")
                self.position.close()
                return

            # Time stop
            bars_held = len(self.data) - self.trades[-1].entry_bar
            if bars_held >= self.max_hold:
                print(f"🌙 Moon Dev EXIT: Time stop after {bars_held} bars ⏰")
                self.position.close()
                return

        else:
            # Entry conditions
            # A: price near rolling high
            cond_a = price >= (rh - self.proximity_atr * atr)
            # B: ATR oscillator < KST z-score (volatility compressed below momentum)
            cond_b = atr_osc < kst_val
            # C: KST > signal (positive momentum confirmation)
            cond_c = kst_line > kst_sig

            # Skip if ATR oscillator already in top decile (news spike filter)
            # Approximate: atr_osc > 1.5 (z-score)
            news_filter = atr_osc > 1.5

            if cond_a and cond_b and cond_c and not news_filter:
                # Risk-based sizing
                stop_price = price - self.stop_atr_mult * atr
                risk_per_unit = price - stop_price
                if risk_per_unit <= 0:
                    return
                risk_amount = self.equity * self.risk_pct
                size = risk_amount / risk_per_unit
                size = int(round(size))
                if size < 1:
                    return

                print(f"🌙 Moon Dev ENTRY: Price {price:.2f} near high {rh:.2f} | ATR-Osc {atr_osc:.2f} < KST {kst_val:.2f} | KST>{kst_sig:.2f} 🚀")
                self.buy(size=size)
                # Store entry metadata on the trade object AFTER it's created
                # Use a try/except in case the trade hasn't been registered yet
                if len(self.trades) > 0:
                    self.trades[-1].atr_entry = atr
                    self.trades[-1].highest_close = price
                    self.trades[-1].entry_bar = len(self.data)


# Load data
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
data['Datetime'] = pd.to_datetime(data['Datetime'])
data = data.set_index('Datetime')

print("🌙 Moon Dev: Data loaded and cleaned! Shape:", data.shape, "✨")

bt = Backtest(data, VolatilityConvergence, cash=1_000_000, commission=0.001)

stats = bt.run()
print(stats)
print(stats._strategy)