import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# 🌙 Moon Dev's DivergentPulse Backtest 🌙

print("🌙✨ Moon Dev's DivergentPulse Strategy Loading... 🚀")

data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')

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

data = data.set_index(pd.to_datetime(data['Datetime']))
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

print(f"🌙 Data loaded: {len(data)} bars ✨")


class DivergentPulse(Strategy):
    # Parameters
    rsi_period = 14
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    atr_period = 14
    vol_sma_period = 20
    pivot_lookback = 5
    vol_mult = 1.5
    risk_pct = 0.01
    atr_sl_mult = 1.5
    tp1_mult = 2.0
    tp2_mult = 3.5
    trail_mult = 1.5
    time_stop_bars = 20
    divergence_window = 5
    atr_avg_period = 30
    cooldown_bars = 3

    def init(self):
        print("🌙 Initializing indicators... ✨")
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=self.rsi_period)

        close = np.asarray(self.data.Close, dtype=float)
        macd, macd_sig, macd_hist = talib.MACD(
            close,
            fastperiod=self.macd_fast,
            slowperiod=self.macd_slow,
            signalperiod=self.macd_signal
        )
        self.macd = self.I(lambda: macd)
        self.macd_sig = self.I(lambda: macd_sig)
        self.macd_hist = self.I(lambda: macd_hist)

        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period)
        self.atr_avg = self.I(talib.SMA, self.atr, timeperiod=self.atr_avg_period)
        self.vol_sma = self.I(talib.SMA, self.data.Volume, timeperiod=self.vol_sma_period)

        # Pivot detection
        self.pivot_high = self.I(talib.MAX, self.data.High, timeperiod=self.pivot_lookback * 2 + 1)
        self.pivot_low = self.I(talib.MIN, self.data.Low, timeperiod=self.pivot_lookback * 2 + 1)

        # State tracking
        self.entry_bar = None
        self.entry_price = None
        self.stop_price = None
        self.tp1_price = None
        self.tp2_price = None
        self.tp1_hit = False
        self.last_exit_bar = -100
        self.trade_direction = None

    def _val(self, arr, i):
        try:
            v = arr[i]
            if v is None:
                return np.nan
            return float(v)
        except Exception:
            return np.nan

    def _is_pivot_high(self, idx):
        if idx < self.pivot_lookback or idx >= len(self.data) - self.pivot_lookback:
            return False
        h = self.data.High[idx]
        for j in range(idx - self.pivot_lookback, idx + self.pivot_lookback + 1):
            if j == idx:
                continue
            if self.data.High[j] >= h:
                return False
        return True

    def _is_pivot_low(self, idx):
        if idx < self.pivot_lookback or idx >= len(self.data) - self.pivot_lookback:
            return False
        l = self.data.Low[idx]
        for j in range(idx - self.pivot_lookback, idx + self.pivot_lookback + 1):
            if j == idx:
                continue
            if self.data.Low[j] <= l:
                return False
        return True

    def _find_recent_pivots(self, idx, kind, n=2, lookback=40):
        """Find last n pivots of given kind before idx."""
        pivots = []
        start = max(self.pivot_lookback, idx - lookback)
        end = idx - self.pivot_lookback
        for i in range(end, start - 1, -1):
            if kind == 'high' and self._is_pivot_high(i):
                pivots.append(i)
                if len(pivots) >= n:
                    break
            elif kind == 'low' and self._is_pivot_low(i):
                pivots.append(i)
                if len(pivots) >= n:
                    break
        return pivots

    def _check_long_setup(self, idx):
        """Bullish RSI divergence + bearish MACD divergence + volume spike + breakout."""
        if idx < 60:
            return False

        pivots = self._find_recent_pivots(idx, 'low', n=2)
        if len(pivots) < 2:
            return False
        p1, p2 = pivots[0], pivots[1]

        if (p1 - p2) > self.divergence_window * 3:
            return False

        r1 = self._val(self.rsi, p1)
        r2 = self._val(self.rsi, p2)
        m1 = self._val(self.macd_hist, p1)
        m2 = self._val(self.macd_hist, p2)
        if np.isnan(r1) or np.isnan(r2) or np.isnan(m1) or np.isnan(m2):
            return False

        price_ll = self.data.Low[p1] < self.data.Low[p2]
        rsi_hl = r1 > r2
        rsi_bull_div = price_ll and rsi_hl

        macd_hl = m1 > m2
        macd_bear_div = price_ll and macd_hl

        if not (rsi_bull_div and macd_bear_div):
            return False

        if abs(p1 - p2) > self.divergence_window * 2:
            return False

        vol = self._val(self.data.Volume, idx)
        vsma = self._val(self.vol_sma, idx)
        if np.isnan(vol) or np.isnan(vsma):
            return False
        if vol < self.vol_mult * vsma:
            return False

        atr_v = self._val(self.atr, idx)
        atr_avg_v = self._val(self.atr_avg, idx)
        if np.isnan(atr_v) or np.isnan(atr_avg_v):
            return False
        if atr_v < atr_avg_v:
            return False

        recent_highs = self._find_recent_pivots(idx, 'high', n=1)
        if recent_highs:
            prior_high = self.data.High[recent_highs[0]]
            if self.data.Close[idx] <= prior_high:
                return False

        return True

    def _check_short_setup(self, idx):
        """Bearish RSI divergence + bullish MACD divergence + volume spike + breakdown."""
        if idx < 60:
            return False

        pivots = self._find_recent_pivots(idx, 'high', n=2)
        if len(pivots) < 2:
            return False
        p1, p2 = pivots[0], pivots[1]

        if (p1 - p2) > self.divergence_window * 3:
            return False

        r1 = self._val(self.rsi, p1)
        r2 = self._val(self.rsi, p2)
        m1 = self._val(self.macd_hist, p1)
        m2 = self._val(self.macd_hist, p2)
        if np.isnan(r1) or np.isnan(r2) or np.isnan(m1) or np.isnan(m2):
            return False

        price_hh = self.data.High[p1] > self.data.High[p2]
        rsi_lh = r1 < r2
        rsi_bear_div = price_hh and rsi_lh

        macd_lh = m1 < m2
        macd_bull_div = price_hh and macd_lh

        if not (rsi_bear_div and macd_bull_div):
            return False

        if abs(p1 - p2) > self.divergence_window * 2:
            return False

        vol = self._val(self.data.Volume, idx)
        vsma = self._val(self.vol_sma, idx)
        if np.isnan(vol) or np.isnan(vsma):
            return False
        if vol < self.vol_mult * vsma:
            return False

        atr_v = self._val(self.atr, idx)
        atr_avg_v = self._val(self.atr_avg, idx)
        if np.isnan(atr_v) or np.isnan(atr_avg_v):
            return False
        if atr_v < atr_avg_v:
            return False

        recent_lows = self._find_recent_pivots(idx, 'low', n=1)
        if recent_lows:
            prior_low = self.data.Low[recent_lows[0]]
            if self.data.Close[idx] >= prior_low:
                return False

        return True

    def next(self):
        idx = len(self.data) - 1
        price = self.data.Close[idx]

        if self.position:
            self._manage_position(idx, price)
            return

        if idx - self.last_exit_bar < self.cooldown_bars:
            return

        if self._check_long_setup(idx):
            atr_val = self._val(self.atr, idx)
            if np.isnan(atr_val) or atr_val <= 0:
                return
            sl = price - self.atr_sl_mult * atr_val
            tp1 = price + self.tp1_mult * atr_val
            tp2 = price + self.tp2_mult * atr_val

            risk_amount = self.equity * self.risk_pct
            stop_dist = price - sl
            if stop_dist <= 0:
                return
            size = int(round(risk_amount / stop_dist))
            if size < 1:
                size = 1

            print(f"🌙🚀 LONG SIGNAL at {price:.2f} | ATR={atr_val:.2f} | SL={sl:.2f} | TP1={tp1:.2f} | TP2={tp2:.2f} | Size={size}")
            self.buy(size=size)
            self.entry_price = price
            self.stop_price = sl
            self.tp1_price = tp1
            self.tp2_price = tp2
            self.tp1_hit = False
            self.entry_bar = idx
            self.trade_direction = 'long'

        elif self._check_short_setup(idx):
            atr_val = self._val(self.atr, idx)
            if np.isnan(atr_val) or atr_val <= 0:
                return
            sl = price + self.atr_sl_mult * atr_val
            tp1 = price - self.tp1_mult * atr_val
            tp2 = price - self.tp2_mult * atr_val

            risk_amount = self.equity * self.risk_pct
            stop_dist = sl - price
            if stop_dist <= 0:
                return
            size = int(round(risk_amount / stop_dist))
            if size < 1:
                size = 1

            print(f"🌙🔻 SHORT SIGNAL at {price:.2f} | ATR={atr_val:.2f} | SL={sl:.2f} | TP1={tp1:.2f} | TP2={tp2:.2f} | Size={size}")
            self.sell(size=size)
            self.entry_price = price
            self.stop_price = sl
            self.tp1_price = tp1
            self.tp2_price = tp2
            self.tp1_hit = False
            self.entry_bar = idx
            self.trade_direction = 'short'

    def _manage_position(self, idx, price):
        atr_val = self._val(self.atr, idx)
        if np.isnan(atr_val):
            atr_val = 0

        if self.trade_direction == 'long':
            if price <= self.stop_price:
                print(f"🌙💥 LONG STOP HIT at {price:.2f}")
                self.position.close()
                self.last_exit_bar = idx
                self.trade_direction = None
                return

            if not self.tp1_hit and price >= self.tp1_price:
                print(f"🌙✨ LONG TP1 HIT at {price:.2f} - scaling 50%")
                self.position.close()
                self.tp1_hit = True
                self.stop_price = max(self.stop_price, price - self.trail_mult * atr_val)
                return

            if self.tp1_hit and price >= self.tp2_price:
                print(f"🌙🎯 LONG TP2 HIT at {price:.2f}")
                self.position.close()
                self.last_exit_bar = idx
                self.trade_direction = None
                return

            if self.tp1_hit and atr_val > 0:
                new_stop = price - self.trail_mult * atr_val
                if new_stop > self.stop_price:
                    self.stop_price = new_stop

            if idx - self.entry_bar >= self.time_stop_bars:
                m_now = self._val(self.macd_hist, idx)
                m_prev = self._val(self.macd_hist, idx - 3)
                if not np.isnan(m_now) and not np.isnan(m_prev):
                    if abs(m_now) < abs(m_prev) * 0.5:
                        print(f"🌙⏰ LONG TIME STOP at {price:.2f}")
                        self.position.close()
                        self.last_exit_bar = idx
                        self.trade_direction = None

        elif self.trade_direction == 'short':
            if price >= self.stop_price:
                print(f"🌙💥 SHORT STOP HIT at {price:.2f}")
                self.position.close()
                self.last_exit_bar = idx
                self.trade_direction = None
                return

            if not self.tp1_hit and price <= self.tp1_price:
                print(f"🌙✨ SHORT TP1 HIT at {price:.2f} - scaling 50%")
                self.position.close()
                self.tp1_hit = True
                self.stop_price = min(self.stop_price, price + self.trail_mult * atr_val)
                return

            if self.tp1_hit and price <= self.tp2_price:
                print(f"🌙🎯 SHORT TP2 HIT at {price:.2f}")
                self.position.close()
                self.last_exit_bar = idx
                self.trade_direction = None
                return

            if self.tp1_hit and atr_val > 0:
                new_stop = price + self.trail_mult * atr_val
                if new_stop < self.stop_price:
                    self.stop_price = new_stop

            if idx - self.entry_bar >= self.time_stop_bars:
                m_now = self._val(self.macd_hist, idx)
                m_prev = self._val(self.macd_hist, idx - 3)
                if not np.isnan(m_now) and not np.isnan(m_prev):
                    if abs(m_now) < abs(m_prev) * 0.5:
                        print(f"🌙⏰ SHORT TIME STOP at {price:.2f}")
                        self.position.close()
                        self.last_exit_bar = idx
                        self.trade_direction = None


print("🌙✨ Running backtest... 🚀")
bt = Backtest(data, DivergentPulse, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙✨ Moon Dev Backtest Complete! 🚀")