import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

print("🌙✨ Moon Dev's DivergentContrarian Backtest Loading... ✨🌙")

# Load and clean data
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'datetime': 'Datetime',
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})
data['Datetime'] = pd.to_datetime(data['Datetime'])
data = data.set_index('Datetime')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
print(f"🚀 Data loaded: {len(data)} bars from {data.index[0]} to {data.index[-1]}")


class DivergentContrarian(Strategy):
    # Parameters
    rsi_period = 14
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    pivot_lookback = 5
    divergence_window = 60
    min_width = 0.5
    risk_pct = 0.015
    atr_period = 14
    atr_stop_mult = 2.0
    reward_ratio = 2.0
    time_exit_bars = 40
    ema_trail = 20

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low

        def _macd(c, fast, slow, sig):
            m, s, h = talib.MACD(c, fastperiod=fast, slowperiod=slow, signalperiod=sig)
            return m, s, h

        self.macd, self.macd_signal, self.macd_hist = self.I(
            _macd, close,
            self.macd_fast, self.macd_slow, self.macd_signal
        )
        self.rsi = self.I(talib.RSI, close, timeperiod=self.rsi_period)
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)
        self.ema = self.I(talib.EMA, close, timeperiod=self.ema_trail)

        self.swing_high = self.I(talib.MAX, high, timeperiod=self.pivot_lookback * 2 + 1)
        self.swing_low = self.I(talib.MIN, low, timeperiod=self.pivot_lookback * 2 + 1)

        self.entry_bar = None
        self.stop_price = None
        self.tp_price = None
        self.trade_dir = 0
        self.entry_price_val = None

        print("🌙 Indicators initialized: MACD, RSI, ATR, EMA, Swings ✨")

    def _find_swing_idx(self, arr, idx, lookback, mode='high'):
        start = max(lookback, idx - self.divergence_window)
        best_i = -1
        best_v = -np.inf if mode == 'high' else np.inf
        for i in range(start, idx - lookback):
            if i - lookback < 0 or i + lookback + 1 > len(arr):
                continue
            v = arr[i]
            if np.isnan(v):
                continue
            window = arr[i - lookback:i + lookback + 1]
            if mode == 'high':
                wmax = np.nanmax(window)
                if v == wmax and v > best_v:
                    best_v = v
                    best_i = i
            else:
                wmin = np.nanmin(window)
                if v == wmin and v < best_v:
                    best_v = v
                    best_i = i
        return best_i, best_v

    def next(self):
        i = len(self.data) - 1
        if i < self.divergence_window + self.pivot_lookback + 5:
            return

        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]

        # ---------------- Manage open trade ----------------
        if self.position:
            if self.entry_bar is not None and (i - self.entry_bar) >= self.time_exit_bars:
                print(f"⏰ Moon Dev Time Exit at {price:.2f} 🌙")
                self.position.close()
                self._reset()
                return

            if self.trade_dir == 1 and low <= self.stop_price:
                print(f"🛑 Long STOP hit at {self.stop_price:.2f} 🌙")
                self.position.close()
                self._reset()
                return
            if self.trade_dir == -1 and high >= self.stop_price:
                print(f"🛑 Short STOP hit at {self.stop_price:.2f} 🌙")
                self.position.close()
                self._reset()
                return

            if self.trade_dir == 1 and high >= self.tp_price:
                print(f"🎯 Long TP hit at {self.tp_price:.2f} 🚀")
                self.position.close()
                self._reset()
                return
            if self.trade_dir == -1 and low <= self.tp_price:
                print(f"🎯 Short TP hit at {self.tp_price:.2f} 🚀")
                self.position.close()
                self._reset()
                return

            if self.trade_dir == 1 and price < self.ema[-1] and price > self.entry_price_val:
                print(f"📉 Long trail exit (EMA) at {price:.2f} ✨")
                self.position.close()
                self._reset()
                return
            if self.trade_dir == -1 and price > self.ema[-1] and price < self.entry_price_val:
                print(f"📈 Short trail exit (EMA) at {price:.2f} ✨")
                self.position.close()
                self._reset()
                return
            return

        # ---------------- Detect divergences ----------------
        high_arr = np.asarray(self.data.High)
        low_arr = np.asarray(self.data.Low)

        sh_idx, sh_val = self._find_swing_idx(high_arr, i, self.pivot_lookback, 'high')
        sl_idx, sl_val = self._find_swing_idx(low_arr, i, self.pivot_lookback, 'low')

        if sh_idx < 0 or sl_idx < 0:
            return

        psh_idx, psh_val = self._find_swing_idx(high_arr, sh_idx, self.pivot_lookback, 'high')
        psl_idx, psl_val = self._find_swing_idx(low_arr, sl_idx, self.pivot_lookback, 'low')

        if psh_idx < 0 or psl_idx < 0:
            return

        # --- MACD divergence ---
        macd_bull = False
        macd_bear = False
        macd_width = 0.0

        if sl_val < psl_val and self.macd[sl_idx] > self.macd[psl_idx]:
            macd_bull = True
            macd_width = abs(self.macd[sl_idx] - self.macd[psl_idx])
        if sh_val > psh_val and self.macd[sh_idx] < self.macd[psh_idx]:
            macd_bear = True
            macd_width = abs(self.macd[sh_idx] - self.macd[psh_idx])

        # --- RSI divergence ---
        rsi_bull = False
        rsi_bear = False
        rsi_width = 0.0

        if sl_val < psl_val and self.rsi[sl_idx] > self.rsi[psl_idx]:
            rsi_bull = True
            rsi_width = abs(self.rsi[sl_idx] - self.rsi[psl_idx])
        if sh_val > psh_val and self.rsi[sh_idx] < self.rsi[psh_idx]:
            rsi_bear = True
            rsi_width = abs(self.rsi[sh_idx] - self.rsi[psh_idx])

        # --- Conflict filter ---
        long_conflict = macd_bull and rsi_bear
        short_conflict = macd_bear and rsi_bull

        # --- Divergence width ---
        width = macd_width + rsi_width
        if width < self.min_width:
            return

        # --- Trigger bar ---
        hist_now = self.macd_hist[-1]
        hist_prev = self.macd_hist[-2]
        rsi_now = self.rsi[-1]
        rsi_prev = self.rsi[-2]

        long_trigger = hist_now > hist_prev and rsi_now > 50 and rsi_prev <= 50
        short_trigger = hist_now < hist_prev and rsi_now < 50 and rsi_prev >= 50

        # ---------------- Entry ----------------
        if long_conflict and long_trigger and price < self.ema[-1]:
            self._enter_long(i, price, sl_val, sh_val)
        elif short_conflict and short_trigger and price > self.ema[-1]:
            self._enter_short(i, price, sl_val, sh_val)

    def _enter_long(self, i, price, sl_val, sh_val):
        atr = self.atr[-1]
        if np.isnan(atr) or atr <= 0:
            return
        stop = min(sl_val, price - atr * self.atr_stop_mult)
        risk = price - stop
        if risk <= 0:
            return
        tp = price + risk * self.reward_ratio

        equity = self.equity
        risk_amount = equity * self.risk_pct
        raw_size = risk_amount / risk
        # Use fraction of equity for sizing to avoid cash issues
        size = min(raw_size * price / equity, 0.99)
        if size <= 0:
            return

        self.buy(size=size)
        self.entry_bar = i
        self.stop_price = stop
        self.tp_price = tp
        self.trade_dir = 1
        self.entry_price_val = price
        print(f"🌙✨ LONG DivergentContrarian @ {price:.2f} | SL {stop:.2f} | TP {tp:.2f} | size {size:.4f} 🚀")

    def _enter_short(self, i, price, sl_val, sh_val):
        atr = self.atr[-1]
        if np.isnan(atr) or atr <= 0:
            return
        stop = max(sh_val, price + atr * self.atr_stop_mult)
        risk = stop - price
        if risk <= 0:
            return
        tp = price - risk * self.reward_ratio

        equity = self.equity
        risk_amount = equity * self.risk_pct
        raw_size = risk_amount / risk
        size = min(raw_size * price / equity, 0.99)
        if size <= 0:
            return

        self.sell(size=size)
        self.entry_bar = i
        self.stop_price = stop
        self.tp_price = tp
        self.trade_dir = -1
        self.entry_price_val = price
        print(f"🌙✨ SHORT DivergentContrarian @ {price:.2f} | SL {stop:.2f} | TP {tp:.2f} | size {size:.4f} 🚀")

    def _reset(self):
        self.entry_bar = None
        self.stop_price = None
        self.tp_price = None
        self.trade_dir = 0
        self.entry_price_val = None


bt = Backtest(
    data,
    DivergentContrarian,
    cash=1_000_000,
    commission=0.0002,
    exclusive_orders=True
)

print("🌙 Running Moon Dev's DivergentContrarian backtest... 🚀")
stats = bt.run()
print(stats)
print(stats._strategy)
print("✨🌙 Moon Dev Backtest Complete 🌙✨")