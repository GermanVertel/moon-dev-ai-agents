import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Load and prepare data
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
print("🌙 Moon Dev Data Loaded! Shape:", data.shape, "✨")
print("🚀 First rows:\n", data.head(), "🌙")


class VolumeDivergence(Strategy):
    # Parameters
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    vol_sma_period = 20
    atr_period = 14
    ema_fast = 50
    ema_slow = 200
    divergence_lookback = 20
    min_divergence_bars = 5
    vol_ratio_threshold = 1.2
    risk_pct = 0.01
    atr_sl_mult = 1.5
    atr_tp1_mult = 2.0
    atr_tp2_mult = 3.5
    atr_trail_mult = 1.0
    time_stop_bars = 15
    max_positions = 3

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        # MACD
        macd, macd_signal, macd_hist = talib.MACD(
            close, fastperiod=self.macd_fast, slowperiod=self.macd_slow, signalperiod=self.macd_signal
        )
        self.macd = self.I(lambda x: x, macd, name='MACD')
        self.macd_sig = self.I(lambda x: x, macd_signal, name='MACD_Signal')
        self.macd_hist = self.I(lambda x: x, macd_hist, name='MACD_Hist')

        # Volume SMA
        self.vol_sma = self.I(talib.SMA, volume, timeperiod=self.vol_sma_period, name='Vol_SMA')

        # ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period, name='ATR')
        self.atr_avg = self.I(talib.SMA, self.atr, timeperiod=20, name='ATR_Avg')

        # EMAs
        self.ema50 = self.I(talib.EMA, close, timeperiod=self.ema_fast, name='EMA50')
        self.ema200 = self.I(talib.EMA, close, timeperiod=self.ema_slow, name='EMA200')

        # Track trade state
        self.trade_state = {}
        print("🌙✨ Moon Dev Indicators Initialized! 🚀")

    def _find_bullish_divergence(self, i):
        """Price lower low, MACD hist higher low over lookback window."""
        lookback = self.divergence_lookback
        if i < lookback + self.min_divergence_bars:
            return False

        # Find recent swing low in price
        window_low = self.data.Low[i - lookback:i]
        if len(window_low) == 0:
            return False
        current_low = self.data.Low[i]
        min_low = np.min(window_low)
        if current_low >= min_low:
            return False

        # Find prior swing low before current
        prior_start = max(0, i - 2 * lookback)
        prior_end = i - self.min_divergence_bars
        if prior_end <= prior_start:
            return False
        prior_window = self.data.Low[prior_start:prior_end]
        if len(prior_window) == 0:
            return False
        prior_low = np.min(prior_window)
        prior_low_idx = prior_start + int(np.argmin(prior_window))

        # Check MACD hist at prior low vs current - higher low
        if prior_low_idx >= i:
            return False
        hist_prior = self.macd_hist[prior_low_idx]
        hist_current = self.macd_hist[i]
        if np.isnan(hist_prior) or np.isnan(hist_current):
            return False

        # Price made lower low, histogram made higher low
        if current_low < prior_low and hist_current > hist_prior and hist_current < 0:
            return True
        return False

    def _find_bearish_divergence(self, i):
        """Price higher high, MACD hist lower high over lookback window."""
        lookback = self.divergence_lookback
        if i < lookback + self.min_divergence_bars:
            return False

        window_high = self.data.High[i - lookback:i]
        if len(window_high) == 0:
            return False
        current_high = self.data.High[i]
        max_high = np.max(window_high)
        if current_high <= max_high:
            return False

        prior_start = max(0, i - 2 * lookback)
        prior_end = i - self.min_divergence_bars
        if prior_end <= prior_start:
            return False
        prior_window = self.data.High[prior_start:prior_end]
        if len(prior_window) == 0:
            return False
        prior_high = np.max(prior_window)
        prior_high_idx = prior_start + int(np.argmax(prior_window))

        if prior_high_idx >= i:
            return False
        hist_prior = self.macd_hist[prior_high_idx]
        hist_current = self.macd_hist[i]
        if np.isnan(hist_prior) or np.isnan(hist_current):
            return False

        if current_high > prior_high and hist_current < hist_prior and hist_current > 0:
            return True
        return False

    def _count_open_trades(self):
        return sum(1 for t in self.trades if t.is_long or t.is_short)

    def next(self):
        i = len(self.data) - 1
        if i < 210:
            return

        close = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        prev_high = self.data.High[-2]
        prev_low = self.data.Low[-2]
        volume = self.data.Volume[-1]

        vol_sma = self.vol_sma[-1]
        atr = self.atr[-1]
        atr_avg = self.atr_avg[-1]
        ema50 = self.ema50[-1]
        ema200 = self.ema200[-1]
        macd_hist = self.macd_hist[-1]

        if np.isnan(atr) or np.isnan(atr_avg) or np.isnan(vol_sma) or vol_sma == 0:
            return

        # Volatility filter
        if atr < atr_avg:
            return

        vol_ratio = volume / vol_sma

        # Manage existing trades
        for trade in list(self.trades):
            tid = id(trade)
            if tid not in self.trade_state:
                entry = trade.entry_price
                if trade.is_long:
                    sl = entry - self.atr_sl_mult * atr
                    tp1 = entry + self.atr_tp1_mult * atr
                    tp2 = entry + self.atr_tp2_mult * atr
                else:
                    sl = entry + self.atr_sl_mult * atr
                    tp1 = entry - self.atr_tp1_mult * atr
                    tp2 = entry - self.atr_tp2_mult * atr
                self.trade_state[tid] = {
                    'sl': sl, 'tp1': tp1, 'tp2': tp2,
                    'tp1_hit': False, 'entry_bar': i,
                    'highest_close': entry, 'lowest_close': entry,
                    'entry_atr': atr,
                }

            st = self.trade_state[tid]
            bars_held = i - st['entry_bar']

            if trade.is_long:
                st['highest_close'] = max(st['highest_close'], close)
                # TP1
                if not st['tp1_hit'] and high >= st['tp1']:
                    st['tp1_hit'] = True
                    close_size = max(1, int(trade.size * 0.5))
                    if close_size >= trade.size:
                        close_size = max(1, int(trade.size) - 1)
                    try:
                        trade.close(close_size)
                        print(f"🌙✨ TP1 HIT LONG! Partial close {close_size} @ {st['tp1']:.2f} 🚀")
                    except Exception as e:
                        print(f"⚠️ TP1 close error: {e}")

                # TP2
                if high >= st['tp2']:
                    trade.close()
                    print(f"🌙🚀 TP2 HIT LONG! Full close @ {st['tp2']:.2f} ✨")
                    del self.trade_state[tid]
                    continue

                # Trailing stop after TP1
                if st['tp1_hit']:
                    trail = st['highest_close'] - self.atr_trail_mult * st['entry_atr']
                    st['sl'] = max(st['sl'], trail)

                # Stop loss
                if low <= st['sl']:
                    trade.close()
                    print(f"🌙💀 STOP LOSS LONG @ {st['sl']:.2f} 🌙")
                    del self.trade_state[tid]
                    continue

                # Reversal exit - MACD hist crosses below zero
                if macd_hist < 0:
                    trade.close()
                    print(f"🌙🔄 REVERSAL EXIT LONG! MACD hist < 0 🌙")
                    del self.trade_state[tid]
                    continue

                # Time stop
                if not st['tp1_hit'] and bars_held >= self.time_stop_bars:
                    trade.close()
                    print(f"🌙⏰ TIME STOP LONG! {bars_held} bars 🌙")
                    del self.trade_state[tid]
                    continue

            else:  # short
                st['lowest_close'] = min(st['lowest_close'], close)
                if not st['tp1_hit'] and low <= st['tp1']:
                    st['tp1_hit'] = True
                    close_size = max(1, int(trade.size * 0.5))
                    if close_size >= trade.size:
                        close_size = max(1, int(trade.size) - 1)
                    try:
                        trade.close(close_size)
                        print(f"🌙✨ TP1 HIT SHORT! Partial close {close_size} @ {st['tp1']:.2f} 🚀")
                    except Exception as e:
                        print(f"⚠️ TP1 close error: {e}")

                if low <= st['tp2']:
                    trade.close()
                    print(f"🌙🚀 TP2 HIT SHORT! Full close @ {st['tp2']:.2f} ✨")
                    del self.trade_state[tid]
                    continue

                if st['tp1_hit']:
                    trail = st['lowest_close'] + self.atr_trail_mult * st['entry_atr']
                    st['sl'] = min(st['sl'], trail)

                if high >= st['sl']:
                    trade.close()
                    print(f"🌙💀 STOP LOSS SHORT @ {st['sl']:.2f} 🌙")
                    del self.trade_state[tid]
                    continue

                if macd_hist > 0:
                    trade.close()
                    print(f"🌙🔄 REVERSAL EXIT SHORT! MACD hist > 0 🌙")
                    del self.trade_state[tid]
                    continue

                if not st['tp1_hit'] and bars_held >= self.time_stop_bars:
                    trade.close()
                    print(f"🌙⏰ TIME STOP SHORT! {bars_held} bars 🌙")
                    del self.trade_state[tid]
                    continue

        # Entry logic
        if self._count_open_trades() >= self.max_positions:
            return

        # Long entry
        if (self._find_bullish_divergence(i)
                and vol_ratio > self.vol_ratio_threshold
                and close > prev_high
                and (close > ema200 or close > ema50)):
            sl_price = close - self.atr_sl_mult * atr
            risk_per_unit = close - sl_price
            if risk_per_unit <= 0:
                return
            risk_amount = self.equity * self.risk_pct
            size = int(round(risk_amount / risk_per_unit))
            if size < 1:
                size = 1
            self.buy(size=size)
            print(f"🌙🚀 LONG ENTRY! Price: {close:.2f} VolRatio: {vol_ratio:.2f} SL: {sl_price:.2f} Size: {size} ✨")

        # Short entry
        elif (self._find_bearish_divergence(i)
                and vol_ratio > self.vol_ratio_threshold
                and close < prev_low
                and (close < ema200 or close < ema50)):
            sl_price = close + self.atr_sl_mult * atr
            risk_per_unit = sl_price - close
            if risk_per_unit <= 0:
                return
            risk_amount = self.equity * self.risk_pct
            size = int(round(risk_amount / risk_per_unit))
            if size < 1:
                size = 1
            self.sell(size=size)
            print(f"🌙🚀 SHORT ENTRY! Price: {close:.2f} VolRatio: {vol_ratio:.2f} SL: {sl_price:.2f} Size: {size} ✨")


print("🌙✨ Starting Moon Dev VolumeDivergence Backtest! 🚀")
bt = Backtest(data, VolumeDivergence, cash=1000000, commission=0.001, exclusive_orders=False)
stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙✨ Moon Dev Backtest Complete! 🚀")