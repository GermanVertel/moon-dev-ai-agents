import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

print("🌙✨ DivergentVolatility Strategy Loading... 🚀")

# Load data
data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Set datetime index BEFORE renaming
if 'datetime' in data.columns:
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')

# Map to proper case
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
print(f"🌙 Data loaded: {len(data)} bars ✨")


class DivergentVolatility(Strategy):
    # Parameters
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    rsi_period = 14
    atr_period = 14
    atr_sma_period = 20
    swing_window = 5
    lookback = 30
    confluence_window = 5
    atr_sl_mult = 1.5
    atr_tp1_mult = 2.0
    atr_tp2_mult = 3.5
    time_stop = 30
    risk_pct = 0.01

    def init(self):
        print("🌙 Initializing indicators... ✨")

        # MACD
        def _macd_hist(close):
            macd, macd_sig, macd_hist = talib.MACD(
                close, fastperiod=self.macd_fast,
                slowperiod=self.macd_slow, signalperiod=self.macd_signal
            )
            return macd_hist
        self.macd_hist = self.I(_macd_hist, self.data.Close, name='MACD_Hist')

        # RSI
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=self.rsi_period, name='RSI')

        # ATR
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close,
                          timeperiod=self.atr_period, name='ATR')

        # ATR SMA
        def _atr_sma(atr_series):
            return talib.SMA(atr_series, timeperiod=self.atr_sma_period)
        self.atr_sma = self.I(_atr_sma, self.atr, name='ATR_SMA')

        # Swing high/low using MAX/MIN
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=self.swing_window, name='SwingHigh')
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=self.swing_window, name='SwingLow')

        # State
        self.entry_price = None
        self.entry_bar = None
        self.tp1_hit = False
        self.stop_price = None
        self.tp1_price = None
        self.tp2_price = None
        self.trade_dir = None

        print("🌙 Indicators ready! 🚀")

    def _reset(self):
        """Reset trade state after position closes. 🌙"""
        self.entry_price = None
        self.entry_bar = None
        self.tp1_hit = False
        self.stop_price = None
        self.tp1_price = None
        self.tp2_price = None
        self.trade_dir = None

    def _find_divergence(self, i, direction):
        """
        direction: 'bull' or 'bear'
        Returns True if divergence detected in last few bars.
        """
        if i < self.lookback + 5:
            return False

        window_start = i - self.lookback
        hist = self.macd_hist
        rsi = self.rsi
        low_arr = self.data.Low
        high_arr = self.data.High

        macd_div = False
        rsi_div = False

        if direction == 'bull':
            lows = [(k, low_arr[k]) for k in range(window_start, i)]
            if len(lows) < 4:
                return False
            recent_slice = lows[-10:]
            prior_slice = lows[:-10]
            if not prior_slice:
                return False
            recent_low_idx, recent_low = min(recent_slice, key=lambda x: x[1])
            prior_low_idx, prior_low = min(prior_slice, key=lambda x: x[1])

            if recent_low < prior_low:
                if not np.isnan(hist[recent_low_idx]) and not np.isnan(hist[prior_low_idx]):
                    if hist[recent_low_idx] > hist[prior_low_idx]:
                        macd_div = True
                if not np.isnan(rsi[recent_low_idx]) and not np.isnan(rsi[prior_low_idx]):
                    if rsi[recent_low_idx] > rsi[prior_low_idx]:
                        rsi_div = True

        else:  # bear
            highs = [(k, high_arr[k]) for k in range(window_start, i)]
            if len(highs) < 4:
                return False
            recent_slice = highs[-10:]
            prior_slice = highs[:-10]
            if not prior_slice:
                return False
            recent_high_idx, recent_high = max(recent_slice, key=lambda x: x[1])
            prior_high_idx, prior_high = max(prior_slice, key=lambda x: x[1])

            if recent_high > prior_high:
                if not np.isnan(hist[recent_high_idx]) and not np.isnan(hist[prior_high_idx]):
                    if hist[recent_high_idx] < hist[prior_high_idx]:
                        macd_div = True
                if not np.isnan(rsi[recent_high_idx]) and not np.isnan(rsi[prior_high_idx]):
                    if rsi[recent_high_idx] < rsi[prior_high_idx]:
                        rsi_div = True

        return macd_div and rsi_div

    def next(self):
        i = len(self.data) - 1

        if i < max(self.lookback, self.atr_sma_period, self.swing_window) + 5:
            return

        price = self.data.Close[-1]
        atr = self.atr[-1]
        atr_sma = self.atr_sma[-1]
        swing_high = self.swing_high[-1]
        swing_low = self.swing_low[-1]

        if np.isnan(atr) or np.isnan(atr_sma):
            return

        if self.position:
            self._manage_position(i, price)
            return

        atr_filter = atr > atr_sma * 0.8
        if not atr_filter:
            return

        # Use previous bar's swing values to avoid look-ahead in signal comparison
        prev_swing_high = self.swing_high[-2] if len(self.swing_high) >= 2 else swing_high
        prev_swing_low = self.swing_low[-2] if len(self.swing_low) >= 2 else swing_low

        if (self._find_divergence(i, 'bull')
                and price > prev_swing_high):
            self._enter_long(price, atr)
        elif (self._find_divergence(i, 'bear')
              and price < prev_swing_low):
            self._enter_short(price, atr)

    def _enter_long(self, price, atr):
        equity = self.equity
        risk_amount = equity * self.risk_pct
        sl_dist = self.atr_sl_mult * atr
        if sl_dist <= 0:
            return
        # Use fraction-of-equity sizing to avoid insufficient margin with unit sizing
        size_units = risk_amount / sl_dist
        size_frac = (size_units * price) / equity
        if size_frac <= 0:
            return
        if size_frac > 0.99:
            size_frac = 0.99

        self.stop_price = price - sl_dist
        self.tp1_price = price + self.atr_tp1_mult * atr
        self.tp2_price = price + self.atr_tp2_mult * atr
        self.entry_price = price
        self.entry_bar = len(self.data) - 1
        self.tp1_hit = False
        self.trade_dir = 'long'

        self.buy(size=size_frac)
        print(f"🌙🚀 LONG ENTRY @ {price:.2f} | SL: {self.stop_price:.2f} | TP1: {self.tp1_price:.2f} | TP2: {self.tp2_price:.2f} | Size: {size_frac:.4f} ✨")

    def _enter_short(self, price, atr):
        equity = self.equity
        risk_amount = equity * self.risk_pct
        sl_dist = self.atr_sl_mult * atr
        if sl_dist <= 0:
            return
        size_units = risk_amount / sl_dist
        size_frac = (size_units * price) / equity
        if size_frac <= 0:
            return
        if size_frac > 0.99:
            size_frac = 0.99

        self.stop_price = price + sl_dist
        self.tp1_price = price - self.atr_tp1_mult * atr
        self.tp2_price = price - self.atr_tp2_mult * atr
        self.entry_price = price
        self.entry_bar = len(self.data) - 1
        self.tp1_hit = False
        self.trade_dir = 'short'

        self.sell(size=size_frac)
        print(f"🌙🔻 SHORT ENTRY @ {price:.2f} | SL: {self.stop_price:.2f} | TP1: {self.tp1_price:.2f} | TP2: {self.tp2_price:.2f} | Size: {size_frac:.4f} ✨")

    def _manage_position(self, i, price):
        bars_held = i - self.entry_bar
        atr = self.atr[-1]

        if self.trade_dir == 'long':
            if price <= self.stop_price:
                self.position.close()
                print(f"🌙❌ LONG STOP HIT @ {price:.2f}")
                self._reset()
                return
            if not self.tp1_hit and price >= self.tp1_price:
                self.tp1_hit = True
                self.stop_price = self.entry_price
                print(f"🌙✅ LONG TP1 HIT @ {price:.2f} — stop to breakeven 🚀")
            if self.tp1_hit and price >= self.tp2_price:
                self.position.close()
                print(f"🌙🎯 LONG TP2 HIT @ {price:.2f} ✨")
                self._reset()
                return
            if self.tp1_hit and not np.isnan(atr):
                new_stop = price - self.atr_sl_mult * atr
                if new_stop > self.stop_price:
                    self.stop_price = new_stop

        else:  # short
            if price >= self.stop_price:
                self.position.close()
                print(f"🌙❌ SHORT STOP HIT @ {price:.2f}")
                self._reset()
                return
            if not self.tp1_hit and price <= self.tp1_price:
                self.tp1_hit = True
                self.stop_price = self.entry_price
                print(f"🌙✅ SHORT TP1 HIT @ {price:.2f} — stop to breakeven 🚀")
            if self.tp1_hit and price <= self.tp2_price:
                self.position.close()
                print(f"🌙🎯 SHORT TP2 HIT @ {price:.2f} ✨")
                self._reset()
                return
            if self.tp1_hit and not np.isnan(atr):
                new_stop = price + self.atr_sl_mult * atr
                if new_stop < self.stop_price:
                    self.stop_price = new_stop

        if bars_held >= self.time_stop:
            self.position.close()
            print(f"🌙⏰ TIME STOP @ {price:.2f} after {bars_held} bars")
            self._reset()


print("🌙✨ Running DivergentVolatility Backtest... 🚀")
bt = Backtest(data, DivergentVolatility, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙✨ Backtest complete! 🚀")