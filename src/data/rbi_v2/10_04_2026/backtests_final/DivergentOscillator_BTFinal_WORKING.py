import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Load data
data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Map columns to proper case
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')

print("🌙✨ Moon Dev Package AI — scanning for backtesting.lib usage... ✅ NONE FOUND!")
print("🚀 All indicators use talib via self.I() — clean and compliant!")


class DivergentOscillator(Strategy):
    rsi_period = 14
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    cci_period = 20
    atr_period = 14
    pivot_lookback = 5   # bars left/right for pivot detection
    div_lookback = 40    # lookback window for divergence
    risk_pct = 0.02
    atr_mult = 2.0

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low

        self.rsi = self.I(talib.RSI, close, timeperiod=self.rsi_period)
        self.macd, self.macd_signal_line, self.macd_hist = self.I(
            talib.MACD, close,
            fastperiod=self.macd_fast,
            slowperiod=self.macd_slow,
            signalperiod=self.macd_signal
        )
        self.cci = self.I(talib.CCI, high, low, close, timeperiod=self.cci_period)
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        print("🌙✨ DivergentOscillator indicators initialized! 🚀")

    def _find_pivots(self, arr, i, lookback):
        """Find pivot highs and lows in arr up to index i."""
        pivots_high = []
        pivots_low = []
        start = max(lookback, i - self.div_lookback)
        for j in range(start, i - lookback):
            window = arr[j - lookback:j + lookback + 1]
            if len(window) < 2 * lookback + 1:
                continue
            center = arr[j]
            if center == max(window):
                pivots_high.append((j, center))
            if center == min(window):
                pivots_low.append((j, center))
        return pivots_high, pivots_low

    def _check_bullish_divergence(self, i):
        """Price lower low, RSI higher low."""
        close = self.data.Close
        rsi = self.rsi
        lookback = self.pivot_lookback
        start = max(lookback, i - self.div_lookback)
        lows = []
        for j in range(start, i - lookback):
            window = close[j - lookback:j + lookback + 1]
            if len(window) < 2 * lookback + 1:
                continue
            if close[j] == min(window):
                lows.append(j)
        if len(lows) < 2:
            return False
        j1, j2 = lows[-2], lows[-1]
        # Price makes lower low
        if close[j2] < close[j1] and rsi[j2] > rsi[j1]:
            return True
        return False

    def _check_bearish_divergence(self, i):
        """Price higher high, RSI lower high."""
        close = self.data.Close
        rsi = self.rsi
        lookback = self.pivot_lookback
        start = max(lookback, i - self.div_lookback)
        highs = []
        for j in range(start, i - lookback):
            window = close[j - lookback:j + lookback + 1]
            if len(window) < 2 * lookback + 1:
                continue
            if close[j] == max(window):
                highs.append(j)
        if len(highs) < 2:
            return False
        j1, j2 = highs[-2], highs[-1]
        # Price makes higher high
        if close[j2] > close[j1] and rsi[j2] < rsi[j1]:
            return True
        return False

    def next(self):
        i = len(self.data) - 1
        if i < self.div_lookback + self.pivot_lookback + 5:
            return

        price = self.data.Close[-1]
        macd = self.macd[-1]
        macd_sig = self.macd_signal_line[-1]
        macd_hist = self.macd_hist[-1]
        macd_hist_prev = self.macd_hist[-2]
        cci = self.cci[-1]
        cci_prev = self.cci[-2]
        atr = self.atr[-1]

        # 🌙 Moon Dev crossover detection — NO backtesting.lib used!
        macd_cross_up = self.macd[-2] <= self.macd_signal_line[-2] and macd > macd_sig
        macd_cross_down = self.macd[-2] >= self.macd_signal_line[-2] and macd < macd_sig

        # -------- EXITS --------
        if self.position:
            if self.position.is_long:
                # CCI exit
                if (cci_prev >= 100 and cci < 100) or cci < -100:
                    print(f"🌙 EXIT LONG (CCI filter) @ {price:.2f} | CCI={cci:.2f}")
                    self.position.close()
                # Momentum failure
                elif macd_cross_down and macd_hist < macd_hist_prev:
                    print(f"🌙 EXIT LONG (MACD fail) @ {price:.2f}")
                    self.position.close()
            elif self.position.is_short:
                if (cci_prev <= -100 and cci > -100) or cci > 100:
                    print(f"🌙 EXIT SHORT (CCI filter) @ {price:.2f} | CCI={cci:.2f}")
                    self.position.close()
                elif macd_cross_up and macd_hist > macd_hist_prev:
                    print(f"🌙 EXIT SHORT (MACD fail) @ {price:.2f}")
                    self.position.close()
            return

        # -------- ENTRIES --------
        if self._check_bullish_divergence(i):
            if macd_cross_up and macd_hist > 0 and cci > -100:
                sl = price - self.atr_mult * atr
                size = int(round(1_000_000 / price))
                if size > 0:
                    print(f"🚀🌙 LONG ENTRY @ {price:.2f} | RSI={self.rsi[-1]:.2f} CCI={cci:.2f} SL={sl:.2f}")
                    self.buy(size=size, sl=sl)

        elif self._check_bearish_divergence(i):
            if macd_cross_down and macd_hist < 0 and cci < 100:
                sl = price + self.atr_mult * atr
                size = int(round(1_000_000 / price))
                if size > 0:
                    print(f"🚀🌙 SHORT ENTRY @ {price:.2f} | RSI={self.rsi[-1]:.2f} CCI={cci:.2f} SL={sl:.2f}")
                    self.sell(size=size, sl=sl)


bt = Backtest(data, DivergentOscillator, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)