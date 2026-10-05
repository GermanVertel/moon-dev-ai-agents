import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Load and clean data
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
print("🌙✨ Moon Dev Data Loaded! Rows:", len(data), "✨🚀")


class DivergentVolatility(Strategy):
    # Strategy parameters
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    rsi_period = 14
    atr_period = 14
    atr_ma_period = 20
    atr_expansion_mult = 1.5
    pivot_window = 5
    lookback = 50
    min_gap = 5
    risk_pct = 0.02

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low

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
        self.atr_ma = self.I(talib.SMA, self.atr, timeperiod=self.atr_ma_period)

        # Swing highs/lows (rolling max/min over pivot window)
        self.swing_high = self.I(talib.MAX, high, timeperiod=self.pivot_window * 2 + 1)
        self.swing_low = self.I(talib.MIN, low, timeperiod=self.pivot_window * 2 + 1)

        print("🌙✨ DivergentVolatility indicators initialized! 🚀")

    def _find_pivots(self, arr, i, mode, window):
        """Find pivot indices in last `lookback` bars ending at i."""
        pivots = []
        start = max(window, i - self.lookback)
        for j in range(start, i):
            if j - window < 0 or j + window >= len(arr):
                continue
            seg = arr[j - window: j + window + 1]
            if mode == 'high':
                if arr[j] == np.max(seg):
                    pivots.append(j)
            else:
                if arr[j] == np.min(seg):
                    pivots.append(j)
        return pivots

    def next(self):
        i = len(self.data) - 1
        if i < self.lookback + self.pivot_window * 2 + 5:
            return

        # ===== EXIT LOGIC =====
        if self.position:
            atr_val = self.atr[-1]
            atr_ma_val = self.atr_ma[-1]
            macd_hist_val = self.macd_hist[-1]

            # Primary exit: ATR expansion
            if not np.isnan(atr_val) and not np.isnan(atr_ma_val):
                if atr_val > self.atr_expansion_mult * atr_ma_val:
                    print(f"🌙✨ ATR EXPANSION EXIT! ATR={atr_val:.2f} > {self.atr_expansion_mult}xMA={atr_ma_val:.2f} 🚀")
                    self.position.close()
                    return

            # Secondary exit: MACD histogram flips positive
            if not np.isnan(macd_hist_val) and macd_hist_val > 0 and self.macd_hist[-2] <= 0:
                print(f"🌙✨ MACD HIST FLIP EXIT! hist={macd_hist_val:.4f} 🚀")
                self.position.close()
                return
            return

        # ===== ENTRY LOGIC =====
        # Find swing lows (for MACD bullish divergence)
        low_pivots = self._find_pivots(self.data.Low, i, 'low', self.pivot_window)
        # Find swing highs (for RSI bearish divergence)
        high_pivots = self._find_pivots(self.data.High, i, 'high', self.pivot_window)

        macd_bull_div = False
        rsi_bear_div = False

        # MACD bullish divergence: price lower low, MACD higher low
        if len(low_pivots) >= 2:
            p1, p2 = low_pivots[-2], low_pivots[-1]
            if p2 - p1 >= self.min_gap:
                price_low_1 = self.data.Low[p1]
                price_low_2 = self.data.Low[p2]
                macd_low_1 = self.macd[p1]
                macd_low_2 = self.macd[p2]
                if (not np.isnan(macd_low_1) and not np.isnan(macd_low_2)
                        and price_low_2 < price_low_1
                        and macd_low_2 > macd_low_1):
                    macd_bull_div = True

        # RSI bearish divergence: price higher high, RSI lower high
        if len(high_pivots) >= 2:
            p1, p2 = high_pivots[-2], high_pivots[-1]
            if p2 - p1 >= self.min_gap:
                price_high_1 = self.data.High[p1]
                price_high_2 = self.data.High[p2]
                rsi_high_1 = self.rsi[p1]
                rsi_high_2 = self.rsi[p2]
                if (not np.isnan(rsi_high_1) and not np.isnan(rsi_high_2)
                        and price_high_2 > price_high_1
                        and rsi_high_2 < rsi_high_1):
                    rsi_bear_div = True

        # Both divergences required
        if macd_bull_div and rsi_bear_div:
            atr_val = self.atr[-1]
            if np.isnan(atr_val) or atr_val <= 0:
                return

            price = self.data.Close[-1]
            stop_loss = price + 2 * atr_val
            take_profit = price - 2 * (2 * atr_val)  # 1:2 R:R

            # Position sizing: risk_pct of equity / stop distance
            equity = self.equity
            risk_amount = equity * self.risk_pct
            risk_per_unit = 2 * atr_val
            if risk_per_unit <= 0:
                return
            position_size = int(round(risk_amount / risk_per_unit))
            if position_size <= 0:
                position_size = 1

            print(f"🌙🚀 SHORT SIGNAL! MACD bull div + RSI bear div | price={price:.2f} "
                  f"SL={stop_loss:.2f} TP={take_profit:.2f} size={position_size}")

            self.sell(size=position_size, sl=stop_loss, tp=take_profit)


bt = Backtest(data, DivergentVolatility, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)