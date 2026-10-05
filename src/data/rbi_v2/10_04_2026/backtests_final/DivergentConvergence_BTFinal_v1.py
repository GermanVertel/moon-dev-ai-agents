import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Load and clean data
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
print("🌙 Data loaded and cleaned! Rows:", len(data))


class DivergentConvergence(Strategy):
    rsi_period = 14
    stoch_k = 14
    stoch_d = 3
    stoch_smooth = 3
    atr_period = 14
    ema_period = 50
    divergence_lookback = 30
    risk_reward = 2.0
    risk_pct = 0.02

    def init(self):
        self.rsi = self.I(talib.RSI, self.data.Close, self.rsi_period)
        self.stoch_k, self.stoch_d = self.I(
            talib.STOCH, self.data.High, self.data.Low, self.data.Close,
            fastk_period=self.stoch_k, slowk_period=self.stoch_smooth,
            slowk_matype=0, slowd_period=self.stoch_d, slowd_matype=0
        )
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.atr_period)
        self.ema = self.I(talib.EMA, self.data.Close, self.ema_period)
        print("🌙✨ Indicators initialized! 🚀")

    def _bullish_divergence(self, i):
        lb = self.divergence_lookback
        if i < lb + 5:
            return False
        window_low = np.min(self.data.Low[i - lb:i - 2])
        current_low = self.data.Low[i - 1]
        if current_low >= window_low:
            return False
        idx_low = i - lb + int(np.argmin(self.data.Low[i - lb:i - 2]))
        rsi_curr = self.rsi[i - 1]
        rsi_prev = self.rsi[idx_low]
        if np.isnan(rsi_curr) or np.isnan(rsi_prev):
            return False
        if rsi_curr > rsi_prev and rsi_curr < 45:
            return True
        return False

    def _bearish_divergence(self, i):
        lb = self.divergence_lookback
        if i < lb + 5:
            return False
        window_high = np.max(self.data.High[i - lb:i - 2])
        current_high = self.data.High[i - 1]
        if current_high <= window_high:
            return False
        idx_high = i - lb + int(np.argmax(self.data.High[i - lb:i - 2]))
        rsi_curr = self.rsi[i - 1]
        rsi_prev = self.rsi[idx_high]
        if np.isnan(rsi_curr) or np.isnan(rsi_prev):
            return False
        if rsi_curr < rsi_prev and rsi_curr > 55:
            return True
        return False

    def next(self):
        i = len(self.data) - 1
        if i < 60:
            return

        price = self.data.Close[-1]
        atr = self.atr[-1]
        if atr <= 0 or np.isnan(atr):
            return

        # ---- LONG ENTRY ----
        if not self.position:
            bull_div = self._bullish_divergence(i)
            stoch_cross_up = (self.stoch_k[-2] < self.stoch_d[-2] and
                              self.stoch_k[-1] > self.stoch_d[-1] and
                              self.stoch_k[-2] < 25)
            stoch_rising = self.stoch_k[-1] > self.stoch_k[-2] and self.stoch_d[-1] > self.stoch_d[-2]
            trend_ok = price > self.ema[-1]

            if bull_div and stoch_cross_up and stoch_rising and trend_ok:
                stop = price - 1.5 * atr
                risk = price - stop
                if risk <= 0:
                    return
                # position sizing: fraction of equity (0 < size < 1)
                size = 0.95
                tp = price + self.risk_reward * risk
                print(f"🌙🚀 LONG SIGNAL! Price={price:.2f} Stop={stop:.2f} TP={tp:.2f} Size={size}")
                self.buy(size=size, sl=stop, tp=tp)
                return

            # ---- SHORT ENTRY ----
            bear_div = self._bearish_divergence(i)
            stoch_cross_down = (self.stoch_k[-2] > self.stoch_d[-2] and
                                self.stoch_k[-1] < self.stoch_d[-1] and
                                self.stoch_k[-2] > 75)
            stoch_falling = self.stoch_k[-1] < self.stoch_k[-2] and self.stoch_d[-1] < self.stoch_d[-2]
            trend_ok_short = price < self.ema[-1]

            if bear_div and stoch_cross_down and stoch_falling and trend_ok_short:
                stop = price + 1.5 * atr
                risk = stop - price
                if risk <= 0:
                    return
                size = 0.95
                tp = price - self.risk_reward * risk
                print(f"🌙🔻 SHORT SIGNAL! Price={price:.2f} Stop={stop:.2f} TP={tp:.2f} Size={size}")
                self.sell(size=size, sl=stop, tp=tp)
                return

        # ---- RSI EXTREME EXITS ----
        if self.position:
            if self.position.is_long and self.rsi[-1] > 75:
                print(f"🌙✨ RSI Overbought exit LONG at {price:.2f}")
                self.position.close()
            elif self.position.is_short and self.rsi[-1] < 25:
                print(f"🌙✨ RSI Oversold exit SHORT at {price:.2f}")
                self.position.close()


bt = Backtest(data, DivergentConvergence, cash=1_000_000, commission=0.002, exclusive_orders=True)
stats = bt.run()
print(stats)
print(stats._strategy)