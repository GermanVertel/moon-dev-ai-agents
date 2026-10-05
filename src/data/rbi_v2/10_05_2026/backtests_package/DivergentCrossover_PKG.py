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

print("🌙 Moon Dev Backtest Engine Initialized ✨")
print(f"📊 Data loaded: {len(data)} bars from {data.index[0]} to {data.index[-1]}")


class DivergentCrossover(Strategy):
    ema_fast = 20
    ema_slow = 50
    rsi_period = 14
    rsi_upper = 70
    rsi_exit = 78
    vol_period = 20
    vol_lower = 0.8
    vol_upper = 1.2
    div_lookback = 15
    swing_window = 5
    atr_period = 14
    atr_mult = 2.0
    risk_pct = 0.02

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        self.ema20 = self.I(talib.EMA, close, timeperiod=self.ema_fast)
        self.ema50 = self.I(talib.EMA, close, timeperiod=self.ema_slow)
        self.rsi = self.I(talib.RSI, close, timeperiod=self.rsi_period)
        self.avg_vol = self.I(talib.SMA, volume, timeperiod=self.vol_period)
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)
        self.swing_low = self.I(talib.MIN, low, timeperiod=self.swing_window)

        print("🌙 Indicators initialized: EMA20, EMA50, RSI, AvgVol, ATR, SwingLow ✨")

    def bullish_divergence(self):
        i = len(self.data) - 1
        if i < self.div_lookback + self.swing_window + 2:
            return False

        # current swing low
        cur_low = self.data.Low[i]
        cur_rsi = self.rsi[i]

        # find prior swing low within lookback
        start = i - self.div_lookback
        end = i - self.swing_window
        if end <= start:
            return False

        segment_low = self.data.Low[start:end]
        if len(segment_low) == 0:
            return False

        prior_idx = int(np.argmin(segment_low)) + start
        prior_low = self.data.Low[prior_idx]
        prior_rsi = self.rsi[prior_idx]

        # price lower low, RSI higher low
        if cur_low < prior_low and cur_rsi > prior_rsi:
            return True
        return False

    def next(self):
        price = self.data.Close[-1]
        i = len(self.data) - 1

        if i < 2:
            return

        ema20 = self.ema20[-1]
        ema50 = self.ema50[-1]
        ema20_prev = self.ema20[-2]
        ema50_prev = self.ema50[-2]
        rsi = self.rsi[-1]
        vol = self.data.Volume[-1]
        avg_vol = self.avg_vol[-1]
        atr = self.atr[-1]

        # exits
        if self.position:
            # trend reversal exit
            if ema20 < ema50:
                print(f"🌙 Trend reversal exit at {price:.2f} ✨")
                self.position.close()
                return
            # RSI overbought exit
            if rsi > self.rsi_exit:
                print(f"🌙 RSI overbought exit at {price:.2f} (RSI={rsi:.1f}) ✨")
                self.position.close()
                return
            # trailing stop via ATR
            if self.position.is_long:
                stop = price - self.atr_mult * atr
                if self.data.Low[-1] <= stop:
                    print(f"🌙 ATR trailing stop hit at {price:.2f} 🛑")
                    self.position.close()
                    return
            return

        # entry conditions - bullish EMA crossover (no backtesting.lib!)
        crossover = ema20 > ema50 and ema20_prev <= ema50_prev
        if not crossover:
            return

        if rsi >= self.rsi_upper:
            return

        if avg_vol <= 0:
            return
        vol_ratio = vol / avg_vol
        if vol_ratio < self.vol_lower or vol_ratio > self.vol_upper:
            return

        if not self.bullish_divergence():
            return

        # risk management: stop at swing low or 2% below, whichever tighter
        swing_low = self.swing_low[-1]
        pct_stop = price * 0.98
        stop_price = max(swing_low, pct_stop)

        risk_per_unit = price - stop_price
        if risk_per_unit <= 0:
            return

        equity = self.equity
        risk_amount = equity * self.risk_pct
        size = int(round(risk_amount / risk_per_unit))
        if size < 1:
            size = 1
        # cap size to available equity
        max_size = int(equity / price)
        if size > max_size:
            size = max_size
        if size < 1:
            return

        print(f"🌙🚀 LONG ENTRY at {price:.2f} | EMA20={ema20:.2f} EMA50={ema50:.2f} RSI={rsi:.1f} VolRatio={vol_ratio:.2f} Size={size} Stop={stop_price:.2f} ✨")
        self.buy(size=size)


bt = Backtest(data, DivergentCrossover, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)