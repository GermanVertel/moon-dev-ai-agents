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

# 🌙 Ensure all numeric columns are float64 for talib compatibility
for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    data[col] = data[col].astype(np.float64)

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

        # 🌙 Wrap talib calls to guarantee float64 input arrays
        def _ema(x, timeperiod):
            return talib.EMA(np.asarray(x, dtype=np.float64), timeperiod=timeperiod)

        def _rsi(x, timeperiod):
            return talib.RSI(np.asarray(x, dtype=np.float64), timeperiod=timeperiod)

        def _sma(x, timeperiod):
            return talib.SMA(np.asarray(x, dtype=np.float64), timeperiod=timeperiod)

        def _atr(h, l, c, timeperiod):
            return talib.ATR(
                np.asarray(h, dtype=np.float64),
                np.asarray(l, dtype=np.float64),
                np.asarray(c, dtype=np.float64),
                timeperiod=timeperiod,
            )

        def _min(x, timeperiod):
            return talib.MIN(np.asarray(x, dtype=np.float64), timeperiod=timeperiod)

        self.ema20 = self.I(_ema, close, timeperiod=self.ema_fast, name='EMA20')
        self.ema50 = self.I(_ema, close, timeperiod=self.ema_slow, name='EMA50')
        self.rsi = self.I(_rsi, close, timeperiod=self.rsi_period, name='RSI')
        self.avg_vol = self.I(_sma, volume, timeperiod=self.vol_period, name='AvgVol')
        self.atr = self.I(_atr, high, low, close, timeperiod=self.atr_period, name='ATR')
        self.swing_low = self.I(_min, low, timeperiod=self.swing_window, name='SwingLow')

        print("🌙 Indicators initialized: EMA20, EMA50, RSI, AvgVol, ATR, SwingLow ✨")

    def bullish_divergence(self):
        i = len(self.data) - 1
        if i < self.div_lookback + self.swing_window + 2:
            return False

        cur_low = self.data.Low[i]
        cur_rsi = self.rsi[i]

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

        if np.isnan(prior_rsi) or np.isnan(cur_rsi):
            return False

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

        # 🌙 Guard against NaN indicator values early on
        if (np.isnan(ema20) or np.isnan(ema50) or np.isnan(ema20_prev)
                or np.isnan(ema50_prev) or np.isnan(rsi) or np.isnan(avg_vol)
                or np.isnan(atr)):
            return

        # exits
        if self.position:
            if ema20 < ema50:
                print(f"🌙 Trend reversal exit at {price:.2f} ✨")
                self.position.close()
                return
            if rsi > self.rsi_exit:
                print(f"🌙 RSI overbought exit at {price:.2f} (RSI={rsi:.1f}) ✨")
                self.position.close()
                return
            if self.position.is_long:
                stop = price - self.atr_mult * atr
                if self.data.Low[-1] <= stop:
                    print(f"🌙 ATR trailing stop hit at {price:.2f} 🛑")
                    self.position.close()
                    return
            return

        # entry conditions - bullish EMA crossover
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

        swing_low = self.swing_low[-1]
        if np.isnan(swing_low):
            return
        pct_stop = price * 0.98
        stop_price = max(swing_low, pct_stop)

        risk_per_unit = price - stop_price
        if risk_per_unit <= 0:
            return

        # 🌙 Use fractional position sizing (percentage of equity) to avoid
        # integer-size / insufficient-cash issues that prevented execution.
        # Risk-based sizing converted to a fraction of equity.
        stop_distance_pct = risk_per_unit / price
        if stop_distance_pct <= 0:
            return
        size_frac = self.risk_pct / stop_distance_pct
        # Cap fraction at 0.95 to keep it a valid fraction and leave cash
        if size_frac > 0.95:
            size_frac = 0.95
        if size_frac <= 0:
            return

        print(f"🌙🚀 LONG ENTRY at {price:.2f} | EMA20={ema20:.2f} EMA50={ema50:.2f} RSI={rsi:.1f} VolRatio={vol_ratio:.2f} SizeFrac={size_frac:.4f} Stop={stop_price:.2f} ✨")
        self.buy(size=size_frac, sl=stop_price)


bt = Backtest(data, DivergentCrossover, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)