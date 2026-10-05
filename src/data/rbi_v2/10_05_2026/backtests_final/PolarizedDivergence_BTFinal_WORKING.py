import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

print("🌙✨🚀 Initializing Moon Dev's PolarizedDivergence Backtest...")

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
print(f"🌙 Data loaded: {len(data)} bars ✨")


class PolarizedDivergence(Strategy):
    rsi_period = 14
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    atr_period = 14
    lookback = 100
    pct_low = 20
    pct_high = 80
    stop_atr_mult = 1.5
    tp_atr_mult = 2.5
    risk_pct = 0.01
    time_exit_bars = 50
    ema_period = 200

    def init(self):
        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        volume = pd.Series(self.data.Volume)

        self.rsi = self.I(talib.RSI, close, timeperiod=self.rsi_period)
        self.macd, self.macd_sig, self.macd_hist = self.I(
            talib.MACD, close,
            fastperiod=self.macd_fast,
            slowperiod=self.macd_slow,
            signalperiod=self.macd_signal
        )
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)
        self.ema200 = self.I(talib.EMA, close, timeperiod=self.ema_period)

        # Rolling percentile bands
        self.pct_low_band = self.I(
            lambda s: s.rolling(self.lookback).quantile(self.pct_low / 100.0),
            close
        )
        self.pct_high_band = self.I(
            lambda s: s.rolling(self.lookback).quantile(self.pct_high / 100.0),
            close
        )

        self.entry_bar = 0
        print("🌙 Indicators initialized ✨")

    def next(self):
        i = len(self.data) - 1
        if i < max(self.lookback, self.ema_period, self.atr_period) + 5:
            return

        price = self.data.Close[-1]
        rsi = self.rsi[-1]
        rsi_prev = self.rsi[-2]
        macd = self.macd[-1]
        macd_sig = self.macd_sig[-1]
        macd_hist = self.macd_hist[-1]
        atr = self.atr[-1]
        ema200 = self.ema200[-1]
        p_low = self.pct_low_band[-1]
        p_high = self.pct_high_band[-1]

        vol0 = self.data.Volume[-1]
        vol1 = self.data.Volume[-2]
        vol2 = self.data.Volume[-3]
        vol_decreasing = vol0 < vol1 < vol2

        if np.isnan(atr) or np.isnan(p_low) or np.isnan(p_high) or np.isnan(ema200):
            return

        # ---------- Manage existing positions ----------
        if self.position:
            bars_held = i - self.entry_bar

            if self.position.is_short:
                # Primary exit: price crossed into lower percentile zone
                if price <= p_low:
                    print(f"🌙✨ SHORT EXIT — price hit lower percentile zone @ {price:.2f}")
                    self.position.close()
                    return
                # Time exit
                if bars_held >= self.time_exit_bars:
                    print(f"🌙⏰ SHORT TIME EXIT after {bars_held} bars @ {price:.2f}")
                    self.position.close()
                    return

            elif self.position.is_long:
                if price >= p_high:
                    print(f"🌙✨ LONG EXIT — price hit upper percentile zone @ {price:.2f}")
                    self.position.close()
                    return
                if bars_held >= self.time_exit_bars:
                    print(f"🌙⏰ LONG TIME EXIT after {bars_held} bars @ {price:.2f}")
                    self.position.close()
                    return
            return

        # ---------- Entry logic ----------
        rsi_bearish = rsi < 50 and rsi < rsi_prev
        rsi_bullish = rsi > 50 and rsi > rsi_prev
        macd_bullish = (macd > macd_sig) or (macd_hist > 0)
        macd_bearish = (macd < macd_sig) or (macd_hist < 0)

        risk_amount = self.equity * self.risk_pct
        stop_dist = self.stop_atr_mult * atr
        if stop_dist <= 0:
            return
        size = int(round(risk_amount / stop_dist))
        if size <= 0:
            return

        # Short setup
        if rsi_bearish and macd_bullish and vol_decreasing and price > p_low:
            if price < ema200:
                sl = price + self.stop_atr_mult * atr
                tp_atr = price - self.tp_atr_mult * atr
                tp = max(tp_atr, p_low)  # closer of the two
                print(f"🌙🔻 SHORT ENTRY @ {price:.2f} | RSI={rsi:.2f} MACD={macd:.4f} ATR={atr:.2f} size={size} SL={sl:.2f} TP={tp:.2f}")
                self.sell(size=size, sl=sl, tp=tp)
                self.entry_bar = i
                return

        # Long setup
        if rsi_bullish and macd_bearish and vol_decreasing and price < p_high:
            if price > ema200:
                sl = price - self.stop_atr_mult * atr
                tp_atr = price + self.tp_atr_mult * atr
                tp = min(tp_atr, p_high)
                print(f"🌙🔺 LONG ENTRY @ {price:.2f} | RSI={rsi:.2f} MACD={macd:.4f} ATR={atr:.2f} size={size} SL={sl:.2f} TP={tp:.2f}")
                self.buy(size=size, sl=sl, tp=tp)
                self.entry_bar = i
                return


bt = Backtest(
    data,
    PolarizedDivergence,
    cash=1_000_000,
    commission=0.0002,
    exclusive_orders=True
)

print("🌙🚀 Running backtest...")
stats = bt.run()
print(stats)
print(stats._strategy)