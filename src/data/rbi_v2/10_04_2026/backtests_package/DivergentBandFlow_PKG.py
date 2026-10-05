import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

print("🌙 Moon Dev's DivergentBandFlow Backtest Initializing... ✨")

data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'datetime': 'Date',
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})
data['Date'] = pd.to_datetime(data['Date'])
data = data.set_index('Date')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

print(f"🚀 Data loaded: {len(data)} bars from {data.index[0]} to {data.index[-1]} 🌙")


class DivergentBandFlow(Strategy):
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    rsi_period = 14
    bb_period = 20
    bb_std = 2
    ema_period = 20
    atr_period = 14
    swing_lookback = 10
    risk_pct = 0.02
    rr_ratio = 2.0

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low

        print("🌙 Calculating indicators... ✨")

        self.macd, self.macd_signal, self.macd_hist = self.I(
            talib.MACD, close,
            fastperiod=self.macd_fast,
            slowperiod=self.macd_slow,
            signalperiod=self.macd_signal
        )

        self.rsi = self.I(talib.RSI, close, timeperiod=self.rsi_period)

        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS, close,
            timeperiod=self.bb_period,
            nbdevup=self.bb_std,
            nbdevdn=self.bb_std
        )

        self.ema = self.I(talib.EMA, close, timeperiod=self.ema_period)
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        self.swing_high = self.I(talib.MAX, high, timeperiod=self.swing_lookback)
        self.swing_low = self.I(talib.MIN, low, timeperiod=self.swing_lookback)

        # Band Width
        self.bw = self.I(
            lambda u, m, l: (u - l) / np.where(m > 0, m, np.nan),
            self.bb_upper, self.bb_middle, self.bb_lower
        )
        self.bw_avg = self.I(talib.SMA, self.bw, timeperiod=self.bb_period)

        print("🌙 Indicators ready! 🚀")

        self.pending_entry = None
        self.entry_price = None
        self.stop_price = None
        self.tp_price = None

    def _bullish_divergence(self, i):
        if i < self.swing_lookback * 2:
            return False
        lookback = self.swing_lookback
        price_lower_low = self.data.Low[i] < self.swing_low[i - 1]
        macd_higher_low = self.macd_hist[i] > self.macd_hist[i - lookback]
        return price_lower_low and macd_higher_low

    def _bearish_divergence(self, i):
        if i < self.swing_lookback * 2:
            return False
        lookback = self.swing_lookback
        price_higher_high = self.data.High[i] > self.swing_high[i - 1]
        macd_lower_high = self.macd_hist[i] < self.macd_hist[i - lookback]
        return price_higher_high and macd_lower_high

    def next(self):
        i = len(self.data) - 1
        if i < 50:
            return

        price = self.data.Close[-1]

        # Manage open position exits
        if self.position:
            if self.position.is_long:
                bw_expansion = self.bw[-1] > 1.5 * self.bw_avg[-1] if not np.isnan(self.bw_avg[-1]) else False
                touch_upper = self.data.High[-1] >= self.bb_upper[-1]
                close_below_mid = self.data.Close[-1] < self.bb_middle[-1]

                if (bw_expansion and touch_upper) or close_below_mid:
                    print(f"🌙 EXIT LONG @ {price:.2f} | BW_exp={bw_expansion} touch_up={touch_upper} below_mid={close_below_mid} ✨")
                    self.position.close()
                    return

                if self.stop_price and self.data.Low[-1] <= self.stop_price:
                    print(f"🛑 STOP LOSS LONG @ {self.stop_price:.2f} 🌙")
                    self.position.close()
                    return

                if self.tp_price and self.data.High[-1] >= self.tp_price:
                    print(f"🎯 TAKE PROFIT LONG @ {self.tp_price:.2f} 🚀")
                    self.position.close()
                    return

            elif self.position.is_short:
                bw_expansion = self.bw[-1] > 1.5 * self.bw_avg[-1] if not np.isnan(self.bw_avg[-1]) else False
                touch_lower = self.data.Low[-1] <= self.bb_lower[-1]
                close_above_mid = self.data.Close[-1] > self.bb_middle[-1]

                if (bw_expansion and touch_lower) or close_above_mid:
                    print(f"🌙 EXIT SHORT @ {price:.2f} | BW_exp={bw_expansion} touch_lo={touch_lower} above_mid={close_above_mid} ✨")
                    self.position.close()
                    return

                if self.stop_price and self.data.High[-1] >= self.stop_price:
                    print(f"🛑 STOP LOSS SHORT @ {self.stop_price:.2f} 🌙")
                    self.position.close()
                    return

                if self.tp_price and self.data.Low[-1] <= self.tp_price:
                    print(f"🎯 TAKE PROFIT SHORT @ {self.tp_price:.2f} 🚀")
                    self.position.close()
                    return
            return

        # Trending filter: BW above its average
        if np.isnan(self.bw_avg[-1]) or self.bw[-1] <= self.bw_avg[-1]:
            return

        # LONG ENTRY
        bull_div = self._bullish_divergence(i)
        rsi_recovering = self.rsi[-2] < 30 and self.rsi[-1] >= 30
        rsi_rising = self.rsi[-1] > self.rsi[-2] and self.rsi[-1] < 50
        above_ema = self.data.Close[-1] > self.ema[-1]

        if bull_div and (rsi_recovering or rsi_rising) and above_ema:
            stop = min(self.swing_low[-1], price - 1.5 * self.atr[-1])
            risk = price - stop
            if risk <= 0:
                return
            tp = price + self.rr_ratio * risk
            size = int(round(1_000_000 / price))
            if size < 1:
                size = 1
            print(f"🚀 LONG ENTRY @ {price:.2f} | RSI={self.rsi[-1]:.1f} | SL={stop:.2f} TP={tp:.2f} size={size} 🌙")
            self.buy(size=size)
            self.stop_price = stop
            self.tp_price = tp
            return

        # SHORT ENTRY
        bear_div = self._bearish_divergence(i)
        rsi_recovering_down = self.rsi[-2] > 70 and self.rsi[-1] <= 70
        rsi_falling = self.rsi[-1] < self.rsi[-2] and self.rsi[-1] > 50
        below_ema = self.data.Close[-1] < self.ema[-1]

        if bear_div and (rsi_recovering_down or rsi_falling) and below_ema:
            stop = max(self.swing_high[-1], price + 1.5 * self.atr[-1])
            risk = stop - price
            if risk <= 0:
                return
            tp = price - self.rr_ratio * risk
            size = int(round(1_000_000 / price))
            if size < 1:
                size = 1
            print(f"🔻 SHORT ENTRY @ {price:.2f} | RSI={self.rsi[-1]:.1f} | SL={stop:.2f} TP={tp:.2f} size={size} 🌙")
            self.sell(size=size)
            self.stop_price = stop
            self.tp_price = tp
            return


bt = Backtest(data, DivergentBandFlow, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)