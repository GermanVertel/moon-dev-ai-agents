import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

print("🌙 Moon Dev VolatilityDivergence Backtest Loading... ✨🚀")

data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Map to backtesting requirements
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume',
    'datetime': 'Date'
})

if 'Date' in data.columns:
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.set_index('Date')

data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

print(f"🌙 Data loaded: {len(data)} bars ✨")


class VolatilityDivergence(Strategy):
    rsi_period = 14
    atr_fast = 14
    atr_slow = 50
    ema_period = 200
    pivot_window = 5
    risk_pct = 0.01
    rr_ratio = 2.0
    atr_stop_buffer = 0.5
    time_exit_bars = 10

    def init(self):
        print("🌙 Initializing indicators... ✨")

        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=self.rsi_period)
        self.atr_fast = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close,
                               timeperiod=self.atr_fast)
        self.atr_slow = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close,
                               timeperiod=self.atr_slow)
        self.ema200 = self.I(talib.EMA, self.data.Close, timeperiod=self.ema_period)

        # Pivot highs/lows via rolling max/min (fractal-like)
        self.pivot_high = self.I(talib.MAX, self.data.High, timeperiod=self.pivot_window)
        self.pivot_low = self.I(talib.MIN, self.data.Low, timeperiod=self.pivot_window)

        self.entry_bar = None
        self.entry_price = None
        self.stop_price = None
        self.tp_price = None

    def next(self):
        i = len(self.data) - 1
        if i < max(self.ema_period, self.atr_slow, self.pivot_window * 3) + 5:
            return

        price = self.data.Close[-1]
        rsi_now = self.rsi[-1]
        rsi_prev = self.rsi[-self.pivot_window - 1]
        atr_f = self.atr_fast[-1]
        atr_s = self.atr_slow[-1]

        # Volatility regime: normal-to-high
        vol_ok = atr_f > atr_s * 0.9 and atr_s > 0

        # Manage existing position
        if self.position:
            bars_held = i - self.entry_bar if self.entry_bar else 0

            if self.position.is_long:
                # Chandelier-style trail
                new_stop = self.data.High[-1] - 3 * atr_f
                if new_stop > self.stop_price:
                    self.stop_price = new_stop

                if self.data.Low[-1] <= self.stop_price:
                    print(f"🌙💥 Long SL hit @ {self.stop_price:.2f}")
                    self.position.close()
                    self.entry_bar = None
                    return
                if self.data.High[-1] >= self.tp_price:
                    print(f"🌙🎯 Long TP hit @ {self.tp_price:.2f} 🚀")
                    self.position.close()
                    self.entry_bar = None
                    return
                if bars_held >= self.time_exit_bars and self.data.Close[-1] < self.entry_price:
                    print(f"🌙⏰ Long time exit @ {price:.2f}")
                    self.position.close()
                    self.entry_bar = None
                    return

            elif self.position.is_short:
                new_stop = self.data.Low[-1] + 3 * atr_f
                if new_stop < self.stop_price:
                    self.stop_price = new_stop

                if self.data.High[-1] >= self.stop_price:
                    print(f"🌙💥 Short SL hit @ {self.stop_price:.2f}")
                    self.position.close()
                    self.entry_bar = None
                    return
                if self.data.Low[-1] <= self.tp_price:
                    print(f"🌙🎯 Short TP hit @ {self.tp_price:.2f} 🚀")
                    self.position.close()
                    self.entry_bar = None
                    return
                if bars_held >= self.time_exit_bars and self.data.Close[-1] > self.entry_price:
                    print(f"🌙⏰ Short time exit @ {price:.2f}")
                    self.position.close()
                    self.entry_bar = None
                    return
            return

        if not vol_ok:
            return

        # Bullish divergence: price lower low, RSI higher low
        price_ll = self.data.Low[-1] < self.pivot_low[-self.pivot_window - 1]
        rsi_hl = rsi_now > rsi_prev
        prev_price_low = self.data.Low[-self.pivot_window - 1]
        near_swing_low = self.data.Low[-1] <= prev_price_low * 1.005

        bullish_div = price_ll and rsi_hl and rsi_now < 45 and near_swing_low

        # Bearish divergence: price higher high, RSI lower high
        price_hh = self.data.High[-1] > self.pivot_high[-self.pivot_window - 1]
        rsi_lh = rsi_now < rsi_prev
        prev_price_high = self.data.High[-self.pivot_window - 1]
        near_swing_high = self.data.High[-1] >= prev_price_high * 0.995

        bearish_div = price_hh and rsi_lh and rsi_now > 55 and near_swing_high

        # Trend context
        trend_up = price > self.ema200[-1]
        trend_down = price < self.ema200[-1]

        # Long entry
        if bullish_div and not trend_down and self.data.Close[-1] > self.data.High[-2]:
            swing_low = self.data.Low[-self.pivot_window:].min()
            stop = swing_low - self.atr_stop_buffer * atr_f
            risk = price - stop
            if risk <= 0:
                return
            tp = price + self.rr_ratio * risk

            equity = self.equity
            risk_amount = equity * self.risk_pct
            size = risk_amount / risk
            size = int(round(float(size)))
            if size <= 0:
                return

            print(f"🌙🚀 BULLISH DIVERGENCE LONG @ {price:.2f} | SL {stop:.2f} | TP {tp:.2f} | size {size}")
            self.buy(size=size)
            self.entry_bar = i
            self.entry_price = price
            self.stop_price = stop
            self.tp_price = tp

        # Short entry
        elif bearish_div and not trend_up and self.data.Close[-1] < self.data.Low[-2]:
            swing_high = self.data.High[-self.pivot_window:].max()
            stop = swing_high + self.atr_stop_buffer * atr_f
            risk = stop - price
            if risk <= 0:
                return
            tp = price - self.rr_ratio * risk

            equity = self.equity
            risk_amount = equity * self.risk_pct
            size = risk_amount / risk
            size = int(round(float(size)))
            if size <= 0:
                return

            print(f"🌙🔻 BEARISH DIVERGENCE SHORT @ {price:.2f} | SL {stop:.2f} | TP {tp:.2f} | size {size}")
            self.sell(size=size)
            self.entry_bar = i
            self.entry_price = price
            self.stop_price = stop
            self.tp_price = tp


bt = Backtest(data, VolatilityDivergence, cash=1_000_000, commission=0.0002)
stats = bt.run()
print(stats)
print(stats._strategy)