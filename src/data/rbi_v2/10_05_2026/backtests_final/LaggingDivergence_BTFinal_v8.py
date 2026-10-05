import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Load data
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
print(f"🌙 Moon Dev Data Loaded: {len(data)} bars ✨")
print(f"🚀 Date range: {data.index[0]} to {data.index[-1]}")


class LaggingDivergence(Strategy):
    rsi_period = 14
    rsi_ob = 70
    rsi_os = 30
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    atr_period = 14
    chikou_lag = 26
    risk_pct = 0.01
    atr_stop_mult = 2.0
    trail_trigger = 1.5
    trail_atr = 1.0
    min_bars_after_exit = 5
    skip_opening_bars = 2  # 30 min on 15m

    def init(self):
        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)

        self.rsi = self.I(talib.RSI, close, timeperiod=self.rsi_period)

        # Compute MACD histogram via talib and wrap with self.I
        def _macd_hist(close_series):
            macd, macdsig, macdhist = talib.MACD(
                close_series,
                fastperiod=self.macd_fast,
                slowperiod=self.macd_slow,
                signalperiod=self.macd_signal,
            )
            return macdhist

        self.macd_hist = self.I(_macd_hist, close)

        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        # Track state
        self.last_exit_bar = -999
        self.trail_active = False
        self.trail_stop = None
        self.entry_price_val = None
        self.entry_atr = None
        self.trade_dir = 0  # 1 long, -1 short
        self._bars_today = 0
        self._last_date = None
        self._last_trade_size = 0

        print("🌙✨ LaggingDivergence Strategy Initialized 🚀")

    def next(self):
        i = len(self.data) - 1
        lag = self.chikou_lag

        # Skip opening bars of session (first N bars of each day)
        if i < lag + 5:
            return

        # Opening noise filter: skip first bars of the day
        current_time = self.data.index[i]
        if self._last_date is None or current_time.date() != self._last_date:
            self._bars_today = 0
            self._last_date = current_time.date()
        else:
            self._bars_today += 1
        if self._bars_today < self.skip_opening_bars:
            return

        # Cooldown after exit
        if i - self.last_exit_bar < self.min_bars_after_exit:
            return

        close = self.data.Close[i]
        high = self.data.High[i]
        low = self.data.Low[i]
        rsi = self.rsi[i]
        atr = self.atr[i]
        macd_h = self.macd_hist[i]

        if np.isnan(rsi) or np.isnan(atr) or np.isnan(macd_h):
            return

        # Chikou span = current close (plotted back 26)
        chikou_now = close
        chikou_prev = self.data.Close[i - lag]

        price_now_high = high
        price_prev_high = self.data.High[i - lag]
        price_now_low = low
        price_prev_low = self.data.Low[i - lag]

        # In position: manage exits
        if self.position:
            self._manage_position(i, atr, macd_h)
            return

        # Check entries
        bearish = (price_now_high > price_prev_high and
                   chikou_now < chikou_prev and
                   rsi > self.rsi_ob)

        bullish = (price_now_low < price_prev_low and
                   chikou_now > chikou_prev and
                   rsi < self.rsi_os)

        if bearish:
            stop_price = close + self.atr_stop_mult * atr
            risk_per_unit = stop_price - close
            if risk_per_unit > 0:
                size = int(round((self.equity * self.risk_pct) / risk_per_unit))
                if size > 0:
                    self.sell(size=size)
                    self.entry_price_val = close
                    self.entry_atr = atr
                    self.trade_dir = -1
                    self.trail_active = False
                    self.trail_stop = None
                    self._last_trade_size = size
                    print(f"🌙🔻 SHORT entry @ {close:.2f} | RSI={rsi:.1f} | ATR={atr:.2f} | size={size} ✨")

        elif bullish:
            stop_price = close - self.atr_stop_mult * atr
            risk_per_unit = close - stop_price
            if risk_per_unit > 0:
                size = int(round((self.equity * self.risk_pct) / risk_per_unit))
                if size > 0:
                    self.buy(size=size)
                    self.entry_price_val = close
                    self.entry_atr = atr
                    self.trade_dir = 1
                    self.trail_active = False
                    self.trail_stop = None
                    self._last_trade_size = size
                    print(f"🌙🔺 LONG entry @ {close:.2f} | RSI={rsi:.1f} | ATR={atr:.2f} | size={size} ✨")

    def _manage_position(self, i, atr, macd_h):
        close = self.data.Close[i]
        high = self.data.High[i]
        low = self.data.Low[i]
        lag = self.chikou_lag

        if i < lag + 1:
            return

        # Pull entry price from last trade (Position has no entry_price attribute)
        if len(self.trades) > 0:
            entry = self.trades[-1].entry_price
        else:
            entry = self.entry_price_val

        entry_atr = self.entry_atr

        if entry is None or entry_atr is None:
            return

        if self.trade_dir == 1:  # Long
            # Hard stop
            hard_stop = entry - self.atr_stop_mult * entry_atr
            if low <= hard_stop:
                self.position.close()
                self.last_exit_bar = i
                print(f"🌙🛑 LONG hard stop @ {hard_stop:.2f} 🚀")
                self._reset()
                return

            # Trailing stop
            if not self.trail_active and high >= entry + self.trail_trigger * entry_atr:
                self.trail_active = True
                self.trail_stop = high - self.trail_atr * atr
                print(f"🌙✨ LONG trailing activated @ {self.trail_stop:.2f} 🚀")

            if self.trail_active:
                new_trail = max(self.trail_stop, high - self.trail_atr * atr)
                self.trail_stop = new_trail
                if low <= self.trail_stop:
                    self.position.close()
                    self.last_exit_bar = i
                    print(f"🌙💰 LONG trail exit @ {self.trail_stop:.2f} ✨")
                    self._reset()
                    return

            # Bearish MACD histogram divergence exit
            price_now_high = high
            price_prev_high = self.data.High[i - lag]
            macd_now = macd_h
            macd_prev = self.macd_hist[i - lag]
            if (price_now_high > price_prev_high and
                not np.isnan(macd_prev) and macd_now < macd_prev):
                self.position.close()
                self.last_exit_bar = i
                print(f"🌙📉 LONG exit: MACD hist bearish divergence @ {close:.2f} 🚀")
                self._reset()

        elif self.trade_dir == -1:  # Short
            hard_stop = entry + self.atr_stop_mult * entry_atr
            if high >= hard_stop:
                self.position.close()
                self.last_exit_bar = i
                print(f"🌙🛑 SHORT hard stop @ {hard_stop:.2f} 🚀")
                self._reset()
                return

            if not self.trail_active and low <= entry - self.trail_trigger * entry_atr:
                self.trail_active = True
                self.trail_stop = low + self.trail_atr * atr
                print(f"🌙✨ SHORT trailing activated @ {self.trail_stop:.2f} 🚀")

            if self.trail_active:
                new_trail = min(self.trail_stop, low + self.trail_atr * atr)
                self.trail_stop = new_trail
                if high >= self.trail_stop:
                    self.position.close()
                    self.last_exit_bar = i
                    print(f"🌙💰 SHORT trail exit @ {self.trail_stop:.2f} ✨")
                    self._reset()
                    return

            price_now_low = low
            price_prev_low = self.data.Low[i - lag]
            macd_now = macd_h
            macd_prev = self.macd_hist[i - lag]
            if (price_now_low < price_prev_low and
                not np.isnan(macd_prev) and macd_now > macd_prev):
                self.position.close()
                self.last_exit_bar = i
                print(f"🌙📈 SHORT exit: MACD hist bullish divergence @ {close:.2f} 🚀")
                self._reset()

    def _reset(self):
        self.trail_active = False
        self.trail_stop = None
        self.entry_price_val = None
        self.entry_atr = None
        self.trade_dir = 0


bt = Backtest(data, LaggingDivergence, cash=1_000_000, commission=0.001, exclusive_orders=False)
stats = bt.run()
print(stats)
print(stats._strategy)