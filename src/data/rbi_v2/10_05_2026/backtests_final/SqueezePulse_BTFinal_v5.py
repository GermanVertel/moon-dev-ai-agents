import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# 🌙 Moon Dev SqueezePulse Backtest ✨🚀

DATA_PATH = '/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv'

print("🌙 Moon Dev loading data from:", DATA_PATH)

data = pd.read_csv(DATA_PATH)
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
print("🌙 Moon Dev data ready! Shape:", data.shape)


class SqueezePulse(Strategy):
    bb_period = 20
    bb_std = 2.0
    pct_lookback = 150
    pct_threshold = 20
    vol_mult = 1.5
    vol_ma_period = 20
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    ck_period = 10
    ck_mult = 1.5
    atr_period = 14
    risk_pct = 0.01
    time_stop_bars = 18

    def init(self):
        # 🌙 Use self.data directly (backtesting.py provides pandas Series)
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        # 🌙 Bollinger Bands
        def bb_bands(close):
            close = np.asarray(close, dtype=np.float64)
            upper, mid, lower = talib.BBANDS(
                close, timeperiod=self.bb_period,
                nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0
            )
            return upper, mid, lower

        self.bb_upper, self.bb_mid, self.bb_lower = self.I(
            bb_bands, close
        )

        # ✨ BB Width
        def bb_width(upper, mid, lower):
            upper = np.asarray(upper, dtype=np.float64)
            mid = np.asarray(mid, dtype=np.float64)
            lower = np.asarray(lower, dtype=np.float64)
            with np.errstate(divide='ignore', invalid='ignore'):
                out = (upper - lower) / mid
            out = np.where(mid == 0, np.nan, out)
            return out
        self.bbw = self.I(bb_width, self.bb_upper, self.bb_mid, self.bb_lower)

        # 🚀 Percentile rank of BB Width over lookback
        def pct_rank(arr, lookback):
            arr = np.asarray(arr, dtype=np.float64)
            out = np.full(len(arr), np.nan)
            for i in range(lookback, len(arr)):
                window = arr[i - lookback:i + 1]
                window = window[~np.isnan(window)]
                if len(window) > 0 and not np.isnan(arr[i]):
                    out[i] = (np.sum(window <= arr[i]) / len(window)) * 100.0
            return out
        self.bbw_pct = self.I(pct_rank, self.bbw, self.pct_lookback)

        # 🌙 Volume MA
        self.vol_ma = self.I(talib.SMA, volume, timeperiod=self.vol_ma_period)

        # ✨ MACD
        def macd_func(close):
            close = np.asarray(close, dtype=np.float64)
            macd, sig, hist = talib.MACD(
                close,
                fastperiod=self.macd_fast,
                slowperiod=self.macd_slow,
                signalperiod=self.macd_signal
            )
            return macd, sig, hist

        self.macd, self.macd_signal_line, self.macd_hist = self.I(macd_func, close)

        # 🚀 ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        # 🌙 Chande Kroll Stop
        def chande_kroll(high, low, close, atr, period, mult):
            high = np.asarray(high, dtype=np.float64)
            low = np.asarray(low, dtype=np.float64)
            close = np.asarray(close, dtype=np.float64)
            atr = np.asarray(atr, dtype=np.float64)
            n = len(close)
            long_stop = np.full(n, np.nan)
            short_stop = np.full(n, np.nan)
            highest = np.full(n, np.nan)
            lowest = np.full(n, np.nan)
            for i in range(n):
                if i < period or np.isnan(atr[i]):
                    continue
                highest[i] = np.max(high[i - period + 1:i + 1])
                lowest[i] = np.min(low[i - period + 1:i + 1])
                long_stop[i] = highest[i] - mult * atr[i]
                short_stop[i] = lowest[i] + mult * atr[i]
            # trail
            for i in range(1, n):
                if not np.isnan(long_stop[i]) and not np.isnan(long_stop[i - 1]):
                    long_stop[i] = max(long_stop[i], long_stop[i - 1])
                if not np.isnan(short_stop[i]) and not np.isnan(short_stop[i - 1]):
                    short_stop[i] = min(short_stop[i], short_stop[i - 1])
            return long_stop, short_stop

        self.ck_long, self.ck_short = self.I(
            chande_kroll, high, low, close, self.atr,
            self.ck_period, self.ck_mult
        )

        self.entry_bar = 0
        self.entry_price = 0
        self.stop_price = 0
        self.highest_since_entry = 0
        self.lowest_since_entry = 0

    def next(self):
        if len(self.data) < self.pct_lookback + 5:
            return

        price = self.data.Close[-1]
        prev_high = self.data.High[-2]
        prev_low = self.data.Low[-2]
        vol = self.data.Volume[-1]

        bbw_pct = self.bbw_pct[-1]
        upper = self.bb_upper[-1]
        lower = self.bb_lower[-1]
        vol_ma = self.vol_ma[-1]
        macd = self.macd[-1]
        macd_sig = self.macd_signal_line[-1]
        macd_prev = self.macd[-2]
        macd_sig_prev = self.macd_signal_line[-2]
        ck_long = self.ck_long[-1]
        ck_short = self.ck_short[-1]

        if (np.isnan(bbw_pct) or np.isnan(vol_ma) or np.isnan(macd)
                or np.isnan(macd_sig) or np.isnan(macd_prev)
                or np.isnan(macd_sig_prev) or np.isnan(ck_long)
                or np.isnan(upper) or np.isnan(lower)):
            return

        squeeze = bbw_pct <= self.pct_threshold
        vol_surge = vol >= self.vol_mult * vol_ma

        macd_cross_up = (macd_prev < macd_sig_prev) and (macd > macd_sig)
        macd_cross_dn = (macd_prev > macd_sig_prev) and (macd < macd_sig)
        macd_rising = macd > macd_sig and macd > macd_prev
        macd_falling = macd < macd_sig and macd < macd_prev

        long_trigger = price > upper and price > prev_high
        short_trigger = price < lower and price < prev_low

        # 🌙 Manage open position
        if self.position:
            if self.position.is_long:
                self.highest_since_entry = max(self.highest_since_entry, self.data.High[-1])
                trail = max(ck_long, self.stop_price)
                self.stop_price = trail
                bars_held = len(self.data) - self.entry_bar

                if price < self.stop_price:
                    print(f"🌙 Moon Dev EXIT LONG @ stop {self.stop_price:.2f} | price {price:.2f} 🛑")
                    self.position.close()
                    return
                if bars_held >= self.time_stop_bars and self.data.High[-1] <= self.highest_since_entry:
                    print(f"🌙 Moon Dev TIME STOP LONG after {bars_held} bars ⏰")
                    self.position.close()
                    return
                if macd_cross_dn and price < upper:
                    print(f"🌙 Moon Dev REVERSAL EXIT LONG ✨")
                    self.position.close()
                    return
            elif self.position.is_short:
                self.lowest_since_entry = min(self.lowest_since_entry, self.data.Low[-1])
                trail = min(ck_short, self.stop_price)
                self.stop_price = trail
                bars_held = len(self.data) - self.entry_bar

                if price > self.stop_price:
                    print(f"🌙 Moon Dev EXIT SHORT @ stop {self.stop_price:.2f} | price {price:.2f} 🛑")
                    self.position.close()
                    return
                if bars_held >= self.time_stop_bars and self.data.Low[-1] >= self.lowest_since_entry:
                    print(f"🌙 Moon Dev TIME STOP SHORT after {bars_held} bars ⏰")
                    self.position.close()
                    return
                if macd_cross_up and price > lower:
                    print(f"🌙 Moon Dev REVERSAL EXIT SHORT ✨")
                    self.position.close()
                    return
            return

        # 🌙 Entry logic
        if squeeze and vol_surge:
            if long_trigger and (macd_cross_up or macd_rising):
                stop = ck_long
                risk_per_unit = price - stop
                if risk_per_unit <= 0:
                    return
                equity = self.equity
                risk_amount = equity * self.risk_pct
                size = int(round(risk_amount / risk_per_unit))
                if size <= 0:
                    return
                max_units_by_cash = int(equity / price)
                if size > max_units_by_cash:
                    size = max_units_by_cash
                if size <= 0:
                    return
                print(f"🚀🌙 Moon Dev LONG SIGNAL | price {price:.2f} | BBW% {bbw_pct:.1f} | vol {vol:.1f} vs {vol_ma:.1f} | stop {stop:.2f} | size {size}")
                self.buy(size=size)
                self.entry_bar = len(self.data)
                self.entry_price = price
                self.stop_price = stop
                self.highest_since_entry = self.data.High[-1]

            elif short_trigger and (macd_cross_dn or macd_falling):
                stop = ck_short
                risk_per_unit = stop - price
                if risk_per_unit <= 0:
                    return
                equity = self.equity
                risk_amount = equity * self.risk_pct
                size = int(round(risk_amount / risk_per_unit))
                if size <= 0:
                    return
                max_units_by_cash = int(equity / price)
                if size > max_units_by_cash:
                    size = max_units_by_cash
                if size <= 0:
                    return
                print(f"🚀🌙 Moon Dev SHORT SIGNAL | price {price:.2f} | BBW% {bbw_pct:.1f} | vol {vol:.1f} vs {vol_ma:.1f} | stop {stop:.2f} | size {size}")
                self.sell(size=size)
                self.entry_bar = len(self.data)
                self.entry_price = price
                self.stop_price = stop
                self.lowest_since_entry = self.data.Low[-1]


print("🌙 Moon Dev running SqueezePulse backtest... 🚀✨")
bt = Backtest(data, SqueezePulse, cash=1_000_000, commission=0.001, exclusive_orders=True)
stats = bt.run()
print(stats)
print(stats._strategy)