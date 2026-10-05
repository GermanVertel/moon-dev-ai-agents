import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# 🌙 Moon Dev CompressionFlip Strategy 🚀
# Volatility compression + Supertrend flip reversal system

def supertrend(high, low, close, period=10, multiplier=3):
    """Calculate Supertrend indicator"""
    hl2 = (high + low) / 2
    atr = talib.ATR(high, low, close, timeperiod=period)
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr

    st = np.zeros(len(close))
    direction = np.ones(len(close))  # 1 = bullish, -1 = bearish

    for i in range(1, len(close)):
        if close[i] > upper_band[i - 1]:
            direction[i] = 1
        elif close[i] < lower_band[i - 1]:
            direction[i] = -1
        else:
            direction[i] = direction[i - 1]
            if direction[i] == 1 and lower_band[i] < lower_band[i - 1]:
                lower_band[i] = lower_band[i - 1]
            if direction[i] == -1 and upper_band[i] > upper_band[i - 1]:
                upper_band[i] = upper_band[i - 1]

        if direction[i] == 1:
            st[i] = lower_band[i]
        else:
            st[i] = upper_band[i]

    return st, direction


class CompressionFlip(Strategy):
    bb_period = 20
    bb_mult = 2.0
    bbw_lookback = 100
    bbw_percentile = 10
    st_period = 10
    st_mult = 3.0
    atr_period = 14
    atr_mult = 2.0
    squeeze_window = 3
    squeeze_timeout = 20
    risk_pct = 0.01

    def init(self):
        # 🌙 Bollinger Bands via talib
        self.bb_up = self.I(lambda c: talib.SMA(c, self.bb_period) + self.bb_mult * talib.STDDEV(c, self.bb_period),
                            self.data.Close)
        self.bb_mid = self.I(talib.SMA, self.data.Close, timeperiod=self.bb_period)
        self.bb_low = self.I(lambda c: talib.SMA(c, self.bb_period) - self.bb_mult * talib.STDDEV(c, self.bb_period),
                             self.data.Close)

        # 🌙 BBW = (Upper - Lower) / Middle
        self.bbw = self.I(lambda u, l, m: (u - l) / m, self.bb_up, self.bb_low, self.bb_mid)

        # 🌙 BBW 10th percentile over lookback
        def bbw_pct(arr):
            out = np.full(len(arr), np.nan)
            for i in range(self.bbw_lookback, len(arr)):
                window = arr[i - self.bbw_lookback:i + 1]
                window = window[~np.isnan(window)]
                if len(window) > 0:
                    out[i] = np.percentile(window, self.bbw_percentile)
            return out

        self.bbw_p10 = self.I(bbw_pct, self.bbw)

        # 🌙 ATR
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close,
                          timeperiod=self.atr_period)

        # 🌙 Supertrend
        st, direction = supertrend(
            self.data.High.values, self.data.Low.values, self.data.Close.values,
            period=self.st_period, multiplier=self.st_mult
        )
        self.st_line = self.I(lambda: st)
        self.st_dir = self.I(lambda: direction)

        # 🌙 State tracking
        self.squeeze_bar = -1
        self.entry_price = 0
        self.highest_high = 0
        self.lowest_low = 0
        self.trail_stop = 0

        print("🌙✨ CompressionFlip indicators initialized! 🚀")

    def next(self):
        i = len(self.data) - 1
        if i < max(self.bbw_lookback, self.st_period, self.atr_period) + 5:
            return

        price = self.data.Close[-1]
        bbw_now = self.bbw[-1]
        bbw_p10_now = self.bbw_p10[-1]

        if np.isnan(bbw_now) or np.isnan(bbw_p10_now):
            return

        # 🌙 Detect squeeze
        squeeze_active = bbw_now < bbw_p10_now
        if squeeze_active and self.squeeze_bar < 0:
            self.squeeze_bar = i
            print(f"🌙 SQUEEZE detected at bar {i} | BBW={bbw_now:.6f} < P10={bbw_p10_now:.6f}")

        # 🌙 Squeeze timeout
        if self.squeeze_bar >= 0 and (i - self.squeeze_bar) > self.squeeze_timeout:
            print(f"⏰ Squeeze timeout at bar {i} — invalidating setup")
            self.squeeze_bar = -1

        # 🌙 Supertrend flip detection (no backtesting.lib crossover)
        st_dir_now = self.st_dir[-1]
        st_dir_prev = self.st_dir[-2]
        bull_flip = st_dir_prev == -1 and st_dir_now == 1
        bear_flip = st_dir_prev == 1 and st_dir_now == -1

        # 🌙 Squeeze active within window
        squeeze_valid = self.squeeze_bar >= 0 and (i - self.squeeze_bar) <= self.squeeze_window

        # ============ ENTRY LOGIC ============
        if not self.position:
            if squeeze_valid and bull_flip:
                stop = price - self.atr_mult * self.atr[-1]
                risk = price - stop
                if risk > 0:
                    size = int(round((self.equity * self.risk_pct) / risk))
                    if size > 0:
                        self.buy(size=size)
                        self.entry_price = price
                        self.highest_high = price
                        self.trail_stop = stop
                        self.squeeze_bar = -1
                        print(f"🚀🌙 LONG ENTRY @ {price:.2f} | size={size} | stop={stop:.2f}")

            elif squeeze_valid and bear_flip:
                stop = price + self.atr_mult * self.atr[-1]
                risk = stop - price
                if risk > 0:
                    size = int(round((self.equity * self.risk_pct) / risk))
                    if size > 0:
                        self.sell(size=size)
                        self.entry_price = price
                        self.lowest_low = price
                        self.trail_stop = stop
                        self.squeeze_bar = -1
                        print(f"🔻🌙 SHORT ENTRY @ {price:.2f} | size={size} | stop={stop:.2f}")

        # ============ EXIT LOGIC ============
        else:
            if self.position.is_long:
                self.highest_high = max(self.highest_high, self.data.High[-1])
                new_stop = self.highest_high - self.atr_mult * self.atr[-1]
                if new_stop > self.trail_stop:
                    self.trail_stop = new_stop

                # Trend exit
                if bear_flip:
                    self.position.close()
                    print(f"🌙 TREND EXIT LONG @ {price:.2f}")
                # Trailing stop exit
                elif self.data.Low[-1] <= self.trail_stop:
                    self.position.close()
                    print(f"🛑 ATR TRAIL EXIT LONG @ {price:.2f} | stop={self.trail_stop:.2f}")

            elif self.position.is_short:
                self.lowest_low = min(self.lowest_low, self.data.Low[-1])
                new_stop = self.lowest_low + self.atr_mult * self.atr[-1]
                if new_stop < self.trail_stop:
                    self.trail_stop = new_stop

                # Trend exit
                if bull_flip:
                    self.position.close()
                    print(f"🌙 TREND EXIT SHORT @ {price:.2f}")
                # Trailing stop exit
                elif self.data.High[-1] >= self.trail_stop:
                    self.position.close()
                    print(f"🛑 ATR TRAIL EXIT SHORT @ {price:.2f} | stop={self.trail_stop:.2f}")


# 🌙 Load and prepare data
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper column mapping
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

print(f"🌙✨ Loaded {len(data)} bars for CompressionFlip backtest 🚀")

bt = Backtest(data, CompressionFlip, cash=1_000_000, commission=0.002, exclusive_orders=True)
stats = bt.run()
print(stats)
print(stats._strategy)