import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# 🌙 Moon Dev's VolatilityCascade Strategy ✨

def ema(series, period):
    return talib.EMA(np.asarray(series, dtype=np.float64), timeperiod=period)

def sma(series, period):
    return talib.SMA(np.asarray(series, dtype=np.float64), timeperiod=period)

def atr_func(high, low, close, period=14):
    return talib.ATR(
        np.asarray(high, dtype=np.float64),
        np.asarray(low, dtype=np.float64),
        np.asarray(close, dtype=np.float64),
        timeperiod=period,
    )

def bb_upper(close, period=20, dev=2):
    return talib.BBANDS(np.asarray(close, dtype=np.float64), timeperiod=period, nbdevup=dev, nbdevdn=dev, matype=0)[0]

def bb_middle(close, period=20, dev=2):
    return talib.BBANDS(np.asarray(close, dtype=np.float64), timeperiod=period, nbdevup=dev, nbdevdn=dev, matype=0)[1]

def bb_lower(close, period=20, dev=2):
    return talib.BBANDS(np.asarray(close, dtype=np.float64), timeperiod=period, nbdevup=dev, nbdevdn=dev, matype=0)[2]

def rolling_percentile(series, window, pct):
    """Rolling percentile rank helper."""
    series = np.asarray(series, dtype=np.float64)
    result = np.full(len(series), np.nan)
    for i in range(window, len(series)):
        window_data = series[i-window:i]
        result[i] = np.nanpercentile(window_data, pct)
    return result


class VolatilityCascade(Strategy):
    # Parameters
    ema_fast_period = 8
    ema_med_period = 21
    sma_slow_period = 50
    sma_trend_period = 200
    bb_period = 20
    bb_dev = 2
    atr_period = 14
    bbw_lookback = 100
    bbw_compress_pct = 25
    atr_avg_period = 20
    risk_pct = 0.01
    tp1_mult = 1.5
    tp2_mult = 3.0

    def init(self):
        print("🌙 Moon Dev VolatilityCascade initializing... ✨")
        close = self.data.Close
        high = self.data.High
        low = self.data.Low

        # EMAs on execution timeframe
        self.ema_fast = self.I(ema, close, self.ema_fast_period)
        self.ema_med = self.I(ema, close, self.ema_med_period)
        self.sma_slow = self.I(sma, close, self.sma_slow_period)
        self.sma_trend = self.I(sma, close, self.sma_trend_period)

        # Bollinger Bands
        self.bb_u = self.I(bb_upper, close, self.bb_period, self.bb_dev)
        self.bb_m = self.I(bb_middle, close, self.bb_period, self.bb_dev)
        self.bb_l = self.I(bb_lower, close, self.bb_period, self.bb_dev)

        # ATR
        self.atr = self.I(atr_func, high, low, close, self.atr_period)
        self.atr_avg = self.I(sma, self.atr, self.atr_avg_period)

        # BBW = (upper - lower) / middle
        bbw = (self.bb_u - self.bb_l) / np.where(self.bb_m == 0, np.nan, self.bb_m)
        self.bbw = self.I(lambda x: x, bbw)
        self.bbw_avg = self.I(sma, self.bbw, 50)

        # Compression threshold (rolling percentile)
        self.bbw_compress_thresh = self.I(
            rolling_percentile, self.bbw, self.bbw_lookback, self.bbw_compress_pct
        )

        # Volume average
        self.vol_avg = self.I(sma, self.data.Volume, 20)

        # State tracking
        self.tp1_hit = False
        self.entry_price = 0.0
        self.stop_price = 0.0
        self.tp1_price = 0.0
        self.tp2_price = 0.0
        self.trade_dir = 0  # 1 long, -1 short

        print("🌙 Moon Dev indicators loaded! 🚀")

    def candle_body_ratio(self, o, h, l, c):
        rng = h - l
        if rng <= 0:
            return 0
        return abs(c - o) / rng

    def is_bullish_engulf(self, i):
        if i < 1:
            return False
        o1, c1 = self.data.Open[i-1], self.data.Close[i-1]
        o2, c2 = self.data.Open[i], self.data.Close[i]
        return c1 < o1 and c2 > o2 and c2 >= o1 and o2 <= c1

    def is_bearish_engulf(self, i):
        if i < 1:
            return False
        o1, c1 = self.data.Open[i-1], self.data.Close[i-1]
        o2, c2 = self.data.Open[i], self.data.Close[i]
        return c1 > o1 and c2 < o2 and c2 <= o1 and o2 >= c1

    def is_pin_bull(self, i):
        o, h, l, c = self.data.Open[i], self.data.High[i], self.data.Low[i], self.data.Close[i]
        rng = h - l
        if rng <= 0:
            return False
        body = abs(c - o)
        lower_wick = min(o, c) - l
        return lower_wick > 2 * body and (c > o)

    def is_pin_bear(self, i):
        o, h, l, c = self.data.Open[i], self.data.High[i], self.data.Low[i], self.data.Close[i]
        rng = h - l
        if rng <= 0:
            return False
        body = abs(c - o)
        upper_wick = h - max(o, c)
        return upper_wick > 2 * body and (c < o)

    def next(self):
        i = len(self.data) - 1
        if i < max(self.sma_trend_period, self.bbw_lookback) + 2:
            return

        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        open_ = self.data.Open[-1]
        vol = self.data.Volume[-1]

        ef = self.ema_fast[-1]
        em = self.ema_med[-1]
        ss = self.sma_slow[-1]
        st = self.sma_trend[-1]

        bbw_val = self.bbw[-1]
        bbw_thresh = self.bbw_compress_thresh[-1]
        bbw_avg = self.bbw_avg[-1]

        atr_val = self.atr[-1]
        atr_avg_val = self.atr_avg[-1]

        # Skip if indicators not ready
        if any(np.isnan(x) for x in [ef, em, ss, st, bbw_val, bbw_thresh, bbw_avg, atr_val, atr_avg_val]):
            return

        bullish_stack = ef > em > ss and price > st
        bearish_stack = ef < em < ss and price < st

        vol_ok = vol > self.vol_avg[-1] * 1.0
        atr_expanding = atr_val > atr_avg_val

        # === Manage open position ===
        if self.position:
            if self.trade_dir == 1:
                # Trailing stop after TP1
                if self.tp1_hit:
                    trail = max(self.ema_fast[-1], price - 2 * atr_val)
                    if trail > self.stop_price:
                        self.stop_price = trail
                        print(f"🌙 Trailing stop raised to {trail:.2f} 🚀")

                # TP2
                if price >= self.tp2_price:
                    print(f"🌙✨ TP2 hit! Closing long at {price:.2f} 💰")
                    self.position.close()
                    self._reset_state()
                    return

                # TP1 partial
                if not self.tp1_hit and price >= self.tp1_price:
                    print(f"🌙 TP1 hit! Taking 50% off at {price:.2f} 🎯")
                    self.position.close(0.5)
                    self.tp1_hit = True

                # Stop loss
                if low <= self.stop_price:
                    print(f"🌙 Stop loss triggered at {self.stop_price:.2f} 💀")
                    self.position.close()
                    self._reset_state()
                    return

                # Trend invalidation
                if price < ss:
                    print(f"🌙 Trend invalidation - price below slow SMA 💀")
                    self.position.close()
                    self._reset_state()
                    return

            elif self.trade_dir == -1:
                if self.tp1_hit:
                    trail = min(self.ema_fast[-1], price + 2 * atr_val)
                    if trail < self.stop_price:
                        self.stop_price = trail
                        print(f"🌙 Trailing stop lowered to {trail:.2f} 🚀")

                if price <= self.tp2_price:
                    print(f"🌙✨ TP2 hit! Closing short at {price:.2f} 💰")
                    self.position.close()
                    self._reset_state()
                    return

                if not self.tp1_hit and price <= self.tp1_price:
                    print(f"🌙 TP1 hit! Taking 50% off at {price:.2f} 🎯")
                    self.position.close(0.5)
                    self.tp1_hit = True

                if high >= self.stop_price:
                    print(f"🌙 Stop loss triggered at {self.stop_price:.2f} 💀")
                    self.position.close()
                    self._reset_state()
                    return

                if price > ss:
                    print(f"🌙 Trend invalidation - price above slow SMA 💀")
                    self.position.close()
                    self._reset_state()
                    return

            return  # already in position

        # === Entry Logic ===
        in_compression = bbw_val <= bbw_thresh
        expanding = bbw_val > bbw_avg

        body_ratio = self.candle_body_ratio(open_, high, low, price)

        # Long breakout
        long_breakout = (
            bullish_stack
            and price > self.bb_u[-1]
            and atr_expanding
            and vol_ok
            and (body_ratio > 0.7 or self.is_bullish_engulf(i))
        )

        # Short breakout
        short_breakout = (
            bearish_stack
            and price < self.bb_l[-1]
            and atr_expanding
            and vol_ok
            and (body_ratio > 0.7 or self.is_bearish_engulf(i))
        )

        # Re-entry long: pullback to fast/medium EMA with bullish pin/engulf
        near_ema = abs(price - ef) / price < 0.005 or abs(price - em) / price < 0.005
        reentry_long = (
            bullish_stack
            and near_ema
            and (self.is_pin_bull(i) or self.is_bullish_engulf(i))
            and price > ss
        )

        reentry_short = (
            bearish_stack
            and near_ema
            and (self.is_pin_bear(i) or self.is_bearish_engulf(i))
            and price < ss
        )

        if long_breakout or reentry_long:
            stop = min(low, price - 1.5 * atr_val)
            risk = price - stop
            if risk <= 0:
                return
            size = int(round(1000000 / price))
            if size < 1:
                size = 1
            self.buy(size=size)
            self.entry_price = price
            self.stop_price = stop
            self.tp1_price = price + self.tp1_mult * atr_val
            self.tp2_price = price + self.tp2_mult * atr_val
            self.trade_dir = 1
            self.tp1_hit = False
            print(f"🌙🚀 LONG entry @ {price:.2f} | Stop {stop:.2f} | TP1 {self.tp1_price:.2f} | TP2 {self.tp2_price:.2f} | size {size}")

        elif short_breakout or reentry_short:
            stop = max(high, price + 1.5 * atr_val)
            risk = stop - price
            if risk <= 0:
                return
            size = int(round(1000000 / price))
            if size < 1:
                size = 1
            self.sell(size=size)
            self.entry_price = price
            self.stop_price = stop
            self.tp1_price = price - self.tp1_mult * atr_val
            self.tp2_price = price - self.tp2_mult * atr_val
            self.trade_dir = -1
            self.tp1_hit = False
            print(f"🌙🔻 SHORT entry @ {price:.2f} | Stop {stop:.2f} | TP1 {self.tp1_price:.2f} | TP2 {self.tp2_price:.2f} | size {size}")

    def _reset_state(self):
        self.tp1_hit = False
        self.trade_dir = 0
        self.entry_price = 0.0
        self.stop_price = 0.0
        self.tp1_price = 0.0
        self.tp2_price = 0.0


# === Load Data ===
print("🌙 Moon Dev loading data... ✨")
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')

# Clean columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Map to proper case
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

# Ensure float64 dtype for talib compatibility
for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    data[col] = data[col].astype(np.float64)

print(f"🌙 Data loaded: {len(data)} bars 🚀")

# === Run Backtest ===
bt = Backtest(data, VolatilityCascade, cash=1_000_000, commission=0.001, exclusive_orders=True)
stats = bt.run()
print(stats)
print(stats._strategy)