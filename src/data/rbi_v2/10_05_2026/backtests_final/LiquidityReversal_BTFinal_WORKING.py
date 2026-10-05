import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# ============================================================
# 🌙 Moon Dev's LiquidityReversal Strategy 🌙
# ============================================================

print("🌙✨ Moon Dev Backtest AI warming up... 🚀")

data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper column mapping
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
data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

# Ensure OHLCV are float64 (talib requires double arrays)
for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    data[col] = data[col].astype(np.float64)

print(f"🌙 Data loaded: {len(data)} bars ✨")
print(f"🚀 Date range: {data.index[0]} -> {data.index[-1]}")


class LiquidityReversal(Strategy):
    # Parameters
    swing_lookback = 20          # for swing high/low detection
    vwap_window = 96             # ~24h on 15m
    obv_ema_period = 9
    atr_period = 14
    risk_pct = 0.02              # 2% risk per trade
    rr_target = 2.0              # take profit R multiple
    time_stop_bars = 20
    zone_atr_mult = 0.5          # zone width in ATR

    def init(self):
        print("🌙 Initializing Moon Dev indicators... ✨")

        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        # ATR for stops and zone sizing
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        # Rolling VWAP (typical price * volume cumulative / volume cumulative)
        typical = (high + low + close) / 3.0
        self.vwap = self.I(
            lambda tp, v, n: (pd.Series(tp * v).rolling(n).sum() /
                              pd.Series(v).rolling(n).sum()).values,
            typical, volume, self.vwap_window
        )

        # OBV + EMA — cast close/volume to float64 arrays for talib
        close_arr = np.asarray(close, dtype=np.float64)
        volume_arr = np.asarray(volume, dtype=np.float64)
        self.obv = self.I(talib.OBV, close_arr, volume_arr)
        self.obv_ema = self.I(talib.EMA, self.obv, timeperiod=self.obv_ema_period)

        # Swing highs/lows (liquidation zone proxies)
        self.swing_high = self.I(talib.MAX, high, timeperiod=self.swing_lookback)
        self.swing_low = self.I(talib.MIN, low, timeperiod=self.swing_lookback)

        # Detect rejection candles
        self.bull_rej = self.I(
            lambda o, h, l, c: ((c > o) &
                                ((np.minimum(o, c) - l) > (c - o) * 1.5) &
                                ((h - np.maximum(o, c)) < (c - o) * 0.5)).astype(float),
            self.data.Open, high, low, close
        )
        self.bear_rej = self.I(
            lambda o, h, l, c: ((c < o) &
                                ((h - np.maximum(o, c)) > (o - c) * 1.5) &
                                ((np.minimum(o, c) - l) < (o - c) * 0.5)).astype(float),
            self.data.Open, high, low, close
        )

        # Track entry info
        self.entry_bar = None
        self.entry_price = None
        self.stop_price = None
        self.tp_price = None

        print("🚀 Indicators ready! Let's hunt some liquidity! 🌙")

    def next(self):
        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        atr = self.atr[-1]

        if np.isnan(atr) or atr <= 0:
            return

        # ---------------- Manage open position ----------------
        if self.position:
            bars_held = len(self.data) - self.entry_bar

            # OBV EMA flip exit
            if self.position.is_long:
                if self.obv_ema[-1] < self.obv_ema[-2]:
                    print(f"🌙✨ OBV flip bearish — exiting LONG @ {price:.2f}")
                    self.position.close()
                    self._reset()
                    return
            else:
                if self.obv_ema[-1] > self.obv_ema[-2]:
                    print(f"🌙✨ OBV flip bullish — exiting SHORT @ {price:.2f}")
                    self.position.close()
                    self._reset()
                    return

            # Time stop
            if bars_held >= self.time_stop_bars:
                print(f"⏰ Time stop hit ({bars_held} bars) — closing position @ {price:.2f}")
                self.position.close()
                self._reset()
                return
            return

        # ---------------- Look for entries ----------------
        if len(self.data) < max(self.swing_lookback, self.vwap_window, self.atr_period) + 5:
            return

        swing_lo = self.swing_low[-1]
        swing_hi = self.swing_high[-1]
        vwap_now = self.vwap[-1]
        vwap_prev = self.vwap[-5] if len(self.vwap) > 5 else vwap_now

        # Liquidation zone widths (ATR-based)
        zone_width = atr * self.zone_atr_mult

        # ---- LONG setup: price hunts below swing low + bullish VWAP divergence ----
        long_zone_lo = swing_lo - zone_width
        long_zone_hi = swing_lo + zone_width
        in_long_zone = long_zone_lo <= low <= long_zone_hi

        # Bullish VWAP divergence: price lower low, VWAP higher low
        price_ll = low < self.data.Low[-2]
        vwap_hl = vwap_now > vwap_prev

        if in_long_zone and price_ll and vwap_hl and self.bull_rej[-1] > 0:
            stop = long_zone_lo - atr * 0.5
            risk = price - stop
            if risk > 0:
                size = int(round((self.equity * self.risk_pct) / risk))
                if size > 0:
                    print(f"🌙🚀 LONG ENTRY @ {price:.2f} | zone=[{long_zone_lo:.2f},{long_zone_hi:.2f}] "
                          f"| stop={stop:.2f} | size={size}")
                    self.buy(size=size)
                    self.entry_bar = len(self.data)
                    self.entry_price = price
                    self.stop_price = stop
                    self.tp_price = price + risk * self.rr_target
                    return

        # ---- SHORT setup: price hunts above swing high + bearish VWAP divergence ----
        short_zone_lo = swing_hi - zone_width
        short_zone_hi = swing_hi + zone_width
        in_short_zone = short_zone_lo <= high <= short_zone_hi

        # Bearish VWAP divergence: price higher high, VWAP lower high
        price_hh = high > self.data.High[-2]
        vwap_lh = vwap_now < vwap_prev

        if in_short_zone and price_hh and vwap_lh and self.bear_rej[-1] > 0:
            stop = short_zone_hi + atr * 0.5
            risk = stop - price
            if risk > 0:
                size = int(round((self.equity * self.risk_pct) / risk))
                if size > 0:
                    print(f"🌙🚀 SHORT ENTRY @ {price:.2f} | zone=[{short_zone_lo:.2f},{short_zone_hi:.2f}] "
                          f"| stop={stop:.2f} | size={size}")
                    self.sell(size=size)
                    self.entry_bar = len(self.data)
                    self.entry_price = price
                    self.stop_price = stop
                    self.tp_price = price - risk * self.rr_target
                    return

    def _reset(self):
        self.entry_bar = None
        self.entry_price = None
        self.stop_price = None
        self.tp_price = None


print("🌙✨ Launching Moon Dev LiquidityReversal Backtest... 🚀")
bt = Backtest(data, LiquidityReversal, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙✨ Backtest complete! Moon Dev out. 🚀")