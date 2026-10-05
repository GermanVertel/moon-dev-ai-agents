import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# ── Data loading & cleaning ──────────────────────────────────────────────
data = pd.read_csv(
    "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
)
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open', 'high': 'High', 'low': 'Low',
    'close': 'Close', 'volume': 'Volume'
})
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

print("🌙✨ CascadeReclaim Backtest Booting Up... Data Loaded:", len(data), "bars 🚀")


class CascadeReclaim(Strategy):
    # ── Tunables ─────────────────────────────────────────────────────────
    vol_lookback      = 20      # avg volume window
    vol_spike_mult    = 2.0     # volume spike threshold
    flush_drop_pct    = 0.008   # min drop of flush candle body (0.8%)
    wick_ratio_min    = 1.0     # lower wick >= body
    ema_period        = 20      # reclaim reference EMA
    rsi_period        = 14
    rsi_oversold      = 35
    atr_period        = 14
    atr_stop_mult     = 1.5
    reward_ratio      = 2.0
    reclaim_window    = 3       # bars allowed to reclaim
    time_stop_bars    = 20
    risk_pct          = 0.01    # 1% risk per trade

    def init(self):
        self.ema = self.I(talib.EMA, self.data.Close, timeperiod=self.ema_period)
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=self.rsi_period)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low,
                          self.data.Close, timeperiod=self.atr_period)
        self.vol_ma = self.I(talib.SMA, self.data.Volume, timeperiod=self.vol_lookback)
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=20)
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=20)
        self.bar_count = 0
        self.entry_bar = None
        self.stop_price = None
        self.tp1 = None
        self.tp2 = None
        self.entry_price = None
        self._flush_bar = None
        self._flush_open = None
        self._flush_low = None
        self._flush_origin = None

    def next(self):
        self.bar_count += 1
        price = self.data.Close[-1]

        # ── Manage open position ────────────────────────────────────────
        if self.position:
            bars_held = self.bar_count - self.entry_bar
            # Time stop
            if bars_held >= self.time_stop_bars:
                print(f"⏰🌙 Time stop hit after {bars_held} bars — exiting.")
                self.position.close()
                return
            # Trailing / TP2 handled via orders; also trail below EMA on strong move
            if self.entry_price is not None and price > self.entry_price + self.atr[-1]:
                new_stop = max(self.stop_price, self.ema[-1])
                if new_stop > self.stop_price:
                    self.stop_price = new_stop
                    print(f"🔧🌙 Trailing stop raised to {new_stop:.2f}")
            return

        # ── Detect liquidation flush ────────────────────────────────────
        if len(self.data) < self.vol_lookback + 5:
            return

        o = self.data.Open[-1]
        h = self.data.High[-1]
        l = self.data.Low[-1]
        c = self.data.Close[-1]
        v = self.data.Volume[-1]

        body = abs(c - o)
        lower_wick = min(o, c) - l
        is_red = c < o

        vol_spike = v > self.vol_spike_mult * self.vol_ma[-1]
        big_drop = is_red and (o - c) / o >= self.flush_drop_pct
        long_wick = lower_wick >= self.wick_ratio_min * max(body, 1e-9)
        rsi_low = self.rsi[-1] <= self.rsi_oversold or self.rsi[-2] <= self.rsi_oversold

        flush = vol_spike and big_drop and long_wick

        if flush and rsi_low:
            print(f"💥🌙 LIQUIDATION FLUSH detected @ {c:.2f} | vol={v:.2f} "
                  f"(x{v / self.vol_ma[-1]:.1f}) RSI={self.rsi[-1]:.1f}")

        # ── Reclaim confirmation (within reclaim_window bars) ───────────
        if flush:
            # store flush info on the instance for next bars
            self._flush_open = o
            self._flush_low = l
            self._flush_bar = self.bar_count
            self._flush_origin = self.swing_high[-1]
            return

        # Check reclaim after a recent flush
        if self._flush_bar is not None and (self.bar_count - self._flush_bar) <= self.reclaim_window:
            reclaim_level = self._flush_open
            reclaimed = c > reclaim_level
            rsi_turn = self.rsi[-1] > self.rsi[-2] and self.rsi[-1] > 30
            vol_ok = v > self.vol_ma[-1]

            if reclaimed and rsi_turn and vol_ok:
                print(f"✅🌙 RECLAIM confirmed @ {c:.2f} (level {reclaim_level:.2f}) "
                      f"RSI={self.rsi[-1]:.1f} — ENTERING LONG 🚀")

                # ── Risk management ─────────────────────────────────────
                stop = self._flush_low - self.atr[-1] * 0.25
                risk = c - stop
                if risk <= 0:
                    return
                tp1 = self._flush_origin if self._flush_origin > c else c + risk * self.reward_ratio
                tp2 = c + risk * self.reward_ratio * 1.5

                # Position size: risk 1% of equity, integer units
                risk_amount = self.equity * self.risk_pct
                size = int(round(risk_amount / risk))
                if size < 1:
                    size = 1

                self.buy(size=size, sl=stop, tp=tp1)
                self.entry_bar = self.bar_count
                self.entry_price = c
                self.stop_price = stop
                self.tp1 = tp1
                self.tp2 = tp2
                print(f"   🎯 SL={stop:.2f} | TP1={tp1:.2f} | TP2={tp2:.2f} | size={size}")
                self._flush_bar = None
        else:
            if self._flush_bar is not None:
                self._flush_bar = None


bt = Backtest(data, CascadeReclaim, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)