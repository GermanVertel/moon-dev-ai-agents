import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# ── Moon Dev Data Loading 🌙 ──────────────────────────────────────
data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper mapping
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume',
})

data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
print(f"🌙 Moon Dev loaded data: {len(data)} candles ✨")


class VolatilityTrapDivergence(Strategy):
    # Parameters
    bb_period = 20
    bb_std = 2.0
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    atr_period = 14
    atr_avg_period = 20
    atr_min_mult = 1.0       # ATR must be > this * ATR average
    atr_spike_mult = 1.5     # ATR spike exit
    atr_collapse_mult = 0.5  # ATR collapse exit
    stop_atr_mult = 1.5
    risk_pct = 0.01
    time_stop_bars = 12

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low

        # Bollinger Bands
        self.bb_upper, self.bb_mid, self.bb_lower = self.I(
            talib.BBANDS, close, timeperiod=self.bb_period,
            nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0
        )

        # MACD
        self.macd, self.macd_signal, self.macd_hist = self.I(
            talib.MACD, close,
            fastperiod=self.macd_fast,
            slowperiod=self.macd_slow,
            signalperiod=self.macd_signal
        )

        # ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)
        self.atr_avg = self.I(talib.SMA, self.atr, timeperiod=self.atr_avg_period)

        # Trackers
        self.entry_bar = None
        self.entry_price = None
        self.stop_price = None

    def next(self):
        if len(self.data) < max(self.bb_period, self.macd_slow, self.atr_avg_period) + 2:
            return

        price = self.data.Close[-1]
        high = self.data.High[-1]
        upper = self.bb_upper[-1]
        mid = self.bb_mid[-1]
        lower = self.bb_lower[-1]

        macd = self.macd[-1]
        sig = self.macd_signal[-1]
        hist = self.macd_hist[-1]
        hist_prev = self.macd_hist[-2]

        atr = self.atr[-1]
        atr_avg = self.atr_avg[-1]

        # ── Manage open position ────────────────────────────────
        if self.position:
            bars_held = len(self.data) - self.entry_bar

            # Exit B: ATR spike against position
            if atr > self.atr_spike_mult * atr_avg:
                print(f"🚨 Moon Dev ATR SPIKE exit @ {price:.2f} 🌙")
                self.position.close()
                return

            # Exit B: ATR collapse
            if atr < self.atr_collapse_mult * atr_avg:
                print(f"😴 Moon Dev ATR COLLAPSE exit @ {price:.2f} 💤")
                self.position.close()
                return

            # Exit A: MACD bullish crossover (momentum turn) — divergence proxy
            if macd > sig and macd_hist > hist_prev:
                print(f"✨ Moon Dev MACD TURN exit @ {price:.2f} 🌙")
                self.position.close()
                return

            # Exit C: Middle band target (mean reversion)
            if price <= mid:
                print(f"🎯 Moon Dev MID-BAND target hit @ {price:.2f} 💰")
                self.position.close()
                return

            # Exit C: Time stop
            if bars_held >= self.time_stop_bars:
                print(f"⏰ Moon Dev TIME STOP exit @ {price:.2f} 🌙")
                self.position.close()
                return

            # Trailing stop check
            if high >= self.stop_price:
                print(f"🛑 Moon Dev STOP hit @ {self.stop_price:.2f} 💥")
                self.position.close()
                return

            return

        # ── Entry logic (short fade setup) ──────────────────────
        touch_upper = high >= upper
        macd_below_signal = macd < sig
        hist_flat_or_down = hist <= hist_prev
        atr_ok = atr > self.atr_min_mult * atr_avg

        if touch_upper and macd_below_signal and hist_flat_or_down and atr_ok:
            # Risk-based sizing
            stop_price = high + self.stop_atr_mult * atr
            risk_per_unit = stop_price - price
            if risk_per_unit <= 0:
                return

            risk_amount = self.equity * self.risk_pct
            size = risk_amount / risk_per_unit
            size = int(round(size))

            if size < 1:
                print("⚠️  Moon Dev: size too small, skipping 🌙")
                return

            self.sell(size=size)
            self.entry_bar = len(self.data)
            self.entry_price = price
            self.stop_price = stop_price

            print(f"🌙✨ TRAP SHORT @ {price:.2f} | Upper: {upper:.2f} | "
                  f"MACD {macd:.2f} < Sig {sig:.2f} | ATR {atr:.2f} | "
                  f"Stop {stop_price:.2f} | Size {size} 🚀")


bt = Backtest(data, VolatilityTrapDivergence, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)