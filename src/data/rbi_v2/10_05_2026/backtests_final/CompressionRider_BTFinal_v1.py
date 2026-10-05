import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# ── Moon Dev Data Wrangling 🌙 ─────────────────────────────────────────────
DATA_PATH = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"

print("🌙 Moon Dev is loading the cosmic data...")
data = pd.read_csv(DATA_PATH)

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[c for c in data.columns if 'unnamed' in c.lower()])

# Ensure proper column mapping
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume',
})

# Parse datetime & set index
if 'datetime' in data.columns:
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')

data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
print(f"✨ Data loaded: {len(data)} bars of intergalactic price action!")


# ── CompressionRider Strategy 🚀 ───────────────────────────────────────────
class CompressionRider(Strategy):
    # Tunable parameters
    bb_period = 20
    bb_std = 2.0
    squeeze_lookback = 20
    atr_period = 14
    stop_atr_mult = 2.5
    trail_atr_mult = 2.0
    vol_ma_period = 20
    time_stop_bars = 15
    profit_threshold = 0.01  # 1% move qualifies as "working"
    risk_pct = 0.01          # 1% equity risk per trade

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        # Bollinger Bands
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS, close, timeperiod=self.bb_period,
            nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0
        )

        # Band width = (upper - lower) / middle
        self.band_width = self.I(
            lambda u, l, m: (u - l) / m,
            self.bb_upper, self.bb_lower, self.bb_middle
        )

        # 20-day min of band width (squeeze detector)
        self.bw_min = self.I(
            talib.MIN, self.band_width, timeperiod=self.squeeze_lookback
        )

        # ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        # Volume MA
        self.vol_ma = self.I(talib.SMA, volume, timeperiod=self.vol_ma_period)

        # Trade state
        self.entry_price_val = None
        self.highest_close = None
        self.trailing_stop = None
        self.bars_in_trade = 0

        print("🌙✨ CompressionRider indicators initialized — ready to ride the squeeze!")

    def next(self):
        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]

        # ── Manage open trade ────────────────────────────────────────────
        if self.position:
            self.bars_in_trade += 1
            self.highest_close = max(self.highest_close, price)

            # Ratcheting ATR trailing stop
            new_trail = self.highest_close - self.trail_atr_mult * self.atr[-1]
            if new_trail > self.trailing_stop:
                self.trailing_stop = new_trail
                print(f"🚀 Trail ratcheted up to {self.trailing_stop:.2f}")

            # Check stop hit (intrabar)
            if low <= self.trailing_stop:
                print(f"🛑 Stop hit at {self.trailing_stop:.2f} | price low={low:.2f} — exiting!")
                self.position.close()
                self._reset_state()
                return

            # Momentum loss: close below middle band
            if price < self.bb_middle[-1]:
                print(f"📉 Close {price:.2f} below middle band {self.bb_middle[-1]:.2f} — momentum lost, exiting!")
                self.position.close()
                self._reset_state()
                return

            # Time stop
            if self.bars_in_trade >= self.time_stop_bars:
                pnl_pct = (price - self.entry_price_val) / self.entry_price_val
                if pnl_pct < self.profit_threshold:
                    print(f"⏰ Time stop ({self.bars_in_trade} bars) with pnl {pnl_pct*100:.2f}% — exiting!")
                    self.position.close()
                    self._reset_state()
                    return
            return

        # ── Entry logic ──────────────────────────────────────────────────
        # Need enough history
        if len(self.data) < self.squeeze_lookback + 5:
            return

        bw = self.band_width[-1]
        bw_min = self.bw_min[-1]

        if np.isnan(bw) or np.isnan(bw_min) or np.isnan(self.atr[-1]):
            return

        # Squeeze condition: current BW is the 20-day minimum
        squeeze = bw <= bw_min * 1.0001  # tiny tolerance

        # Breakout confirmation: close above upper band
        breakout = price > self.bb_upper[-1]

        # Filters
        above_middle = price > self.bb_middle[-1]
        vol_confirm = self.data.Volume[-1] > self.vol_ma[-1]

        if squeeze and breakout and above_middle and vol_confirm:
            # Position sizing based on risk
            stop_price = price - self.stop_atr_mult * self.atr[-1]
            risk_per_unit = price - stop_price
            if risk_per_unit <= 0:
                return

            risk_amount = self.equity * self.risk_pct
            size = int(round(risk_amount / risk_per_unit))
            if size < 1:
                size = 1

            print(f"🌙✨ SQUEEZE BREAKOUT! Entry @ {price:.2f} | BW={bw:.4f} | "
                  f"ATR={self.atr[-1]:.2f} | Stop={stop_price:.2f} | Size={size}")

            self.buy(size=size)
            self.entry_price_val = price
            self.highest_close = price
            self.trailing_stop = stop_price
            self.bars_in_trade = 0

    def _reset_state(self):
        self.entry_price_val = None
        self.highest_close = None
        self.trailing_stop = None
        self.bars_in_trade = 0


# ── Run the Backtest 🌙 ────────────────────────────────────────────────────
print("🚀 Moon Dev launching CompressionRider backtest...")
bt = Backtest(
    data,
    CompressionRider,
    cash=1_000_000,
    commission=0.001,
)
stats = bt.run()
print(stats)
print(stats._strategy)