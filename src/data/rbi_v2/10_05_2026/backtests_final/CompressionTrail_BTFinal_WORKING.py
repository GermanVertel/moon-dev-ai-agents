import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# ── Moon Dev Data Loading ─────────────────────────────────────────────
print("🌙 Moon Dev: Loading data from the cosmic archives...")
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')

# Clean column names
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
print(f"🚀 Moon Dev: Data loaded, {len(data)} bars ready for launch!")


class CompressionTrail(Strategy):
    # Strategy parameters
    bb_period = 20
    bb_std = 2.0
    bbw_lookback = 30
    squeeze_tolerance = 0.02  # within 2% of 30-period min BBW
    trail_pct = 0.75          # 75% of breakout candle range
    risk_pct = 0.02           # 2% risk per trade
    max_hold_bars = 200       # maximum holding period

    def init(self):
        print("🌙 Moon Dev: Initializing CompressionTrail indicators...")
        close = pd.Series(self.data.Close)

        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS, close, timeperiod=self.bb_period,
            nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0
        )

        # Bollinger Band Width = (upper - lower) / middle
        bbw = (self.bb_upper - self.bb_lower) / self.bb_middle
        self.bbw = self.I(lambda: bbw, name='BBW')

        # 30-period rolling minimum of BBW
        self.bbw_min = self.I(
            talib.MIN, bbw, timeperiod=self.bbw_lookback, name='BBW_MIN'
        )

        # Volume SMA for optional confirmation
        self.vol_sma = self.I(
            talib.SMA, pd.Series(self.data.Volume), timeperiod=20, name='VOL_SMA'
        )

        # State tracking
        self.entry_price = None
        self.highest_high = None
        self.breakout_range = None
        self.bars_held = 0
        self.trail_stop = None

        print("✨ Moon Dev: Indicators online! Let the squeeze hunt begin!")

    def next(self):
        price = self.data.Close[-1]

        # ── Manage open position ─────────────────────────────────
        if self.position:
            self.bars_held += 1
            self.highest_high = max(self.highest_high, self.data.High[-1])

            # Update trailing stop
            self.trail_stop = self.highest_high - (self.trail_pct * self.breakout_range)

            # Exit if price closes below trailing stop
            if price < self.trail_stop:
                print(f"🌙 Moon Dev EXIT: Close {price:.2f} < Trail {self.trail_stop:.2f} | PnL: {self.position.pl:.2f}")
                self.position.close()
                self._reset_state()
                return

            # Exit if max holding period reached
            if self.bars_held >= self.max_hold_bars:
                print(f"⏰ Moon Dev EXIT: Max hold {self.max_hold_bars} bars reached | PnL: {self.position.pl:.2f}")
                self.position.close()
                self._reset_state()
                return
            return

        # ── Entry logic ──────────────────────────────────────────
        # Need enough history
        if len(self.data) < self.bbw_lookback + self.bb_period + 2:
            return

        cur_bbw = self.bbw[-1]
        cur_bbw_min = self.bbw_min[-1]
        upper = self.bb_upper[-1]

        if np.isnan(cur_bbw) or np.isnan(cur_bbw_min) or np.isnan(upper):
            return

        # Squeeze condition: BBW at/near 30-period minimum
        squeeze = cur_bbw <= cur_bbw_min * (1 + self.squeeze_tolerance)

        # Breakout: close above upper band
        breakout = price > upper

        # Optional volume confirmation
        vol_ok = self.data.Volume[-1] > self.vol_sma[-1]

        if squeeze and breakout and vol_ok:
            # Breakout candle range
            br = self.data.High[-1] - self.data.Low[-1]
            if br <= 0:
                return

            # Position sizing: risk_pct of equity / trail distance
            equity = self.equity
            risk_amount = equity * self.risk_pct
            trail_distance = self.trail_pct * br

            if trail_distance <= 0:
                return

            size = int(round(risk_amount / trail_distance))
            if size < 1:
                size = 1

            print(f"🚀 Moon Dev ENTRY: Squeeze={squeeze} Breakout={breakout} | Price={price:.2f} Upper={upper:.2f} "
                  f"BBW={cur_bbw:.6f} BBW_MIN={cur_bbw_min:.6f} | Size={size} Range={br:.2f}")

            self.buy(size=size)
            self.entry_price = price
            self.highest_high = self.data.High[-1]
            self.breakout_range = br
            self.bars_held = 0
            self.trail_stop = self.highest_high - (self.trail_pct * br)

    def _reset_state(self):
        self.entry_price = None
        self.highest_high = None
        self.breakout_range = None
        self.bars_held = 0
        self.trail_stop = None


print("🌙 Moon Dev: Launching backtest...")
bt = Backtest(
    data, CompressionTrail,
    cash=1_000_000,
    commission=0.001,
    exclusive_orders=True
)
stats = bt.run()
print(stats)
print(stats._strategy)