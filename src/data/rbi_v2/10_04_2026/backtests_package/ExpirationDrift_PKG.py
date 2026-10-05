import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# 🌙 Moon Dev's ExpirationDrift Backtest 🚀

data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"

print("🌙✨ Loading cosmic data from the Moon Dev vault...")
data = pd.read_csv(data_path)

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Map to backtesting requirements
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

print(f"🌙✨ Data loaded! {len(data)} rows of cosmic price action 🚀")


class ExpirationDrift(Strategy):
    """
    🌙 ExpirationDrift Strategy 🌙
    Exploits post-expiration drift after monthly options expiry.
    Since we don't have options OI/GEX data, we approximate the
    'post-expiry release' using a monthly third-Friday window and
    a volatility-expansion filter (ATR rising) + breakout of the
    pre-window range.
    """
    # Risk parameters
    risk_pct = 0.02          # 2% risk per trade
    atr_period = 14
    lookback = 20            # range lookback

    # Targets / stops
    target_pct = 0.10        # +10% target
    hard_stop_pct = 0.035    # -3.5% hard stop
    breakeven_at = 0.03      # move stop to BE after +3%
    trail_at = 0.05          # start trailing after +5%
    max_days = 3             # time stop: 3 trading days

    def init(self):
        print("🌙✨ Initializing ExpirationDrift indicators...")
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close,
                          timeperiod=self.atr_period)
        self.atr_ma = self.I(talib.SMA, self.atr, timeperiod=self.atr_period)
        self.range_high = self.I(talib.MAX, self.data.High, timeperiod=self.lookback)
        self.range_low = self.I(talib.MIN, self.data.Low, timeperiod=self.lookback)

        # Track trade state
        self.entry_price = None
        self.entry_bar = None
        self.stop_price = None
        self.be_moved = False
        self.trail_active = False

    def _is_post_expiry_window(self, dt):
        """
        Returns True if dt is within the 2-hour window after the
        third-Friday monthly expiry (approx 4:00 PM ET = 20:00 UTC).
        We approximate: third Friday of the month, 20:00-22:00 UTC.
        """
        # Third Friday
        first_day = dt.replace(day=1)
        # weekday(): Monday=0 ... Friday=4
        offset = (4 - first_day.weekday()) % 7
        first_friday = first_day + pd.Timedelta(days=offset)
        third_friday = first_friday + pd.Timedelta(days=14)
        if dt.date() != third_friday.date():
            return False
        # 2-hour window after 20:00 UTC (4 PM ET)
        hour = dt.hour
        return 20 <= hour < 22

    def next(self):
        price = self.data.Close[-1]
        dt = self.data.index[-1]

        # -------- MANAGE OPEN POSITION --------
        if self.position:
            pnl_pct = (price - self.entry_price) / self.entry_price

            # Move stop to breakeven after +3%
            if not self.be_moved and pnl_pct >= self.breakeven_at:
                self.stop_price = max(self.stop_price, self.entry_price)
                self.be_moved = True
                print(f"🌙✨ Moving stop to BREAKEVEN @ {self.entry_price:.2f} 🚀")

            # Activate trailing stop after +5%
            if not self.trail_active and pnl_pct >= self.trail_at:
                self.trail_active = True
                print(f"🌙🚀 +5% reached! Activating trailing stop for {self.data.index[-1]}")

            if self.trail_active:
                new_trail = price * (1 - 0.02)  # 2% trailing
                if new_trail > self.stop_price:
                    self.stop_price = new_trail
                    print(f"🌙📈 Trailing stop raised to {self.stop_price:.2f}")

            # Target hit
            if pnl_pct >= self.target_pct:
                print(f"🌙🎯 TARGET HIT! +{pnl_pct*100:.2f}% @ {price:.2f} — closing long 🚀")
                self.position.close()
                self._reset_state()
                return

            # Hard stop hit
            if price <= self.stop_price:
                print(f"🌙🛑 STOP HIT @ {price:.2f} (stop={self.stop_price:.2f}) — exiting")
                self.position.close()
                self._reset_state()
                return

            # Time stop: 3 trading days = 3 * 96 15-min bars (approx)
            bars_held = len(self.data) - 1 - self.entry_bar
            if bars_held >= self.max_days * 96:
                print(f"🌙⏰ TIME STOP (3 days) reached — closing @ {price:.2f}")
                self.position.close()
                self._reset_state()
                return
            return

        # -------- LOOK FOR ENTRY --------
        if not self._is_post_expiry_window(dt):
            return

        # Volatility expansion filter: ATR > ATR MA (no collapsing vol)
        if self.atr[-1] <= self.atr_ma[-1]:
            return

        # Long trigger: close breaks above prior range high
        # 🌙 Manual crossover detection (no backtesting.lib)
        if price > self.range_high[-2]:
            # Position sizing: risk 2% of equity, stop 3.5%
            equity = self.equity
            risk_amount = equity * self.risk_pct
            stop_dist = price * self.hard_stop_pct
            size = int(round(risk_amount / stop_dist))
            if size < 1:
                size = 1

            print(f"🌙🚀 POST-EXPIRY BREAKOUT! Entering LONG @ {price:.2f} "
                  f"| size={size} | ATR={self.atr[-1]:.2f} > ATR_MA={self.atr_ma[-1]:.2f} ✨")
            self.buy(size=size)
            self.entry_price = price
            self.entry_bar = len(self.data) - 1
            self.stop_price = price * (1 - self.hard_stop_pct)
            self.be_moved = False
            self.trail_active = False

    def _reset_state(self):
        self.entry_price = None
        self.entry_bar = None
        self.stop_price = None
        self.be_moved = False
        self.trail_active = False


print("🌙✨ Launching ExpirationDrift backtest on the Moon... 🚀")
bt = Backtest(data, ExpirationDrift, cash=1_000_000, commission=0.0002)
stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙✨ Backtest complete! To the moon! 🚀")