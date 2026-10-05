import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Load and clean data
data = pd.read_csv("/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv")
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Map columns to backtesting.py requirements
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
data = data.dropna()

print("🌙 Moon Dev data loaded! Shape:", data.shape)
print("🚀 First few rows:\n", data.head())


class InversePivotTrigger(Strategy):
    """
    InversePivotTrigger Strategy 🌙
    Trades based on the previous day's open-high midpoint (PivotMid).
    Enters when close < PivotMid with volume confirmation.
    Exits when close reclaims PivotMid, or on time stop.
    """

    # Strategy parameters
    adv_lookback = 20          # ADV lookback in days
    vol_threshold_pct = 0.10   # 10% of ADV
    regime_sma = 50            # 50-day SMA regime filter
    stop_loss_pct = 0.03       # 3% hard stop
    time_stop_days = 5         # Time-based exit
    risk_pct = 0.02            # 2% account risk per trade
    position_size_cap = 0.50   # 50% max equity allocation (2x leverage safety)

    def init(self):
        print("🌙✨ Initializing InversePivotTrigger indicators...")

        # Daily aggregation is required for daily-level indicators.
        # We compute rolling daily pivot mid using the previous day's Open & High.
        # Since data is intraday (15m), we approximate by using rolling max of High
        # and a shifted open over the daily-equivalent window.
        # Bars per day = 96 (15m bars in a 24h day)
        bars_per_day = 96

        # Previous day's Open: shift Open by bars_per_day
        self.prev_open = self.I(
            lambda x: pd.Series(x).shift(bars_per_day).values,
            self.data.Open,
            name="PrevOpen"
        )

        # Previous day's High: rolling max of High over 1 day, shifted by 1 day
        self.prev_high = self.I(
            lambda x: pd.Series(x).rolling(bars_per_day).max().shift(bars_per_day).values,
            self.data.High,
            name="PrevHigh"
        )

        # PivotMid = (PrevOpen + PrevHigh) / 2
        self.pivot_mid = self.I(
            lambda o, h: (pd.Series(o) + pd.Series(h)) / 2.0,
            self.prev_open, self.prev_high,
            name="PivotMid"
        )

        # ADV: rolling mean of Volume over adv_lookback days (in bars)
        adv_bars = self.adv_lookback * bars_per_day
        self.adv = self.I(talib.SMA, self.data.Volume, timeperiod=adv_bars, name="ADV")

        # 50-day SMA (regime filter)
        sma_bars = self.regime_sma * bars_per_day
        self.sma50 = self.I(talib.SMA, self.data.Close, timeperiod=sma_bars, name="SMA50")

        # ATR for optional stops
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14, name="ATR")

        print("🚀 Indicators ready! PivotMid, ADV, SMA50, ATR initialized.")

    def next(self):
        # Skip if indicators not ready
        if len(self.data) < 2:
            return
        if pd.isna(self.pivot_mid[-1]) or pd.isna(self.adv[-1]) or pd.isna(self.sma50[-1]):
            return

        price = self.data.Close[-1]
        pivot = self.pivot_mid[-1]
        vol = self.data.Volume[-1]
        adv = self.adv[-1]
        sma = self.sma50[-1]

        # ================= POSITION MANAGEMENT =================
        if self.position:
            # Track entry bar for time stop
            if not hasattr(self, 'entry_bar'):
                self.entry_bar = len(self.data)

            bars_held = len(self.data) - self.entry_bar
            bars_per_day = 96
            days_held = bars_held / bars_per_day

            # Exit condition 1: Close back above PivotMid (setup invalidated)
            if price > pivot:
                print(f"🌙 EXIT: Close {price:.2f} reclaimed PivotMid {pivot:.2f}. Closing position.")
                self.position.close()
                self.entry_bar = None
                return

            # Exit condition 2: Time stop
            if days_held >= self.time_stop_days:
                print(f"⏰ TIME STOP: Held {days_held:.2f} days. Closing position.")
                self.position.close()
                self.entry_bar = None
                return

        # ================= ENTRY LOGIC =================
        if not self.position:
            # Regime filter: only take signals in downtrend (price below 50-day SMA)
            regime_ok = price < sma

            # Condition A: Close < PivotMid
            cond_a = price < pivot

            # Condition B: Volume > threshold * ADV
            cond_b = vol > (self.vol_threshold_pct * adv)

            if cond_a and cond_b and regime_ok:
                # Position sizing: risk-based with cap
                risk_amount = self.equity * self.risk_pct
                stop_price = price * (1 + self.stop_loss_pct)
                risk_per_unit = stop_price - price

                if risk_per_unit > 0:
                    size = int(round(risk_amount / risk_per_unit))
                    # Cap at 50% of equity (2x leverage safety)
                    max_size = int((self.equity * self.position_size_cap) / price)
                    size = min(size, max_size)
                    size = max(size, 1)

                    print(f"🌙✨ ENTRY SIGNAL: Close {price:.2f} < PivotMid {pivot:.2f} | "
                          f"Vol {vol:.2f} > {self.vol_threshold_pct*adv:.2f} | "
                          f"Regime OK (below SMA50 {sma:.2f})")
                    print(f"🚀 BUY {size} units @ ~{price:.2f} | Stop @ {stop_price:.2f}")
                    self.buy(size=size)
                    self.entry_bar = len(self.data)
            else:
                if cond_a and not cond_b:
                    pass  # Volume filter not met
                if cond_a and cond_b and not regime_ok:
                    pass  # Regime filter not met


# ================= RUN BACKTEST =================
print("🌙 Running initial backtest with default parameters...")
bt = Backtest(
    data,
    InversePivotTrigger,
    cash=1_000_000,
    commission=0.002,
    exclusive_orders=True
)

stats = bt.run()
print(stats)
print(stats._strategy)