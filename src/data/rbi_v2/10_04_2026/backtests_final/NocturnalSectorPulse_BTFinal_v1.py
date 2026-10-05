import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Load data
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

# Set datetime index
if 'datetime' in data.columns:
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')

print("🌙✨ NocturnalSectorPulse Backtest Initializing... 🚀")
print(f"📊 Data loaded: {len(data)} rows")
print(f"📅 Date range: {data.index[0]} to {data.index[-1]}")


class NocturnalSectorPulse(Strategy):
    """
    🌙 NocturnalSectorPulse Strategy 🌙

    Overnight session breakout trading with sector rotation analysis.
    - 60-min rolling high breakout
    - EROD (relative strength) filter
    - ATR-adjusted stop loss (max 5%)
    - 3:1 R:R take profit
    - Time-based session exit
    """

    # Strategy parameters
    high_period = 4          # 60-min high on 15m data (4 bars = 60 min)
    atr_period = 14
    ma_period = 50
    erod_window = 20         # rolling window for EROD proxy
    risk_pct = 0.01          # 1% risk per trade
    max_stop_pct = 0.05      # 5% max stop
    atr_mult = 1.5           # ATR multiplier for stop
    rr_ratio = 3.0           # 3:1 reward:risk
    position_size = 1_000_000  # fixed size

    def init(self):
        print("🌙 Initializing indicators...")

        # 60-minute rolling high (talib.MAX)
        self.rolling_high = self.I(talib.MAX, self.data.High, timeperiod=self.high_period)

        # 30-minute rolling low (trailing exit) - talib.MIN
        self.rolling_low_30 = self.I(talib.MIN, self.data.Low, timeperiod=2)

        # ATR(14) - talib.ATR
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period)

        # 50-period MA on Close (proxy for sector ETF trend) - talib.SMA
        self.ma50 = self.I(talib.SMA, self.data.Close, timeperiod=self.ma_period)

        # EROD proxy: rolling return of asset vs its own long-term baseline
        # EROD = pct_change(erod_window) - pct_change(erod_window * 2)
        # Implemented as a numpy function to avoid backtesting.lib usage
        close_arr = np.asarray(self.data.Close, dtype=float)
        short_w = self.erod_window
        long_w = self.erod_window * 2

        def _erod(x):
            arr = np.asarray(x, dtype=float)
            short_ret = np.full_like(arr, np.nan, dtype=float)
            long_ret = np.full_like(arr, np.nan, dtype=float)
            short_ret[short_w:] = (arr[short_w:] - arr[:-short_w]) / arr[:-short_w]
            long_ret[long_w:] = (arr[long_w:] - arr[:-long_w]) / arr[:-long_w]
            erod = short_ret - long_ret
            erod = np.where(np.isnan(erod), 0.0, erod)
            return erod

        self.erod = self.I(_erod, self.data.Close, name="EROD")

        print("✨ Indicators ready: rolling_high, atr, ma50, erod 🌙")

    def next(self):
        price = self.data.Close[-1]
        high_60 = self.rolling_high[-1]
        atr_val = self.atr[-1]
        ma50_val = self.ma50[-1]
        erod_val = self.erod[-1]

        # Skip if indicators not warmed up
        if np.isnan(high_60) or np.isnan(atr_val) or np.isnan(ma50_val):
            return

        # Overnight session detection (using hour from index)
        try:
            hour = self.data.index[-1].hour
        except Exception:
            hour = 0

        # Overnight: 18:00 - 09:30 (proxy: hours >= 18 or < 9)
        is_overnight = (hour >= 18) or (hour < 9)

        # ---- ENTRY LOGIC ----
        if not self.position and is_overnight:
            # Breakout above 60-min high
            breakout = price > high_60

            # EROD positive (relative strength)
            erod_positive = erod_val > 0

            # Sector trend filter (proxy: price above 50 MA)
            sector_uptrend = price > ma50_val

            if breakout and erod_positive and sector_uptrend:
                # ATR-adjusted stop
                atr_stop_dist = self.atr_mult * atr_val
                max_stop_dist = self.max_stop_pct * price
                stop_dist = min(atr_stop_dist, max_stop_dist)

                stop_price = price - stop_dist
                tp_price = price + (stop_dist * self.rr_ratio)

                # Position sizing: fraction of equity (0 < size < 1)
                # Risk-based sizing: risk_pct of equity / stop distance
                if stop_dist > 0 and price > 0:
                    size = self.risk_pct * self.equity / stop_dist
                    size = size / price  # convert units to fraction of equity
                else:
                    size = 0.95
                # Clamp fraction to valid range
                if size <= 0 or size >= 1:
                    size = 0.95

                print(f"🚀🌙 MOON DEV ENTRY SIGNAL! 🚀")
                print(f"   Price: {price:.2f} | 60m High: {high_60:.2f}")
                print(f"   EROD: {erod_val:.4f} | MA50: {ma50_val:.2f}")
                print(f"   Stop: {stop_price:.2f} | TP: {tp_price:.2f}")
                print(f"   Size (fraction): {size:.4f}")

                self.buy(size=size, sl=stop_price, tp=tp_price)

        # ---- EXIT LOGIC ----
        elif self.position:
            # Time-based exit: if we exit overnight session
            if not is_overnight:
                print(f"🌅 Session ended - closing position at {price:.2f}")
                self.position.close()
                return

            # Trailing exit using 30-min lows
            low_30 = self.rolling_low_30[-1]
            if not np.isnan(low_30) and price < low_30:
                print(f"📉 30-min low breach - trailing exit at {price:.2f}")
                self.position.close()


# Run backtest
print("🌙✨ Launching NocturnalSectorPulse backtest... 🚀")
bt = Backtest(
    data,
    NocturnalSectorPulse,
    cash=1_000_000,
    commission=0.002
)

stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙 Moon Dev Backtest Complete! ✨")