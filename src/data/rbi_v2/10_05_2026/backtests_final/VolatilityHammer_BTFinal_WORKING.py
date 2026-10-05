import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Load data
data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Ensure proper column mapping
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
data = data.dropna()

print("🌙✨ Moon Dev VolatilityHammer Backtest Initializing... 🚀")
print(f"📊 Data loaded: {len(data)} bars from {data.index[0]} to {data.index[-1]}")


def hammer_pattern(open_, high, low, close):
    """Detect Hammer candlestick pattern."""
    body = np.abs(close - open_)
    lower_shadow = np.minimum(open_, close) - low
    upper_shadow = high - np.maximum(open_, close)
    range_total = high - low

    # Avoid division by zero
    body_safe = np.where(body == 0, 1e-10, body)
    range_safe = np.where(range_total == 0, 1e-10, range_total)

    # Hammer conditions
    small_body = body <= 0.3 * range_safe
    long_lower = lower_shadow >= 2 * body_safe
    small_upper = upper_shadow <= 0.2 * range_safe

    return (small_body & long_lower & small_upper).astype(float)


def vix_proxy(close, high, low):
    """Proxy for VIX using realized volatility (annualized)."""
    returns = np.log(close / np.roll(close, 1))
    returns[0] = 0
    vol = np.std(returns) * np.sqrt(252) * 100
    return vol


class VolatilityHammer(Strategy):
    # Parameters
    vix_threshold = 20.0
    vix_exit_threshold = 15.0
    risk_pct = 0.02  # 2% risk per trade
    stop_loss_pct = 0.02  # 2% below hammer low
    take_profit_pct = 0.50  # 50% credit capture
    position_size = 1_000_000

    def init(self):
        print("🌙 Initializing indicators...")
        self.hammer = self.I(hammer_pattern, self.data.Open, self.data.High,
                             self.data.Low, self.data.Close, name="Hammer")

        # Realized volatility as VIX proxy (rolling) - using pure pandas/numpy, no backtesting.lib
        def rolling_vol(c):
            s = pd.Series(c)
            return (s.pct_change().rolling(20).std().values * np.sqrt(252) * 100)

        self.realized_vol = self.I(rolling_vol, self.data.Close, name="RealizedVol")

        self.entry_price = None
        self.stop_price = None
        self.target_price = None
        print("✨ Indicators ready!")

    def next(self):
        price = self.data.Close[-1]
        hammer_val = self.hammer[-1]
        vix_val = self.realized_vol[-1]

        if np.isnan(vix_val):
            return

        # Exit logic for open position
        if self.position:
            # Take profit (for long position, this is price up)
            if self.target_price and price >= self.target_price:
                print(f"🎯 Take profit hit at {price:.2f} | VIX proxy: {vix_val:.2f}")
                self.position.close()
                self.entry_price = None
                return

            # Stop loss
            if self.stop_price and price <= self.stop_price:
                print(f"🛑 Stop loss hit at {price:.2f} | VIX proxy: {vix_val:.2f}")
                self.position.close()
                self.entry_price = None
                return

            # VIX-based exit
            if vix_val < self.vix_exit_threshold:
                print(f"🌊 VIX proxy dropped below {self.vix_exit_threshold} ({vix_val:.2f}) - closing early")
                self.position.close()
                self.entry_price = None
                return

        # Entry logic
        if not self.position:
            if hammer_val > 0 and vix_val > self.vix_threshold:
                # Hammer low as reference
                hammer_low = self.data.Low[-1]
                stop = hammer_low * (1 - self.stop_loss_pct)
                # Target: 2:1 reward-to-risk approximation
                risk = price - stop
                target = price + risk * 2

                # Position sizing based on risk
                equity = self.equity
                risk_amount = equity * self.risk_pct
                size = risk_amount / risk if risk > 0 else 0
                size = int(round(size))
                if size < 1:
                    size = 1

                print(f"🔨 HAMMER detected! Entry: {price:.2f} | Hammer low: {hammer_low:.2f}")
                print(f"🚀 VIX proxy: {vix_val:.2f} > {self.vix_threshold} | Size: {size}")
                print(f"🛑 Stop: {stop:.2f} | 🎯 Target: {target:.2f}")

                self.buy(size=size)
                self.entry_price = price
                self.stop_price = stop
                self.target_price = target


# Run backtest
print("🌙🚀 Starting Moon Dev VolatilityHammer backtest...")
bt = Backtest(data, VolatilityHammer, cash=1_000_000, commission=0.002)
stats = bt.run()
print(stats)
print(stats._strategy)