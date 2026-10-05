import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

print("🌙 Moon Dev Backtest AI loading ContrarianFundingBreakout... ✨🚀")

# Load data
data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Ensure proper column mapping
data = data.rename(columns={
    'datetime': 'Datetime',
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

data['Datetime'] = pd.to_datetime(data['Datetime'])
data = data.set_index('Datetime')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

print(f"🌙 Data loaded: {len(data)} bars ✨")

# Since funding rate isn't in the data, we'll synthesize a proxy funding rate from price action.
# We'll use negative returns as a proxy for bearish sentiment / low funding.
data['Funding_Proxy'] = -data['Close'].pct_change().fillna(0)

print("🌙 Synthesized funding rate proxy from negative returns ✨")


class ContrarianFundingBreakout(Strategy):
    # Parameters
    funding_lookback = 30
    breakout_lookback = 10
    atr_period = 14
    atr_multiplier = 2.0
    retracement_pct = 0.5
    risk_per_trade = 0.01  # 1% of equity

    def init(self):
        print("🌙 Initializing ContrarianFundingBreakout indicators... ✨")
        self.funding = self.I(lambda x: x, self.data.Funding_Proxy, name='Funding')
        self.funding_min_30 = self.I(
            talib.MIN, self.data.Funding_Proxy, timeperiod=self.funding_lookback,
            name='Funding_Min_30'
        )
        self.high_10 = self.I(
            talib.MAX, self.data.High, timeperiod=self.breakout_lookback,
            name='High_10'
        )
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close,
                          timeperiod=self.atr_period, name='ATR')

        # State variables
        self.entry_price = None
        self.peak_price = None
        self.initial_stop = None
        self.trailing_stop = None

    def next(self):
        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]

        # If in a position, manage exits
        if self.position:
            if self.peak_price is None:
                # Recover state from last trade if lost
                try:
                    self.entry_price = self.trades[-1].entry_price
                    self.peak_price = self.entry_price
                except Exception:
                    self.entry_price = price
                    self.peak_price = price

            # Update peak
            if high > self.peak_price:
                self.peak_price = high

            # Update trailing volatility stop
            atr_val = self.atr[-1]
            if atr_val is not None and not np.isnan(atr_val):
                new_trail = self.peak_price - (self.atr_multiplier * atr_val)
                if self.trailing_stop is None or new_trail > self.trailing_stop:
                    self.trailing_stop = new_trail

            # Exit Rule A: 50% retracement from peak to entry
            if self.entry_price is not None and self.peak_price is not None:
                retracement_level = self.peak_price - (self.peak_price - self.entry_price) * self.retracement_pct
                if price <= retracement_level:
                    print(f"🌙 EXIT: 50% retracement hit at {price:.2f} (level {retracement_level:.2f}) 💫")
                    self.position.close()
                    self._reset_state()
                    return

            # Exit Rule B: Volatility stop
            if self.trailing_stop is not None and price <= self.trailing_stop:
                print(f"🌙 EXIT: Volatility stop hit at {price:.2f} (stop {self.trailing_stop:.2f}) 🛑")
                self.position.close()
                self._reset_state()
                return

            return

        # Entry logic - need enough history
        min_bars = max(self.funding_lookback, self.breakout_lookback, self.atr_period) + 2
        if len(self.data) < min_bars:
            return

        funding_now = self.funding[-1]
        funding_min = self.funding_min_30[-1]
        high_10_prev = self.high_10[-2]

        # Skip if indicators are NaN
        if (funding_now is None or funding_min is None or high_10_prev is None or
                np.isnan(funding_now) or np.isnan(funding_min) or np.isnan(high_10_prev)):
            return

        # Condition 1: funding at 30-day low (with small tolerance)
        funding_extreme = funding_now <= funding_min * 1.001

        # Condition 2: close breaks above prior 10-day high
        breakout = price > high_10_prev

        if funding_extreme and breakout:
            atr_val = self.atr[-1]
            if atr_val is None or np.isnan(atr_val) or atr_val <= 0:
                return

            stop_distance = self.atr_multiplier * atr_val
            if stop_distance <= 0:
                return

            # Position sizing: risk 1% of equity, size = risk_amount / stop_distance
            equity = self.equity
            risk_amount = equity * self.risk_per_trade
            position_size_units = risk_amount / stop_distance

            # Use fraction of equity as size (0 < size < 1)
            size_fraction = (position_size_units * price) / equity
            # Cap fraction to avoid over-leverage
            size_fraction = min(max(size_fraction, 0.001), 0.99)

            self.entry_price = price
            self.peak_price = price
            self.initial_stop = price - stop_distance
            self.trailing_stop = self.initial_stop

            print(f"🚀🌙 ENTRY LONG: price={price:.2f}, size_frac={size_fraction:.4f}, "
                  f"stop={self.trailing_stop:.2f}, ATR={atr_val:.2f}, "
                  f"funding={funding_now:.6f} (min={funding_min:.6f}) ✨")

            self.buy(size=size_fraction)

    def _reset_state(self):
        self.entry_price = None
        self.peak_price = None
        self.initial_stop = None
        self.trailing_stop = None


print("🌙 Setting up backtest... ✨")
bt = Backtest(
    data,
    ContrarianFundingBreakout,
    cash=1_000_000,
    commission=0.002,
    exclusive_orders=True
)

print("🚀 Running backtest with 1,000,000 size... 🌙")
stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙 Moon Dev Backtest Complete! ✨🚀")