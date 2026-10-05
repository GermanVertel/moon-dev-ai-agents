import pandas as pd
import numpy as np
import talib
from backtesting import Backtest, Strategy

print("🌙 Moon Dev Backtest AI initializing VolatilityTermStructure strategy...")
print("✨ Loading data and preparing lunar charts...")

data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'datetime': 'datetime',
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
print(f"🚀 Data loaded: {len(data)} rows from {data.index[0]} to {data.index[-1]}")


class VolatilityTermStructure(Strategy):
    # Strategy parameters
    z_entry = 2.0          # Entry threshold
    z_exit = 0.5           # Exit threshold (mean reversion)
    z_stop = 3.5           # Hard stop
    zscore_window = 60     # Rolling window for z-score
    time_stop_bars = 40    # ~10 trading days on 15m bars
    risk_pct = 0.02        # Risk per trade
    size = 0.95            # Fraction of equity (0 < size < 1)

    def init(self):
        print("🌙 Initializing indicators for VolatilityTermStructure...")

        close = pd.Series(self.data.Close)
        log_ret = np.log(close / close.shift(1)).fillna(0)

        # Near-term vol proxy (short window)
        self.vol_near = self.I(
            lambda: log_ret.rolling(10).std().bfill().values * np.sqrt(96),
            name='Vol_Near'
        )
        # Far-term vol proxy (long window)
        self.vol_far = self.I(
            lambda: log_ret.rolling(60).std().bfill().values * np.sqrt(96),
            name='Vol_Far'
        )

        # Spread = near - far
        self.spread = self.I(
            lambda: (self.vol_near - self.vol_far),
            name='IV_Spread'
        )

        spread_series = pd.Series(self.spread)

        self.spread_mean = self.I(
            lambda: spread_series.rolling(self.zscore_window).mean().bfill().values,
            name='Spread_Mean'
        )
        self.spread_std = self.I(
            lambda: spread_series.rolling(self.zscore_window).std().bfill().values,
            name='Spread_Std'
        )

        # Z-score
        self.zscore = self.I(
            lambda: ((spread_series - spread_series.rolling(self.zscore_window).mean())
                     / spread_series.rolling(self.zscore_window).std()).bfill().values,
            name='ZScore'
        )

        # Realized vol regime filter
        self.realized_vol = self.I(
            lambda: log_ret.rolling(20).std().bfill().values * np.sqrt(96),
            name='Realized_Vol'
        )

        print("✨ Indicators initialized. Ready for lunar trading! 🌙")

    def next(self):
        price = self.data.Close[-1]

        # Regime filter: stand down if realized vol is in extreme spike
        rv_series = pd.Series(self.realized_vol)
        rv_90 = rv_series.rolling(200, min_periods=50).quantile(0.90).iloc[-1]
        regime_ok = True
        if not np.isnan(rv_90) and self.realized_vol[-1] > rv_90:
            regime_ok = False

        z = self.zscore[-1]

        # ---- ENTRY LOGIC ----
        if not self.position:
            if regime_ok and not np.isnan(z):
                if z >= self.z_entry:
                    print(f"🌙 SHORT spread signal! z={z:.2f} | price={price:.2f} 🚀")
                    sl = price * (1 + 0.02)
                    tp = price * (1 - 0.04)
                    self.sell(size=self.size, sl=sl, tp=tp)
                elif z <= -self.z_entry:
                    print(f"🌙 LONG spread signal! z={z:.2f} | price={price:.2f} 🚀")
                    sl = price * (1 - 0.02)
                    tp = price * (1 + 0.04)
                    self.buy(size=self.size, sl=sl, tp=tp)

        # ---- EXIT LOGIC ----
        else:
            bars_held = len(self.data) - self.trades[-1].entry_bar if self.trades else 0

            if self.position.is_long:
                if z > -self.z_exit or z < -self.z_stop or bars_held >= self.time_stop_bars:
                    print(f"🌙 Closing LONG | z={z:.2f} bars={bars_held} 💫")
                    self.position.close()
            elif self.position.is_short:
                if z < self.z_exit or z > self.z_stop or bars_held >= self.time_stop_bars:
                    print(f"🌙 Closing SHORT | z={z:.2f} bars={bars_held} 💫")
                    self.position.close()


print("🚀 Launching Moon Dev backtest engine...")
bt = Backtest(data, VolatilityTermStructure, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙✨ Backtest complete! To the moon! 🚀")