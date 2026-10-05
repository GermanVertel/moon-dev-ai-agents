import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Load and prepare data
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
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
data = data[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)

print("🌙 Moon Dev data loaded! Shape:", data.shape)
print("✨ First few rows:\n", data.head())


class KineticConvergence(Strategy):
    # Strategy parameters
    tenkan_period = 9
    kijun_period = 26
    senkou_b_period = 52
    atr_period = 14
    spread_window = 50
    spread_percentile = 15
    atr_stop_mult = 2.0
    time_stop_bars = 20
    risk_pct = 0.02
    size = 1_000_000

    def init(self):
        high = self.data.High
        low = self.data.Low
        close = self.data.Close
        volume = self.data.Volume

        # Ichimoku components
        self.tenkan = self.I(talib.SMA, (high + low) / 2, timeperiod=self.tenkan_period)
        # Approximate Kijun using rolling max/min via talib MAX/MIN then average
        hh26 = self.I(talib.MAX, high, timeperiod=self.kijun_period)
        ll26 = self.I(talib.MIN, low, timeperiod=self.kijun_period)
        self.kijun = self.I(lambda a, b: (a + b) / 2, hh26, ll26)

        hh9 = self.I(talib.MAX, high, timeperiod=self.tenkan_period)
        ll9 = self.I(talib.MIN, low, timeperiod=self.tenkan_period)
        self.tenkan = self.I(lambda a, b: (a + b) / 2, hh9, ll9)

        # Senkou Span B (for cloud bias)
        hh52 = self.I(talib.MAX, high, timeperiod=self.senkou_b_period)
        ll52 = self.I(talib.MIN, low, timeperiod=self.senkou_b_period)
        self.senkou_b = self.I(lambda a, b: (a + b) / 2, hh52, ll52)

        # Spread
        self.spread = self.I(lambda t, k: np.abs(t - k), self.tenkan, self.kijun)

        # Rolling percentile lower band of spread
        def rolling_lower_band(spread_series, window, pct):
            s = pd.Series(spread_series)
            return s.rolling(window).quantile(pct / 100.0).values

        self.lower_band = self.I(
            rolling_lower_band,
            self.spread,
            self.spread_window,
            self.spread_percentile
        )

        # ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        # Trackers
        self.entry_bar = None
        self.entry_price = None
        self.highest_close = None
        self.trailing_stop = None

        print("🚀 KineticConvergence indicators initialized!")
        print(f"🌙 Tenkan period: {self.tenkan_period}, Kijun period: {self.kijun_period}")
        print(f"✨ Spread window: {self.spread_window}, Percentile: {self.spread_percentile}")

    def next(self):
        price = self.data.Close[-1]
        vol = self.data.Volume[-1]
        atr = self.atr[-1]
        tenkan = self.tenkan[-1]
        kijun = self.kijun[-1]
        spread = self.spread[-1]
        lower_band = self.lower_band[-1]

        # Skip if indicators not ready
        if (np.isnan(atr) or np.isnan(tenkan) or np.isnan(kijun) or
                np.isnan(spread) or np.isnan(lower_band) or atr <= 0):
            return

        # ---- Manage open position ----
        if self.position:
            self.highest_close = max(self.highest_close, price)

            # Update trailing stop: 2*ATR from highest close
            new_trailing = self.highest_close - self.atr_stop_mult * atr
            if self.trailing_stop is None or new_trailing > self.trailing_stop:
                self.trailing_stop = new_trailing

            # Stop loss hit
            if price <= self.trailing_stop:
                print(f"🛑 Moon Dev STOP HIT! Price={price:.2f} <= Trail={self.trailing_stop:.2f} 🌙")
                self.position.close()
                self._reset_trackers()
                return

            # Bearish crossover exit
            if tenkan < kijun:
                print(f"📉 Bearish crossover! Tenkan={tenkan:.2f} < Kijun={kijun:.2f} — exiting 🌙")
                self.position.close()
                self._reset_trackers()
                return

            # Time stop
            bars_held = len(self.data) - 1 - self.entry_bar
            if bars_held >= self.time_stop_bars:
                print(f"⏰ Time stop hit after {bars_held} bars — exiting 🌙")
                self.position.close()
                self._reset_trackers()
                return

            return

        # ---- Entry logic ----
        # Spread compression
        compression = spread <= lower_band
        # Volume confirmation: volume > 0.5 * ATR
        volume_ok = vol > 0.5 * atr
        # Bullish bias: price > kijun and tenkan > kijun
        bullish_bias = (price > kijun) and (tenkan > kijun)
        # Cloud bias: price above Senkou B
        cloud_ok = price > self.senkou_b[-1] if not np.isnan(self.senkou_b[-1]) else True

        if compression and volume_ok and bullish_bias and cloud_ok:
            # Risk-based position sizing
            risk_per_unit = self.atr_stop_mult * atr
            if risk_per_unit <= 0:
                return
            equity = self.equity
            risk_amount = equity * self.risk_pct
            position_size = risk_amount / risk_per_unit
            position_size = int(round(position_size))
            if position_size <= 0:
                position_size = 1

            print(f"🌙✨ KINETIC CONVERGENCE SIGNAL! ✨🌙")
            print(f"   Price={price:.2f} | Tenkan={tenkan:.2f} | Kijun={kijun:.2f}")
            print(f"   Spread={spread:.4f} <= LowerBand={lower_band:.4f}")
            print(f"   Volume={vol:.2f} > 0.5*ATR={0.5*atr:.2f}")
            print(f"   ATR={atr:.2f} | Risk/unit={risk_per_unit:.2f}")
            print(f"   Position size={position_size} 🚀")

            self.buy(size=position_size)
            self.entry_bar = len(self.data) - 1
            self.entry_price = price
            self.highest_close = price
            self.trailing_stop = price - self.atr_stop_mult * atr

    def _reset_trackers(self):
        self.entry_bar = None
        self.entry_price = None
        self.highest_close = None
        self.trailing_stop = None


print("🌙🚀 Starting Moon Dev KineticConvergence backtest...")
bt = Backtest(data, KineticConvergence, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)