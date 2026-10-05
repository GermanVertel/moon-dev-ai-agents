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

print("🌙 Moon Dev HarmonicVolatility Backtest Loading... ✨")
print(f"📊 Data shape: {data.shape}")
print(f"🚀 Price range: {data['Close'].min():.2f} - {data['Close'].max():.2f}")


class HarmonicVolatility(Strategy):
    # Parameters
    atr_period_1 = 10
    atr_period_2 = 20
    atr_period_3 = 40
    baseline_period = 20
    oscillator_period = 14
    band_multiplier = 1.5
    risk_pct = 0.01
    atr_stop_mult = 0.5
    time_exit_bars = 80

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low

        # Multi-period ATR for harmonic mean
        self.atr1 = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period_1)
        self.atr2 = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period_2)
        self.atr3 = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period_3)

        # Harmonic mean of ATRs: 3 / (1/a + 1/b + 1/c)
        def harm_mean(a, b, c):
            a = np.asarray(a, dtype=float)
            b = np.asarray(b, dtype=float)
            c = np.asarray(c, dtype=float)
            denom = (1.0 / np.where(a == 0, np.nan, a) +
                     1.0 / np.where(b == 0, np.nan, b) +
                     1.0 / np.where(c == 0, np.nan, c))
            return 3.0 / denom

        self.harm_atr = self.I(harm_mean, self.atr1, self.atr2, self.atr3)

        # Baseline: Hull Moving Average (HMA)
        def hma(series, period):
            series = pd.Series(series)
            half = int(period / 2)
            sqrt_p = int(np.sqrt(period))
            wma_half = talib.WMA(series.values, timeperiod=half)
            wma_full = talib.WMA(series.values, timeperiod=period)
            raw = 2 * wma_half - wma_full
            return talib.WMA(raw, timeperiod=sqrt_p)

        self.baseline = self.I(hma, close, self.baseline_period)

        # Harmonic oscillator: Fisher Transform for cycle timing
        def fisher(high, low, period):
            high = pd.Series(high)
            low = pd.Series(low)
            hl2 = (high + low) / 2
            mn = hl2.rolling(period).min()
            mx = hl2.rolling(period).max()
            rng = (mx - mn).replace(0, np.nan)
            val = 2 * ((hl2 - mn) / rng - 0.5)
            val = val.fillna(0)
            val = val.clip(-0.999, 0.999)
            fish = 0.5 * np.log((1 + val) / (1 - val))
            return fish.values

        self.fisher = self.I(fisher, high, low, self.oscillator_period)

        # Bands
        def upper_band(baseline, harm_atr, mult):
            baseline = np.asarray(baseline, dtype=float)
            harm_atr = np.asarray(harm_atr, dtype=float)
            # Dynamic multiplier: expand when fisher is extreme (volatility regime)
            return baseline + mult * harm_atr

        def lower_band(baseline, harm_atr, mult):
            baseline = np.asarray(baseline, dtype=float)
            harm_atr = np.asarray(harm_atr, dtype=float)
            return baseline - mult * harm_atr

        self.upper = self.I(upper_band, self.baseline, self.harm_atr, self.band_multiplier)
        self.lower = self.I(lower_band, self.baseline, self.harm_atr, self.band_multiplier)

        self.entry_bar = 0
        self.entry_price = 0
        self.stop_price = 0
        self.target_price = 0

    def next(self):
        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        upper = self.upper[-1]
        lower = self.lower[-1]
        baseline = self.baseline[-1]
        fisher = self.fisher[-1]
        atr = self.harm_atr[-1]

        if np.isnan(upper) or np.isnan(lower) or np.isnan(baseline) or np.isnan(fisher) or np.isnan(atr):
            return

        # Manage open position
        if self.position:
            bars_held = len(self.data) - self.entry_bar

            if self.position.is_long:
                # Trailing stop via Chandelier-like: high - atr*mult
                trail = high - atr * 2.0
                if trail > self.stop_price:
                    self.stop_price = max(self.stop_price, trail)

                # Exit at baseline (mean reversion target)
                if price >= self.target_price or price <= self.stop_price:
                    self.position.close()
                    print(f"🌙 EXIT LONG @ {price:.2f} | target={self.target_price:.2f} stop={self.stop_price:.2f} ✨")
                elif bars_held >= self.time_exit_bars:
                    self.position.close()
                    print(f"⏰ TIME EXIT LONG @ {price:.2f} after {bars_held} bars 🌙")

            elif self.position.is_short:
                trail = low + atr * 2.0
                if trail < self.stop_price or self.stop_price == 0:
                    self.stop_price = min(self.stop_price, trail) if self.stop_price != 0 else trail

                if price <= self.target_price or price >= self.stop_price:
                    self.position.close()
                    print(f"🌙 EXIT SHORT @ {price:.2f} | target={self.target_price:.2f} stop={self.stop_price:.2f} ✨")
                elif bars_held >= self.time_exit_bars:
                    self.position.close()
                    print(f"⏰ TIME EXIT SHORT @ {price:.2f} after {bars_held} bars 🌙")
            return

        # Entry logic
        # Long: price below lower band + fisher oversold
        if price < lower and fisher < -0.5:
            stop = lower - atr * self.atr_stop_mult
            risk = price - stop
            if risk <= 0:
                return
            size = int(round((self.equity * self.risk_pct) / risk))
            if size <= 0:
                return
            self.buy(size=size)
            self.entry_bar = len(self.data)
            self.entry_price = price
            self.stop_price = stop
            self.target_price = baseline
            print(f"🚀 LONG ENTRY @ {price:.2f} | lower={lower:.2f} fisher={fisher:.2f} stop={stop:.2f} target={baseline:.2f} size={size} 🌙")

        # Short: price above upper band + fisher overbought
        elif price > upper and fisher > 0.5:
            stop = upper + atr * self.atr_stop_mult
            risk = stop - price
            if risk <= 0:
                return
            size = int(round((self.equity * self.risk_pct) / risk))
            if size <= 0:
                return
            self.sell(size=size)
            self.entry_bar = len(self.data)
            self.entry_price = price
            self.stop_price = stop
            self.target_price = baseline
            print(f"🔻 SHORT ENTRY @ {price:.2f} | upper={upper:.2f} fisher={fisher:.2f} stop={stop:.2f} target={baseline:.2f} size={size} 🌙")


bt = Backtest(data, HarmonicVolatility, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)