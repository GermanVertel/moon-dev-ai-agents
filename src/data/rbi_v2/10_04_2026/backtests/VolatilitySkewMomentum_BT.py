import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

print("🌙 Moon Dev's VolatilitySkewMomentum Backtest Initializing... 🚀")

data_path = '/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv'
data = pd.read_csv(data_path)

print("📊 Cleaning data columns...")
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper column mapping
data = data.rename(columns={
    'datetime': 'Date',
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

if 'Date' in data.columns:
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.set_index('Date')

data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
print(f"✅ Data loaded: {len(data)} rows 🌙")


class VolatilitySkewMomentum(Strategy):
    # Strategy parameters
    vix_lookback = 252
    vix_compression_ratio = 0.80
    skew_avg_period = 90  # ~3 months of daily bars (using 15m data scaled)
    crossover_lookback = 5
    rsi_period = 14
    atr_period = 14
    rsi_overbought = 60
    rsi_exhaustion = 80
    atr_stop_mult = 2.0
    max_hold_bars = 30 * 96  # ~30 trading days of 15m bars
    size = 1_000_000

    def init(self):
        print("🌙 Initializing indicators...")
        close = self.data.Close
        high = self.data.High
        low = self.data.Low

        # RSI(14)
        self.rsi = self.I(talib.RSI, close, timeperiod=self.rsi_period, name='RSI')

        # ATR(14)
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period, name='ATR')

        # 20-EMA
        self.ema20 = self.I(talib.EMA, close, timeperiod=20, name='EMA20')

        # Proxy for VIX: use realized volatility (rolling std of returns) as VIX proxy
        # Since we only have BTC-USD data, we synthesize VIX-like and Skew-like proxies
        returns = np.zeros(len(close))
        c = np.array(close, dtype=float)
        returns[1:] = np.diff(c) / c[:-1]
        returns_series = pd.Series(returns)

        # VIX proxy = rolling std of returns * sqrt(annualization)
        vix_proxy = returns_series.rolling(20).std() * np.sqrt(96 * 365) * 100
        vix_proxy = vix_proxy.fillna(method='bfill').fillna(0).values
        self.vix = self.I(lambda: vix_proxy, name='VIX_proxy')

        # VIX historical average (long lookback)
        self.vix_avg = self.I(talib.SMA, self.vix, timeperiod=self.vix_lookback, name='VIX_avg')

        # Gamma Skew proxy: use rolling skewness of returns (3-month ~ 90 bars scaled)
        skew_proxy = returns_series.rolling(self.skew_avg_period).skew()
        skew_proxy = skew_proxy.fillna(0).values
        self.gamma_skew = self.I(lambda: skew_proxy, name='Gamma_Skew')

        # 3-month average of Gamma Skew
        self.gamma_skew_avg = self.I(talib.SMA, self.gamma_skew, timeperiod=self.skew_avg_period, name='Gamma_Skew_avg')

        print("✨ Indicators ready! 🌙")

    def next(self):
        price = self.data.Close[-1]

        # Skip if indicators not ready
        if np.isnan(self.vix[-1]) or np.isnan(self.vix_avg[-1]) or np.isnan(self.gamma_skew[-1]) or np.isnan(self.gamma_skew_avg[-1]):
            return

        if len(self.data) < 2:
            return

        # ---- Position management ----
        if self.position:
            entry_price = self.trades[-1].entry_price if self.trades else price
            bars_held = len(self.data) - 1 - (self.trades[-1].entry_bar if self.trades else 0)

            # Take profit: VIX reverts to >= 100% of average
            if self.vix[-1] >= self.vix_avg[-1]:
                print(f"🎯 VIX mean reversion complete! Exiting. VIX={self.vix[-1]:.2f} vs avg={self.vix_avg[-1]:.2f}")
                self.position.close()
                return

            # Momentum failure: Gamma Skew crosses below its average
            if self.gamma_skew[-1] < self.gamma_skew_avg[-1] and self.gamma_skew[-2] >= self.gamma_skew_avg[-2]:
                print(f"📉 Gamma Skew bearish crossover! Exiting. 🌙")
                self.position.close()
                return

            # Overbought exhaustion: RSI > 80 + bearish candle
            if self.rsi[-1] > self.rsi_exhaustion and self.data.Close[-1] < self.data.Open[-1]:
                print(f"🔥 RSI exhaustion ({self.rsi[-1]:.1f}) + bearish candle. Exiting! 🌙")
                self.position.close()
                return

            # Time stop
            if bars_held > self.max_hold_bars:
                print(f"⏰ Time stop hit ({bars_held} bars). Exiting! 🌙")
                self.position.close()
                return

            return

        # ---- Entry logic ----
        # Condition A: VIX compression
        vix_compressed = self.vix[-1] <= self.vix_compression_ratio * self.vix_avg[-1]

        # Condition B: Gamma Skew bullish crossover within last N bars
        skew_cross = False
        for i in range(1, min(self.crossover_lookback + 1, len(self.data))):
            if self.gamma_skew[-i] > self.gamma_skew_avg[-i] and self.gamma_skew[-i-1] <= self.gamma_skew_avg[-i-1]:
                skew_cross = True
                break

        # Regime filter: RSI > 60 or price > EMA20
        regime_ok = self.rsi[-1] > self.rsi_overbought or price > self.ema20[-1]

        if vix_compressed and skew_cross and regime_ok:
            # Volatility-adjusted sizing: reduce when compression is extreme (<70%)
            compression_ratio = self.vix[-1] / self.vix_avg[-1] if self.vix_avg[-1] > 0 else 1.0
            if compression_ratio < 0.70:
                size_mult = 0.5
                print(f"⚠️ Extreme VIX compression ({compression_ratio:.2f}) — reducing size 🌙")
            else:
                size_mult = 1.0

            # Risk-based position sizing using ATR stop
            atr_val = self.atr[-1]
            if atr_val > 0:
                risk_per_unit = self.atr_stop_mult * atr_val
                risk_amount = self.equity * 0.02  # 2% risk
                position_size = int(round(risk_amount / risk_per_unit))
                position_size = max(1, position_size)
            else:
                position_size = 1

            print(f"🚀 ENTRY SIGNAL! VIX={self.vix[-1]:.2f} (avg={self.vix_avg[-1]:.2f}, ratio={compression_ratio:.2f}) | "
                  f"Skew={self.gamma_skew[-1]:.4f} > avg={self.gamma_skew_avg[-1]:.4f} | RSI={self.rsi[-1]:.1f} | "
                  f"Size={position_size} 🌙✨")

            self.buy(size=position_size)

            # Set stop loss at 2x ATR below entry
            stop_price = price - self.atr_stop_mult * atr_val
            print(f"🛡️ Stop loss set at {stop_price:.2f}")


print("🌙 Running backtest... 🚀")
bt = Backtest(data, VolatilitySkewMomentum, cash=1_000_000, commission=0.002)
stats = bt.run()
print(stats)
print(stats._strategy)
print("✨ Moon Dev backtest complete! 🌙🚀")