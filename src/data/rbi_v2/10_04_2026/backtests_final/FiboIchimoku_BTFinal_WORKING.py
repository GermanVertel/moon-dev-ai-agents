import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Load data
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Ensure proper column mapping
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

# Set datetime index
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')

print("🌙 Moon Dev Backtest Initialized - FiboIchimoku Strategy ✨")
print(f"📊 Data loaded: {len(data)} candles")
print(f"🚀 Starting backtest...\n")


class FiboIchimoku(Strategy):
    # Ichimoku parameters
    tenkan_period = 9
    kijun_period = 26
    senkou_b_period = 52
    chikou_shift = 26

    # Fibonacci swing detection
    swing_lookback = 50

    # Risk management
    risk_pct = 0.02
    size = 0.95  # 🌙 Use fraction of equity for position sizing

    def init(self):
        # Ichimoku indicators - use (High + Low) / 2 for Tenkan/Kijun
        hl = (self.data.High + self.data.Low) / 2
        self.tenkan = self.I(talib.SMA, hl, timeperiod=self.tenkan_period)
        self.kijun = self.I(talib.SMA, hl, timeperiod=self.kijun_period)

        # Senkou Span A = (Tenkan + Kijun)/2 (no forward shift in backtesting.py)
        self.senkou_a = self.I(lambda t, k: (t + k) / 2, self.tenkan, self.kijun)
        # Senkou Span B = SMA of HL/2 over 52
        self.senkou_b = self.I(talib.SMA, hl, timeperiod=self.senkou_b_period)

        # ATR for volatility
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14)

        # Swing highs/lows
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=self.swing_lookback)
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=self.swing_lookback)

        print("🌙✨ Ichimoku indicators initialized successfully!")

    def next(self):
        # Skip if not enough data
        if len(self.data) < self.senkou_b_period + self.chikou_shift + 5:
            return

        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]

        # Current Ichimoku values
        tenkan = self.tenkan[-1]
        kijun = self.kijun[-1]
        senkou_a = self.senkou_a[-1]
        senkou_b = self.senkou_b[-1]
        atr = self.atr[-1]

        # Skip if any indicator is NaN
        if (np.isnan(tenkan) or np.isnan(kijun) or np.isnan(senkou_a) or
                np.isnan(senkou_b) or np.isnan(atr)):
            return

        # Cloud top/bottom
        cloud_top = max(senkou_a, senkou_b)
        cloud_bottom = min(senkou_a, senkou_b)

        # Chikou span check: current close vs close 26 periods ago
        chikou_close = self.data.Close[-self.chikou_shift - 1] if len(self.data) > self.chikou_shift else None

        # Swing high/low
        sw_high = self.swing_high[-1]
        sw_low = self.swing_low[-1]
        fib_range = sw_high - sw_low

        if fib_range <= 0 or np.isnan(fib_range):
            return

        # Fib levels
        fib_50 = sw_high - 0.5 * fib_range
        fib_618 = sw_high - 0.618 * fib_range
        fib_786 = sw_high - 0.786 * fib_range
        fib_382 = sw_high - 0.382 * fib_range

        # Trend confirmation
        bullish_trend = (price > cloud_top and tenkan > kijun and
                         (chikou_close is None or price > chikou_close))
        bearish_trend = (price < cloud_bottom and tenkan < kijun and
                         (chikou_close is None or price < chikou_close))

        # In Fibonacci golden pocket zone (50%-61.8%)
        in_long_zone = fib_618 <= price <= fib_50
        in_short_zone = fib_50 <= price <= fib_618

        # Candlestick confirmation
        body = abs(self.data.Close[-1] - self.data.Open[-1])
        range_candle = high - low
        lower_wick = min(self.data.Open[-1], self.data.Close[-1]) - low
        upper_wick = high - max(self.data.Open[-1], self.data.Close[-1])

        bullish_candle = (self.data.Close[-1] > self.data.Open[-1] and
                          body > 0 and
                          (lower_wick > body * 0.5 or
                           (self.data.Close[-1] > self.data.Open[-1] and
                            self.data.Close[-2] < self.data.Open[-2] and
                            self.data.Close[-1] > self.data.Open[-2])))

        bearish_candle = (self.data.Close[-1] < self.data.Open[-1] and
                          body > 0 and
                          (upper_wick > body * 0.5 or
                           (self.data.Close[-1] < self.data.Open[-1] and
                            self.data.Close[-2] > self.data.Open[-2] and
                            self.data.Close[-1] < self.data.Open[-2])))

        # Entry logic
        if not self.position:
            # Long entry
            if bullish_trend and in_long_zone and bullish_candle and price > kijun:
                sl = min(fib_786, kijun) - atr * 0.5
                risk = price - sl
                if risk > 0:
                    tp = sw_high
                    rr = (tp - price) / risk
                    if rr >= 2.0:
                        print(f"🌙🚀 LONG ENTRY! Price: {price:.2f} | SL: {sl:.2f} | TP: {tp:.2f} | RR: {rr:.2f}")
                        self.buy(size=self.size, sl=sl, tp=tp)

            # Short entry
            elif bearish_trend and in_short_zone and bearish_candle and price < kijun:
                sl = max(fib_786, kijun) + atr * 0.5
                risk = sl - price
                if risk > 0:
                    tp = sw_low
                    rr = (price - tp) / risk
                    if rr >= 2.0:
                        print(f"🌙🔻 SHORT ENTRY! Price: {price:.2f} | SL: {sl:.2f} | TP: {tp:.2f} | RR: {rr:.2f}")
                        self.sell(size=self.size, sl=sl, tp=tp)

        # Exit logic - trend invalidation
        else:
            if self.position.is_long:
                # Exit if price closes below cloud or bearish tenkan/kijun cross
                if price < cloud_bottom or tenkan < kijun:
                    print(f"🌙⚠️ LONG EXIT - Trend invalidation at {price:.2f}")
                    self.position.close()
            elif self.position.is_short:
                if price > cloud_top or tenkan > kijun:
                    print(f"🌙⚠️ SHORT EXIT - Trend invalidation at {price:.2f}")
                    self.position.close()


# Run backtest
bt = Backtest(data, FiboIchimoku, cash=1_000_000, commission=0.002)
stats = bt.run()
print(stats)
print(stats._strategy)