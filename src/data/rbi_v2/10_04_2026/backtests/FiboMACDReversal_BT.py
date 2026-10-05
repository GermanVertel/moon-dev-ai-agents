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

print("🌙✨ Moon Dev FiboMACD Reversal Backtest Loading... 🚀")
print(f"📊 Data shape: {data.shape}")
print(f"📅 Date range: {data.index[0]} to {data.index[-1]}")


class FiboMACDReversal(Strategy):
    swing_lookback = 60
    fib_tolerance = 0.015  # 1.5% proximity to fib level
    risk_pct = 0.02
    rr_ratio = 2.0
    atr_period = 14
    ema_trend_period = 50

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low

        # MACD
        self.macd, self.macd_signal, self.macd_hist = self.I(
            talib.MACD, close, fastperiod=12, slowperiod=26, signalperiod=9
        )

        # ATR for stop sizing
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        # Trend EMA
        self.ema_trend = self.I(talib.EMA, close, timeperiod=self.ema_trend_period)

        # Swing highs/lows
        self.swing_high = self.I(talib.MAX, high, timeperiod=self.swing_lookback)
        self.swing_low = self.I(talib.MIN, low, timeperiod=self.swing_lookback)

        print("🌙 Indicators initialized: MACD(12,26,9), ATR(14), EMA(50), Swing(60) ✨")

    def next(self):
        if len(self.data) < self.swing_lookback + 5:
            return

        price = self.data.Close[-1]
        sh = self.swing_high[-1]
        sl = self.swing_low[-1]

        if sh <= sl or np.isnan(sh) or np.isnan(sl):
            return

        rng = sh - sl

        # Fibonacci levels
        # For uptrend (bullish retracement): from swing low -> swing high
        fib_up_382 = sh - 0.382 * rng
        fib_up_500 = sh - 0.500 * rng
        fib_up_618 = sh - 0.618 * rng
        fib_up_786 = sh - 0.786 * rng

        # For downtrend (bearish retracement): from swing high -> swing low
        fib_dn_382 = sl + 0.382 * rng
        fib_dn_500 = sl + 0.500 * rng
        fib_dn_618 = sl + 0.618 * rng
        fib_dn_786 = sl + 0.786 * rng

        golden_zone_long = (fib_up_382, fib_up_618)
        golden_zone_short = (fib_dn_382, fib_dn_618)

        macd_now = self.macd[-1]
        macd_prev = self.macd[-2]
        sig_now = self.macd_signal[-1]
        sig_prev = self.macd_signal[-2]
        hist_now = self.macd_hist[-1]
        hist_prev = self.macd_hist[-2]

        bullish_cross = macd_prev < sig_prev and macd_now > sig_now
        bearish_cross = macd_prev > sig_prev and macd_now < sig_now

        in_long_zone = golden_zone_long[0] >= price >= golden_zone_long[1]
        in_short_zone = golden_zone_short[0] <= price <= golden_zone_short[1]

        # Trend context
        uptrend = price > self.ema_trend[-1]
        downtrend = price < self.ema_trend[-1]

        if not self.position:
            # LONG: bullish MACD cross + price in golden fib zone of prior uptrend
            if bullish_cross and in_long_zone and hist_prev < 0 and hist_now > 0:
                stop = min(fib_up_786, sl) * 0.998
                risk = price - stop
                if risk > 0:
                    size = int(round((self.equity * self.risk_pct) / risk))
                    if size > 0:
                        tp = price + self.rr_ratio * risk
                        self.buy(size=size, sl=stop, tp=tp)
                        print(f"🚀🌙 MOON DEV LONG! Price={price:.2f} Fib Zone [{fib_up_382:.2f}-{fib_up_618:.2f}] "
                              f"SL={stop:.2f} TP={tp:.2f} Size={size}")

            # SHORT: bearish MACD cross + price in golden fib zone of prior downtrend
            elif bearish_cross and in_short_zone and hist_prev > 0 and hist_now < 0:
                stop = max(fib_dn_786, sh) * 1.002
                risk = stop - price
                if risk > 0:
                    size = int(round((self.equity * self.risk_pct) / risk))
                    if size > 0:
                        tp = price - self.rr_ratio * risk
                        self.sell(size=size, sl=stop, tp=tp)
                        print(f"📉🌙 MOON DEV SHORT! Price={price:.2f} Fib Zone [{fib_dn_618:.2f}-{fib_dn_382:.2f}] "
                              f"SL={stop:.2f} TP={tp:.2f} Size={size}")


# Run backtest
bt = Backtest(
    data,
    FiboMACDReversal,
    cash=1_000_000,
    commission=0.001,
    exclusive=False,
    trade_on_close=False
)

stats = bt.run()
print(stats)
print(stats._strategy)