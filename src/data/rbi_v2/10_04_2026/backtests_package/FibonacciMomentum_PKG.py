import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

print("🌙✨ Moon Dev Backtest AI initializing FibonacciMomentum strategy... 🚀")

# Load and clean data
data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

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
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

print(f"🌙 Data loaded: {len(data)} candles from {data.index[0]} to {data.index[-1]}")


class FibonacciMomentum(Strategy):
    # Strategy parameters
    swing_window = 20          # lookback for swing high/low detection
    rsi_period = 14
    ema_period = 20
    risk_pct = 0.02            # 2% risk per trade
    fib_low = 0.382
    fib_high = 0.618
    fib_stop = 0.786
    fib_tp2 = 1.618

    def init(self):
        print("🌙✨ Initializing indicators...")
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=self.rsi_period)
        self.ema = self.I(talib.EMA, self.data.Close, timeperiod=self.ema_period)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14)

        # Swing high/low using talib MAX/MIN
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=self.swing_window)
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=self.swing_window)

        # Momentum: MACD histogram for confirmation
        self.macd, self.macd_signal, self.macd_hist = self.I(
            talib.MACD, self.data.Close, fastperiod=12, slowperiod=26, signalperiod=9
        )
        print("🚀 Indicators ready!")

    def next(self):
        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]

        # Need enough history
        if len(self.data) < self.swing_window + 5:
            return

        # Current swing range
        sh = self.swing_high[-1]
        sl = self.swing_low[-1]
        rng = sh - sl
        if rng <= 0:
            return

        # Fibonacci levels for an uptrend impulse (sl -> sh)
        fib_382 = sh - rng * 0.382
        fib_618 = sh - rng * 0.618
        fib_786 = sh - rng * 0.786
        fib_1618 = sh + rng * 0.618  # extension

        # Fibonacci levels for a downtrend impulse (sh -> sl)
        dfib_382 = sl + rng * 0.382
        dfib_618 = sl + rng * 0.618
        dfib_786 = sl + rng * 0.786
        dfib_1618 = sl - rng * 0.618

        rsi = self.rsi[-1]
        rsi_prev = self.rsi[-2]
        macd_hist = self.macd_hist[-1]
        macd_hist_prev = self.macd_hist[-2]
        ema = self.ema[-1]
        atr = self.atr[-1]

        # ---- LONG SETUP ----
        # Uptrend: price above EMA, prior impulse up, retracement into golden pocket
        if not self.position:
            in_golden_long = fib_618 <= price <= fib_382
            trend_up = price > ema
            momentum_up = (rsi > 40 and rsi > rsi_prev) or (macd_hist > macd_hist_prev and macd_hist > -0.5 * atr)
            bullish_candle = self.data.Close[-1] > self.data.Open[-1] and self.data.Close[-1] > self.data.High[-2]

            if in_golden_long and trend_up and momentum_up and bullish_candle:
                stop = min(fib_786, sl) - 0.5 * atr
                risk = price - stop
                if risk <= 0:
                    return
                tp1 = sh
                tp2 = fib_1618
                rr = (tp1 - price) / risk
                if rr < 2.0:
                    print(f"🌙 Skipping long: R:R {rr:.2f} < 2.0")
                    return

                size = int(round((self.equity * self.risk_pct) / risk))
                if size < 1:
                    return
                print(f"🚀🌙 LONG entry @ {price:.2f} | SL {stop:.2f} | TP1 {tp1:.2f} | TP2 {tp2:.2f} | size {size}")
                self.buy(size=size, sl=stop, tp=tp1)

            # ---- SHORT SETUP ----
            in_golden_short = dfib_382 <= price <= dfib_618
            trend_down = price < ema
            momentum_down = (rsi < 60 and rsi < rsi_prev) or (macd_hist < macd_hist_prev and macd_hist < 0.5 * atr)
            bearish_candle = self.data.Close[-1] < self.data.Open[-1] and self.data.Close[-1] < self.data.Low[-2]

            if in_golden_short and trend_down and momentum_down and bearish_candle:
                stop = max(dfib_786, sh) + 0.5 * atr
                risk = stop - price
                if risk <= 0:
                    return
                tp1 = sl
                tp2 = dfib_1618
                rr = (price - tp1) / risk
                if rr < 2.0:
                    print(f"🌙 Skipping short: R:R {rr:.2f} < 2.0")
                    return

                size = int(round((self.equity * self.risk_pct) / risk))
                if size < 1:
                    return
                print(f"🚀🌙 SHORT entry @ {price:.2f} | SL {stop:.2f} | TP1 {tp1:.2f} | TP2 {tp2:.2f} | size {size}")
                self.sell(size=size, sl=stop, tp=tp1)


print("🌙 Running initial backtest...")
bt = Backtest(data, FibonacciMomentum, cash=1_000_000, commission=0.001)

stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙✨ Backtest complete! 🚀")