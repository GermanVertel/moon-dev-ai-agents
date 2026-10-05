import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# Load and clean data
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

print("🌙 Moon Dev Backtest Engine Initialized ✨")
print(f"📊 Data loaded: {len(data)} bars from {data.index[0]} to {data.index[-1]} 🚀")


class FibonacciDivergence(Strategy):
    ema_period = 200
    rsi_period = 14
    atr_period = 14
    swing_lookback = 50
    risk_pct = 0.02

    def init(self):
        print("🌙 Initializing FibonacciDivergence strategy indicators... ✨")
        self.ema200 = self.I(talib.EMA, self.data.Close, timeperiod=self.ema_period)
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=self.rsi_period)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period)

        macd, macdsignal, macdhist = talib.MACD(self.data.Close, fastperiod=12, slowperiod=26, signalperiod=9)
        self.macd = self.I(lambda: macd)
        self.macd_signal = self.I(lambda: macdsignal)
        self.macd_hist = self.I(lambda: macdhist)

        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=self.swing_lookback)
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=self.swing_lookback)

        self.entry_price = None
        self.stop_price = None
        self.tp1 = None
        self.tp2 = None
        self.tp3 = None
        self.tp1_hit = False
        self.tp2_hit = False
        print("🚀 Indicators ready! Let's find some Fibonacci setups 🌙")

    def next(self):
        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]

        if len(self.data) < self.swing_lookback + 5:
            return

        # Manage open position
        if self.position:
            # Invalidation: close below 78.6% level or swing low
            if self.stop_price and low <= self.stop_price:
                print(f"🛑 MOON DEV EXIT: Stop hit at {self.stop_price:.2f} (price={price:.2f}) 🌙")
                self.position.close()
                self._reset_trade()
                return

            # Scale out at TP1
            if not self.tp1_hit and self.tp1 and high >= self.tp1:
                print(f"✨ MOON DEV TP1 HIT at {self.tp1:.2f} — scaling out 33% 🚀")
                self.position.close(portion=0.33)
                self.tp1_hit = True
                # Move stop to breakeven
                self.stop_price = max(self.stop_price, self.entry_price)
                print(f"🌙 Stop moved to breakeven: {self.stop_price:.2f}")

            # TP2
            if self.tp1_hit and not self.tp2_hit and self.tp2 and high >= self.tp2:
                print(f"✨ MOON DEV TP2 HIT at {self.tp2:.2f} — scaling out another 33% 🚀")
                self.position.close(portion=0.5)
                self.tp2_hit = True

            # TP3
            if self.tp2_hit and self.tp3 and high >= self.tp3:
                print(f"🌟 MOON DEV TP3 HIT at {self.tp3:.2f} — closing remainder 🚀")
                self.position.close()
                self._reset_trade()
                return

            # Trail stop below new higher lows
            if self.tp1_hit:
                new_stop = self.swing_low[-1] - 0.5 * self.atr[-1]
                if new_stop > self.stop_price:
                    self.stop_price = new_stop
                    print(f"🌙 Trailing stop raised to {self.stop_price:.2f} ✨")

            return

        # Entry logic
        if len(self.data) < self.swing_lookback + 5:
            return

        # 1. Trend filter: price above 200 EMA
        if price <= self.ema200[-1]:
            return

        # 2. Identify prior impulse leg (swing low -> swing high)
        recent_high = self.swing_high[-1]
        recent_low = self.swing_low[-1]
        if recent_high <= recent_low:
            return

        leg_range = recent_high - recent_low
        if leg_range <= 0:
            return

        # Fibonacci retracement zones
        fib_382 = recent_high - 0.382 * leg_range
        fib_618 = recent_high - 0.618 * leg_range
        fib_786 = recent_high - 0.786 * leg_range

        # 3. Pullback into 38.2% - 61.8% zone
        in_fib_zone = fib_618 <= price <= fib_382
        if not in_fib_zone:
            return

        # 4. MACD divergence weakening: histogram shrinking (less negative than prior bars)
        hist_ok = False
        if len(self.macd_hist) >= 3:
            h1, h2, h3 = self.macd_hist[-3], self.macd_hist[-2], self.macd_hist[-1]
            # Histogram improving (rising toward zero) and MACD line flattening/turning up
            if h3 > h2 and h2 >= h1:
                hist_ok = True
            if self.macd[-1] > self.macd[-2] and self.macd[-2] <= self.macd[-3]:
                hist_ok = True

        if not hist_ok:
            return

        # 5. RSI condition: below 40 and curling up
        rsi_ok = self.rsi[-1] < 45 and self.rsi[-1] > self.rsi[-2] and self.rsi[-2] <= self.rsi[-3] + 2
        if not rsi_ok:
            return

        # 6. Entry trigger: bullish candle close
        if self.data.Close[-1] <= self.data.Open[-1]:
            return

        # Calculate stop: below swing low or 78.6% fib, whichever tighter with ATR buffer
        atr_val = self.atr[-1]
        stop_below_swing = recent_low - 1.0 * atr_val
        stop_below_786 = fib_786 - 0.5 * atr_val
        stop = max(stop_below_swing, stop_below_786)

        risk = price - stop
        if risk <= 0:
            return

        # Targets
        tp1 = recent_high - 0.382 * leg_range
        tp2 = recent_high
        tp3 = recent_high + 0.272 * leg_range  # 127.2% extension

        # R:R check to TP2
        reward = tp2 - price
        if reward / risk < 2.0:
            return

        # Position sizing: risk 2% of equity
        equity = self.equity
        risk_amount = equity * self.risk_pct
        size = int(round(risk_amount / risk))
        if size <= 0:
            return

        print(f"🌙✨ MOON DEV LONG SIGNAL ✨🌙")
        print(f"   Price: {price:.2f} | EMA200: {self.ema200[-1]:.2f}")
        print(f"   Fib zone: {fib_618:.2f} - {fib_382:.2f} | RSI: {self.rsi[-1]:.2f}")
        print(f"   MACD hist: {self.macd_hist[-1]:.4f} | Stop: {stop:.2f}")
        print(f"   TP1: {tp1:.2f} | TP2: {tp2:.2f} | TP3: {tp3:.2f}")
        print(f"   Size: {size} | R:R to TP2 = {reward/risk:.2f} 🚀")

        self.buy(size=size)
        self.entry_price = price
        self.stop_price = stop
        self.tp1 = tp1
        self.tp2 = tp2
        self.tp3 = tp3
        self.tp1_hit = False
        self.tp2_hit = False

    def _reset_trade(self):
        self.entry_price = None
        self.stop_price = None
        self.tp1 = None
        self.tp2 = None
        self.tp3 = None
        self.tp1_hit = False
        self.tp2_hit = False


print("🌙 Starting Moon Dev Backtest... 🚀")
bt = Backtest(data, FibonacciDivergence, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)
print("✨ Moon Dev Backtest Complete! 🌙")