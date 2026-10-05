import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Load data
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')

print("🌙✨ Moon Dev Backtest Starting for Fibonacci Momentum ✨🌙")
print(f"📊 Data loaded: {len(data)} candles from {data.index[0]} to {data.index[-1]}")


class FibonacciMomentum(Strategy):
    ema_fast = 50
    ema_slow = 200
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    swing_lookback = 30
    risk_pct = 0.02
    rr_ratio = 2.0

    def init(self):
        print("🚀 Initializing Moon Dev indicators...")
        self.ema50 = self.I(talib.EMA, self.data.Close, timeperiod=self.ema_fast)
        self.ema200 = self.I(talib.EMA, self.data.Close, timeperiod=self.ema_slow)
        self.macd, self.macd_signal_line, self.macd_hist = self.I(
            talib.MACD, self.data.Close,
            fastperiod=self.macd_fast,
            slowperiod=self.macd_slow,
            signalperiod=self.macd_signal
        )
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14)
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=self.swing_lookback)
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=self.swing_lookback)
        print("✅ Indicators ready: EMA50, EMA200, MACD, ATR, Swing levels")

    def next(self):
        if len(self.data) < self.ema_slow + 5:
            return

        price = self.data.Close[-1]
        e50 = self.ema50[-1]
        e200 = self.ema200[-1]
        macd_now = self.macd[-1]
        macd_prev = self.macd[-2]
        sig_now = self.macd_signal_line[-1]
        sig_prev = self.macd_signal_line[-2]
        hist_now = self.macd_hist[-1]
        hist_prev = self.macd_hist[-2]
        sh = self.swing_high[-1]
        sl = self.swing_low[-1]
        atr = self.atr[-1]

        if np.isnan(e200) or np.isnan(sh) or np.isnan(sl) or atr <= 0:
            return

        # Fibonacci retracement zones from swing low -> swing high
        rng = sh - sl
        if rng <= 0:
            return
        fib_382 = sh - rng * 0.382
        fib_500 = sh - rng * 0.500
        fib_618 = sh - rng * 0.618

        # Bullish trend
        uptrend = price > e200 and e50 > e200
        # Bearish trend
        downtrend = price < e200 and e50 < e200

        # MACD bullish momentum (manual crossover detection - no backtesting.lib)
        bull_macd_cross = macd_prev <= sig_prev and macd_now > sig_now
        bull_hist_flip = hist_prev <= 0 and hist_now > 0
        bear_macd_cross = macd_prev >= sig_prev and macd_now < sig_now
        bear_hist_flip = hist_prev >= 0 and hist_now < 0

        # In Fibonacci pullback zone (between 38.2% and 61.8%)
        in_long_zone = fib_618 <= price <= fib_382
        in_short_zone = fib_382 <= price <= fib_618

        # Manage existing positions
        if self.position:
            entry = self.position.entry_price
            if self.position.is_long:
                # Stop: below 61.8% or below swing low, whichever tighter
                stop = max(fib_618, sl) - atr * 0.25
                # TP1 = swing high, TP2 = 127.2% extension
                tp1 = sh
                tp2 = sl + rng * 1.272
                # Momentum exit
                if bear_macd_cross or bear_hist_flip:
                    print(f"🌙 Bearish MACD reversal — closing long @ {price:.2f}")
                    self.position.close()
                    return
                if price <= stop:
                    print(f"🛑 Long stop hit @ {price:.2f}")
                    self.position.close()
                    return
                if price >= tp2:
                    print(f"🎯 TP2 hit on long @ {price:.2f}")
                    self.position.close()
                    return
                if price >= tp1:
                    # Move stop to breakeven after TP1
                    pass
            else:
                stop = min(fib_382, sh) + atr * 0.25
                tp1 = sl
                tp2 = sh - rng * 1.272
                if bull_macd_cross or bull_hist_flip:
                    print(f"🌙 Bullish MACD reversal — closing short @ {price:.2f}")
                    self.position.close()
                    return
                if price >= stop:
                    print(f"🛑 Short stop hit @ {price:.2f}")
                    self.position.close()
                    return
                if price <= tp2:
                    print(f"🎯 TP2 hit on short @ {price:.2f}")
                    self.position.close()
                    return
            return

        # Long entry
        if uptrend and in_long_zone and (bull_macd_cross or bull_hist_flip):
            stop = max(fib_618, sl) - atr * 0.25
            risk = price - stop
            if risk <= 0:
                return
            tp = price + risk * self.rr_ratio
            # Position sizing: 1M equity fraction by risk
            size = int(round(1_000_000 * self.risk_pct / risk))
            if size < 1:
                return
            print(f"🚀🌙 LONG entry @ {price:.2f} | Fib zone | stop={stop:.2f} tp={tp:.2f} size={size}")
            self.buy(size=size, sl=stop, tp=tp)

        # Short entry
        elif downtrend and in_short_zone and (bear_macd_cross or bear_hist_flip):
            stop = min(fib_382, sh) + atr * 0.25
            risk = stop - price
            if risk <= 0:
                return
            tp = price - risk * self.rr_ratio
            size = int(round(1_000_000 * self.risk_pct / risk))
            if size < 1:
                return
            print(f"🔻🌙 SHORT entry @ {price:.2f} | Fib zone | stop={stop:.2f} tp={tp:.2f} size={size}")
            self.sell(size=size, sl=stop, tp=tp)


bt = Backtest(data, FibonacciMomentum, cash=1_000_000, commission=0.0002)
stats = bt.run()
print(stats)
print(stats._strategy)