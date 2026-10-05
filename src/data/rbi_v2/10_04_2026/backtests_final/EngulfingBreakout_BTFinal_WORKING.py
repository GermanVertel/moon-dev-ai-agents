import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Load and clean data
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
print("🌙 Moon Dev data loaded and cleaned! Shape:", data.shape)
print("✨ Columns:", list(data.columns))


class EngulfingBreakout(Strategy):
    """
    🌙 EngulfingBreakout Strategy by Moon Dev
    Trades bearish engulfing patterns confirmed by a break below the pattern low.
    """

    ema_period = 50
    atr_period = 14
    risk_pct = 0.02
    rr_ratio = 2.0
    volume_mult = 1.2
    min_body_atr = 0.5

    def init(self):
        print("🚀 Initializing Moon Dev EngulfingBreakout indicators...")
        self.ema = self.I(talib.EMA, self.data.Close, timeperiod=self.ema_period)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close,
                          timeperiod=self.atr_period)
        self.vol_sma = self.I(talib.SMA, self.data.Volume, timeperiod=20)
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=14)
        # Detect bearish engulfing on each bar
        self.engulfing = self.I(self._detect_engulfing, self.data.Open, self.data.High,
                                self.data.Low, self.data.Close)
        # Track our own stop loss for short position (Position has no .sl attribute)
        self._short_sl = None
        print("✅ Indicators ready! 🌙")

    def _detect_engulfing(self, o, h, l, c):
        n = len(o)
        result = np.zeros(n)
        for i in range(1, n):
            prev_o, prev_c = o[i-1], c[i-1]
            cur_o, cur_c = o[i], c[i]
            # Candle 1 bullish
            bull1 = prev_c > prev_o
            # Candle 2 bearish
            bear2 = cur_c < cur_o
            # Body engulfs: open >= prior close, close <= prior open
            engulfs = (cur_o >= prev_c) and (cur_c <= prev_o)
            if bull1 and bear2 and engulfs:
                result[i] = 1.0
        return result

    def next(self):
        price = self.data.Close[-1]
        i = len(self.data) - 1

        # Manage open trade
        if self.position:
            # Trailing stop using ATR for short positions
            if self.position.is_short:
                new_sl = self.data.Close[-1] + 1.5 * self.atr[-1]
                if self._short_sl is None or new_sl < self._short_sl:
                    self._short_sl = new_sl
                    # Manually exit if price hits our trailing stop
                if price >= self._short_sl:
                    print(f"🌙 Trailing SL hit at {price:.2f} (SL={self._short_sl:.2f}) — closing short")
                    self.position.close()
                    self._short_sl = None
            return

        if i < 2:
            return

        # Check if a bearish engulfing pattern occurred on a recent bar
        # Look back up to 10 bars for an engulfing pattern and check break of its low
        lookback = min(10, i)
        for k in range(1, lookback + 1):
            idx = i - k
            if self.engulfing[idx] == 1.0:
                eng_low = self.data.Low[idx]
                eng_high = self.data.High[idx]
                eng_body = abs(self.data.Close[idx] - self.data.Open[idx])

                # Filters
                if price >= self.ema[-1]:
                    print(f"🌙 Skip: price above EMA (trend not bearish)")
                    break
                if self.rsi[-1] < 30:
                    print(f"🌙 Skip: RSI oversold {self.rsi[-1]:.1f}")
                    break
                if eng_body < self.min_body_atr * self.atr[idx]:
                    print(f"🌙 Skip: engulfing body too small")
                    break
                if self.data.Volume[idx] < self.volume_mult * self.vol_sma[idx]:
                    print(f"🌙 Skip: low volume on engulfing")
                    break

                # Breakout trigger: close below engulfing low
                if price < eng_low:
                    entry = price
                    sl = eng_high
                    risk = sl - entry
                    if risk <= 0:
                        break
                    tp = entry - self.rr_ratio * risk

                    # Position sizing: risk 2% of equity
                    equity = self.equity
                    risk_amount = equity * self.risk_pct
                    size = risk_amount / risk
                    size = int(round(size))
                    if size < 1:
                        print("🌙 Size too small, skipping")
                        break

                    print(f"🚀🌙 SHORT SIGNAL! Entry={entry:.2f} SL={sl:.2f} TP={tp:.2f} Size={size}")
                    self._short_sl = sl
                    self.sell(size=size, sl=sl, tp=tp)
                break


# Run backtest
bt = Backtest(data, EngulfingBreakout, cash=1_000_000, commission=0.001)
print("🌙✨ Running Moon Dev EngulfingBreakout backtest...")
stats = bt.run()
print(stats)
print(stats._strategy)