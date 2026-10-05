import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Load and clean data
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
print("🌙✨ Moon Dev Data Loaded! Shape:", data.shape)


class DivergentReversion(Strategy):
    bb_period = 20
    bb_std = 2.0
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    atr_period = 14
    adx_period = 14
    adx_threshold = 30
    risk_pct = 0.02
    atr_mult = 1.5
    time_stop = 12
    swing_lookback = 5

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low

        # Bollinger Bands
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS, close, timeperiod=self.bb_period,
            nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0
        )

        # MACD
        self.macd, self.macd_signal_line, self.macd_hist = self.I(
            talib.MACD, close, fastperiod=self.macd_fast,
            slowperiod=self.macd_slow, signalperiod=self.macd_signal
        )

        # ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        # ADX
        self.adx = self.I(talib.ADX, high, low, close, timeperiod=self.adx_period)

        # Swing highs/lows
        self.swing_high = self.I(talib.MAX, high, timeperiod=self.swing_lookback)
        self.swing_low = self.I(talib.MIN, low, timeperiod=self.swing_lookback)

        print("🌙✨ DivergentReversion indicators initialized! 🚀")

    def next(self):
        price = self.data.Close[-1]
        upper = self.bb_upper[-1]
        lower = self.bb_lower[-1]
        middle = self.bb_middle[-1]
        macd = self.macd[-1]
        macd_hist = self.macd_hist[-1]
        atr = self.atr[-1]
        adx = self.adx[-1]

        if np.isnan(upper) or np.isnan(lower) or np.isnan(atr) or np.isnan(adx):
            return

        # Manage open position
        if self.position:
            bars_held = len(self.data) - self.trades[-1].entry_bar if self.trades else 0
            if self.position.is_long:
                # Exit on revert inside channel (close above lower band)
                if price > lower:
                    print(f"🌙✅ LONG exit mean-reversion at {price:.2f} (inside channel)")
                    self.position.close()
                elif bars_held >= self.time_stop:
                    print(f"⏰ LONG time-stop exit at {price:.2f}")
                    self.position.close()
            elif self.position.is_short:
                if price < upper:
                    print(f"🌙✅ SHORT exit mean-reversion at {price:.2f} (inside channel)")
                    self.position.close()
                elif bars_held >= self.time_stop:
                    print(f"⏰ SHORT time-stop exit at {price:.2f}")
                    self.position.close()
            return

        # ADX filter — avoid strong trends
        if adx > self.adx_threshold:
            return

        # Detect bullish divergence: recent lower low in price, higher low in MACD
        if len(self.data) < self.swing_lookback * 3:
            return

        lookback = self.swing_lookback * 2
        price_slice = np.array(self.data.Low[-lookback:])
        macd_slice = np.array(self.macd[-lookback:])

        if len(price_slice) < lookback or len(macd_slice) < lookback:
            return

        # Split into two halves for swing comparison
        half = lookback // 2
        p1_low = np.min(price_slice[:half])
        p2_low = np.min(price_slice[half:])
        m1_low = np.nanmin(macd_slice[:half])
        m2_low = np.nanmin(macd_slice[half:])

        bullish_div = (p2_low < p1_low) and (m2_low > m1_low)

        p1_high = np.max(price_slice[:half])
        p2_high = np.max(price_slice[half:])
        m1_high = np.nanmax(macd_slice[:half])
        m2_high = np.nanmax(macd_slice[half:])

        bearish_div = (p2_high > p1_high) and (m2_high < m1_high)

        # Bullish reversal candle check
        o = self.data.Open[-1]
        c = self.data.Close[-1]
        h = self.data.High[-1]
        l = self.data.Low[-1]
        body = abs(c - o)
        rng = h - l if h > l else 1e-9
        bullish_candle = (c > o) and ((c - l) / rng > 0.5)
        bearish_candle = (c < o) and ((h - c) / rng > 0.5)

        # Long entry
        if price < lower and bullish_div and bullish_candle:
            stop_price = min(self.swing_low[-1], price - self.atr_mult * atr)
            risk = price - stop_price
            if risk <= 0:
                return
            size = max(1, int(round(1000000 / price)))
            print(f"🚀🌙 LONG DivergentReversion! Price={price:.2f} < Lower={lower:.2f}, MACD bull div, ADX={adx:.1f}, Stop={stop_price:.2f}")
            self.buy(size=size, sl=stop_price)

        # Short entry
        elif price > upper and bearish_div and bearish_candle:
            stop_price = max(self.swing_high[-1], price + self.atr_mult * atr)
            risk = stop_price - price
            if risk <= 0:
                return
            size = max(1, int(round(1000000 / price)))
            print(f"🚀🌙 SHORT DivergentReversion! Price={price:.2f} > Upper={upper:.2f}, MACD bear div, ADX={adx:.1f}, Stop={stop_price:.2f}")
            self.sell(size=size, sl=stop_price)


bt = Backtest(data, DivergentReversion, cash=1000000, commission=0.001, exclusive_orders=True)
stats = bt.run()
print(stats)
print(stats._strategy)