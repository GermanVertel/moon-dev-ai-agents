import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

print("🌙 Moon Dev's AdaptiveCascade Backtest Initializing... ✨🚀")

# Load data
data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

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
if 'datetime' in data.columns:
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')

print(f"🌙 Data loaded: {len(data)} bars from {data.index[0]} to {data.index[-1]} ✨")


class AdaptiveCascade(Strategy):
    # Strategy parameters
    fast_ema_period = 20
    medium_ema_period = 50
    slow_ema_period = 100
    regime_ema_period = 200
    rsi_period = 14
    atr_period = 14
    atr_ma_period = 20
    swing_lookback = 20
    risk_pct = 0.01  # 1% risk per trade
    atr_stop_mult = 2.0
    tp_r_multiple = 2.0
    time_stop_bars = 40

    def init(self):
        print("🌙 Initializing AdaptiveCascade indicators... ✨")
        close = self.data.Close
        high = self.data.High
        low = self.data.Low

        # EMAs
        self.fast_ema = self.I(talib.EMA, close, timeperiod=self.fast_ema_period, name="FastEMA")
        self.medium_ema = self.I(talib.EMA, close, timeperiod=self.medium_ema_period, name="MedEMA")
        self.slow_ema = self.I(talib.EMA, close, timeperiod=self.slow_ema_period, name="SlowEMA")
        self.regime_ema = self.I(talib.EMA, close, timeperiod=self.regime_ema_period, name="RegimeEMA")

        # RSI
        self.rsi = self.I(talib.RSI, close, timeperiod=self.rsi_period, name="RSI")

        # MACD histogram
        macd, macdsignal, macdhist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        self.macd_hist = self.I(lambda: macdhist, name="MACDHist")

        # ATR and ATR MA
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period, name="ATR")
        self.atr_ma = self.I(talib.SMA, self.atr, timeperiod=self.atr_ma_period, name="ATR_MA")

        # Swing highs/lows
        self.swing_high = self.I(talib.MAX, high, timeperiod=self.swing_lookback, name="SwingHigh")
        self.swing_low = self.I(talib.MIN, low, timeperiod=self.swing_lookback, name="SwingLow")

        # State
        self.entry_price = None
        self.stop_price = None
        self.tp_price = None
        self.bars_in_trade = 0
        self.scaled_in = False

        print("🌙 Indicators ready! 🚀")

    def next(self):
        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]

        # Skip if indicators not ready
        if (np.isnan(self.slow_ema[-1]) or np.isnan(self.regime_ema[-1]) or
                np.isnan(self.rsi[-1]) or np.isnan(self.atr[-1]) or
                np.isnan(self.atr_ma[-1]) or np.isnan(self.swing_high[-1]) or
                np.isnan(self.swing_low[-1])):
            return

        # Manage open position
        if self.position:
            self.bars_in_trade += 1

            if self.position.is_long:
                # Move stop to breakeven after 1R
                if self.entry_price and self.stop_price:
                    risk = self.entry_price - self.stop_price
                    if price >= self.entry_price + risk and self.stop_price < self.entry_price:
                        self.stop_price = self.entry_price
                        print(f"🌙 Long stop moved to breakeven @ {self.stop_price:.2f} ✨")

                # Trailing stop by 1.5x ATR
                new_stop = price - 1.5 * self.atr[-1]
                if new_stop > self.stop_price:
                    self.stop_price = new_stop

                # Exit conditions
                if price <= self.stop_price:
                    print(f"🌙 Long STOP hit @ {price:.2f} 💥")
                    self.position.close()
                    self._reset()
                    return

                if price >= self.tp_price:
                    print(f"🌙 Long TP hit @ {price:.2f} 🎯")
                    self.position.close()
                    self._reset()
                    return

                # Trend invalidation
                if len(self.data) > 2 and self.fast_ema[-1] < self.medium_ema[-1] and self.fast_ema[-2] >= self.medium_ema[-2]:
                    print(f"🌙 Long trend invalidation @ {price:.2f} ⚠️")
                    self.position.close()
                    self._reset()
                    return

                # Time stop
                if self.bars_in_trade >= self.time_stop_bars:
                    risk = self.entry_price - self.stop_price if self.stop_price else 0
                    if risk > 0 and (price - self.entry_price) < risk:
                        print(f"🌙 Long time stop @ {price:.2f} ⏰")
                        self.position.close()
                        self._reset()
                        return

            elif self.position.is_short:
                if self.entry_price and self.stop_price:
                    risk = self.stop_price - self.entry_price
                    if price <= self.entry_price - risk and self.stop_price > self.entry_price:
                        self.stop_price = self.entry_price
                        print(f"🌙 Short stop moved to breakeven @ {self.stop_price:.2f} ✨")

                new_stop = price + 1.5 * self.atr[-1]
                if new_stop < self.stop_price:
                    self.stop_price = new_stop

                if price >= self.stop_price:
                    print(f"🌙 Short STOP hit @ {price:.2f} 💥")
                    self.position.close()
                    self._reset()
                    return

                if price <= self.tp_price:
                    print(f"🌙 Short TP hit @ {price:.2f} 🎯")
                    self.position.close()
                    self._reset()
                    return

                if len(self.data) > 2 and self.fast_ema[-1] > self.medium_ema[-1] and self.fast_ema[-2] <= self.medium_ema[-2]:
                    print(f"🌙 Short trend invalidation @ {price:.2f} ⚠️")
                    self.position.close()
                    self._reset()
                    return

                if self.bars_in_trade >= self.time_stop_bars:
                    risk = self.stop_price - self.entry_price if self.stop_price else 0
                    if risk > 0 and (self.entry_price - price) < risk:
                        print(f"🌙 Short time stop @ {price:.2f} ⏰")
                        self.position.close()
                        self._reset()
                        return

            return

        # Entry logic
        fast = self.fast_ema[-1]
        med = self.medium_ema[-1]
        slow = self.slow_ema[-1]
        regime = self.regime_ema[-1]
        rsi = self.rsi[-1]
        rsi_prev = self.rsi[-2]
        macd_hist = self.macd_hist[-1]
        macd_hist_prev = self.macd_hist[-2]
        atr = self.atr[-1]
        atr_ma = self.atr_ma[-1]
        swing_high = self.swing_high[-2]  # prior swing high to avoid lookahead
        swing_low = self.swing_low[-2]

        # Volatility filter
        vol_expansion = atr > atr_ma

        # Long conditions
        long_trend = fast > med > slow and price > regime
        long_momentum = (rsi > 50 and rsi > rsi_prev) or (macd_hist > 0 and macd_hist > macd_hist_prev)
        long_breakout = price > swing_high

        if long_trend and long_momentum and vol_expansion and long_breakout:
            stop = min(swing_low, price - self.atr_stop_mult * atr)
            risk = price - stop
            if risk > 0:
                size = int(round(1000000 / price))
                if size > 0:
                    self.entry_price = price
                    self.stop_price = stop
                    self.tp_price = price + self.tp_r_multiple * risk
                    self.bars_in_trade = 0
                    self.buy(size=size)
                    print(f"🚀🌙 LONG ENTRY @ {price:.2f} | Stop: {stop:.2f} | TP: {self.tp_price:.2f} | Size: {size} ✨")

        # Short conditions
        short_trend = fast < med < slow and price < regime
        short_momentum = (rsi < 50 and rsi < rsi_prev) or (macd_hist < 0 and macd_hist < macd_hist_prev)
        short_breakout = price < swing_low

        if short_trend and short_momentum and vol_expansion and short_breakout:
            stop = max(swing_high, price + self.atr_stop_mult * atr)
            risk = stop - price
            if risk > 0:
                size = int(round(1000000 / price))
                if size > 0:
                    self.entry_price = price
                    self.stop_price = stop
                    self.tp_price = price - self.tp_r_multiple * risk
                    self.bars_in_trade = 0
                    self.sell(size=size)
                    print(f"🚀🌙 SHORT ENTRY @ {price:.2f} | Stop: {stop:.2f} | TP: {self.tp_price:.2f} | Size: {size} ✨")

    def _reset(self):
        self.entry_price = None
        self.stop_price = None
        self.tp_price = None
        self.bars_in_trade = 0
        self.scaled_in = False


print("🌙 Running AdaptiveCascade Backtest... 🚀✨")
bt = Backtest(data, AdaptiveCascade, cash=1000000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙 Moon Dev Backtest Complete! ✨")