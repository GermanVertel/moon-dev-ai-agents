import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Load data
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Map columns to backtesting.py requirements
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
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

print("🌙 Moon Dev OscillatorDivergence Backtest Starting... ✨")
print(f"📊 Data loaded: {len(data)} bars")
print(f"🚀 Price range: {data['Close'].min():.2f} - {data['Close'].max():.2f}")


class OscillatorDivergence(Strategy):
    # Strategy parameters
    rsi_period = 14
    rsi_oversold = 30
    rsi_overbought = 70
    atr_period = 14
    atr_zscore_lookback = 20
    atr_zscore_threshold = 1.5
    bb_period = 20
    bb_std = 2.0
    adx_period = 14
    adx_threshold = 20
    risk_pct = 0.01
    swing_lookback = 10
    max_bars_held = 15

    def init(self):
        print("🌙 Initializing Moon Dev indicators... ✨")
        close = self.data.Close
        high = self.data.High
        low = self.data.Low

        # RSI
        self.rsi = self.I(talib.RSI, close, timeperiod=self.rsi_period)

        # ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        # ATR Z-Score
        atr_series = pd.Series(self.atr)
        atr_mean = atr_series.rolling(self.atr_zscore_lookback).mean()
        atr_std = atr_series.rolling(self.atr_zscore_lookback).std()
        atr_zscore = (atr_series - atr_mean) / atr_std
        self.atr_zscore = self.I(lambda: atr_zscore.values)

        # Bollinger Bands
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS, close, timeperiod=self.bb_period,
            nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0
        )

        # ADX
        self.adx = self.I(talib.ADX, high, low, close, timeperiod=self.adx_period)

        # Swing highs/lows
        self.swing_high = self.I(talib.MAX, high, timeperiod=self.swing_lookback)
        self.swing_low = self.I(talib.MIN, low, timeperiod=self.swing_lookback)

        # Track entry bar and stop levels
        self.entry_bar = None
        self.stop_price = None
        self.tp_price = None
        self.trailing_active = False

        print("🚀 Moon Dev indicators initialized! 🌙")

    def next(self):
        # Skip if not enough data
        if len(self.data) < max(self.atr_zscore_lookback, self.swing_lookback) + 5:
            return

        price = self.data.Close[-1]
        rsi_now = self.rsi[-1]
        rsi_prev = self.rsi[-2]
        atr_now = self.atr[-1]
        atr_z = self.atr_zscore[-1]
        adx_now = self.adx[-1]
        bb_upper = self.bb_upper[-1]
        bb_lower = self.bb_lower[-1]
        bb_middle = self.bb_middle[-1]
        swing_high = self.swing_high[-1]
        swing_low = self.swing_low[-1]

        # Skip if indicators are NaN
        if any(np.isnan([rsi_now, rsi_prev, atr_now, atr_z, adx_now,
                          bb_upper, bb_lower, bb_middle, swing_high, swing_low])):
            return

        # ============ POSITION MANAGEMENT ============
        if self.position:
            bars_held = len(self.data) - self.entry_bar

            # Time-based exit
            if bars_held >= self.max_bars_held:
                print(f"⏰ Moon Dev Time Exit at {price:.2f} after {bars_held} bars 🌙")
                self.position.close()
                return

            # Volatility contraction exit
            if atr_z < 0.5:
                print(f"📉 Moon Dev Volatility Exit (ATR Z={atr_z:.2f}) at {price:.2f} 🌙")
                self.position.close()
                return

            if self.position.is_long:
                # Trailing stop activation
                if not self.trailing_active and price >= self.position.entry_price + atr_now:
                    self.trailing_active = True
                    self.stop_price = price - atr_now
                    print(f"🎯 Moon Dev Trailing Stop Activated LONG at {price:.2f} 🌙")

                if self.trailing_active:
                    new_stop = price - atr_now
                    if new_stop > self.stop_price:
                        self.stop_price = new_stop

                # Stop loss
                if price <= self.stop_price:
                    print(f"🛑 Moon Dev Stop Loss LONG at {price:.2f} 🌙")
                    self.position.close()
                    return

                # Take profit at middle BB or RSI 50
                if price >= self.tp_price or rsi_now >= 50:
                    print(f"✅ Moon Dev Take Profit LONG at {price:.2f} (TP={self.tp_price:.2f}, RSI={rsi_now:.1f}) 🚀")
                    self.position.close()
                    return

            elif self.position.is_short:
                # Trailing stop activation
                if not self.trailing_active and price <= self.position.entry_price - atr_now:
                    self.trailing_active = True
                    self.stop_price = price + atr_now
                    print(f"🎯 Moon Dev Trailing Stop Activated SHORT at {price:.2f} 🌙")

                if self.trailing_active:
                    new_stop = price + atr_now
                    if new_stop < self.stop_price:
                        self.stop_price = new_stop

                # Stop loss
                if price >= self.stop_price:
                    print(f"🛑 Moon Dev Stop Loss SHORT at {price:.2f} 🌙")
                    self.position.close()
                    return

                # Take profit
                if price <= self.tp_price or rsi_now <= 50:
                    print(f"✅ Moon Dev Take Profit SHORT at {price:.2f} (TP={self.tp_price:.2f}, RSI={rsi_now:.1f}) 🚀")
                    self.position.close()
                    return

            return

        # ============ ENTRY LOGIC ============
        # Regime filter: trade mean-reversion when ADX < 20
        regime_ok = adx_now < self.adx_threshold

        if not regime_ok:
            return

        # Bullish divergence detection: price lower low, RSI higher low
        price_lower_low = self.data.Low[-1] < self.data.Low[-2] and self.data.Low[-2] <= self.data.Low[-3]
        rsi_higher_low = rsi_now > rsi_prev and rsi_prev <= self.rsi[-3] if not np.isnan(self.rsi[-3]) else False

        # Bearish divergence: price higher high, RSI lower high
        price_higher_high = self.data.High[-1] > self.data.High[-2] and self.data.High[-2] >= self.data.High[-3]
        rsi_lower_high = rsi_now < rsi_prev and rsi_prev >= self.rsi[-3] if not np.isnan(self.rsi[-3]) else False

        # Volatility spike condition
        vol_spike = atr_z > self.atr_zscore_threshold

        # LONG ENTRY
        long_cond = (
            rsi_now < self.rsi_oversold and
            price_lower_low and rsi_higher_low and
            (price <= bb_lower or vol_spike) and
            rsi_now > rsi_prev  # RSI crossing back up (confirmation)
        )

        # SHORT ENTRY
        short_cond = (
            rsi_now > self.rsi_overbought and
            price_higher_high and rsi_lower_high and
            (price >= bb_upper or vol_spike) and
            rsi_now < rsi_prev  # RSI crossing back down
        )

        if long_cond:
            stop = swing_low - 1.5 * atr_now
            risk = price - stop
            if risk <= 0:
                return
            tp = bb_middle
            reward = tp - price
            if reward / risk >= 2.0:
                size = int(round(1000000 / price))
                if size > 0:
                    self.stop_price = stop
                    self.tp_price = tp
                    self.entry_bar = len(self.data)
                    self.trailing_active = False
                    self.buy(size=size)
                    print(f"🌙🚀 Moon Dev LONG ENTRY at {price:.2f} | RSI={rsi_now:.1f} | ATR_Z={atr_z:.2f} | Stop={stop:.2f} | TP={tp:.2f} | Size={size} ✨")

        elif short_cond:
            stop = swing_high + 1.5 * atr_now
            risk = stop - price
            if risk <= 0:
                return
            tp = bb_middle
            reward = price - tp
            if reward / risk >= 2.0:
                size = int(round(1000000 / price))
                if size > 0:
                    self.stop_price = stop
                    self.tp_price = tp
                    self.entry_bar = len(self.data)
                    self.trailing_active = False
                    self.sell(size=size)
                    print(f"🌙🚀 Moon Dev SHORT ENTRY at {price:.2f} | RSI={rsi_now:.1f} | ATR_Z={atr_z:.2f} | Stop={stop:.2f} | TP={tp:.2f} | Size={size} ✨")


# Run backtest
bt = Backtest(data, OscillatorDivergence, cash=1000000, commission=0.002, exclusive_orders=True)
stats = bt.run()
print(stats)
print(stats._strategy)