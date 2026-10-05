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
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')

print("🌙 Moon Dev GoldenSurge Backtest Initializing... ✨")
print(f"🚀 Data loaded: {len(data)} candles")
print(f"📊 Columns: {list(data.columns)}")


class GoldenSurge(Strategy):
    ema_fast_period = 50
    ema_slow_period = 200
    adx_period = 14
    atr_period = 14
    rsi_period = 5
    adx_threshold = 25
    atr_multiplier = 2.0
    risk_pct = 0.02

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low

        self.ema_fast = self.I(talib.EMA, close, timeperiod=self.ema_fast_period)
        self.ema_slow = self.I(talib.EMA, close, timeperiod=self.ema_slow_period)
        self.adx = self.I(talib.ADX, high, low, close, timeperiod=self.adx_period)
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)
        self.rsi = self.I(talib.RSI, close, timeperiod=self.rsi_period)

        self.stop_price = None
        self.trade_direction = None

        print("🌙✨ Indicators initialized: EMA50, EMA200, ADX, ATR, RSI ✨🌙")

    def next(self):
        # Skip if indicators not ready
        if len(self.data) < self.ema_slow_period + 5:
            return

        price = self.data.Close[-1]
        ema_fast = self.ema_fast[-1]
        ema_slow = self.ema_slow[-1]
        adx_now = self.adx[-1]
        adx_prev = self.adx[-2]
        atr_now = self.atr[-1]
        rsi_now = self.rsi[-1]

        # Rising ADX slope
        adx_rising = adx_now > adx_prev

        # Trend regime
        golden_cross = ema_fast > ema_slow
        death_cross = ema_fast < ema_slow

        # Manage open position
        if self.position:
            # Update trailing stop
            if self.trade_direction == 'long':
                new_stop = price - self.atr_multiplier * atr_now
                if self.stop_price is None or new_stop > self.stop_price:
                    self.stop_price = new_stop

                # Momentum exit
                if rsi_now > 70:
                    print(f"🌙✨ MOMENTUM EXIT LONG | RSI={rsi_now:.2f} | Price={price:.2f} 🚀")
                    self.position.close()
                    self.stop_price = None
                    self.trade_direction = None
                    return

                # Stop loss
                if price <= self.stop_price:
                    print(f"🌙🛑 STOP LOSS LONG | Price={price:.2f} | Stop={self.stop_price:.2f}")
                    self.position.close()
                    self.stop_price = None
                    self.trade_direction = None
                    return

                # Trend invalidation
                if death_cross:
                    print(f"🌙⚠️ TREND INVALIDATION LONG | EMA50<EMA200 | Price={price:.2f}")
                    self.position.close()
                    self.stop_price = None
                    self.trade_direction = None
                    return

            elif self.trade_direction == 'short':
                new_stop = price + self.atr_multiplier * atr_now
                if self.stop_price is None or new_stop < self.stop_price:
                    self.stop_price = new_stop

                if rsi_now < 30:
                    print(f"🌙✨ MOMENTUM EXIT SHORT | RSI={rsi_now:.2f} | Price={price:.2f} 🚀")
                    self.position.close()
                    self.stop_price = None
                    self.trade_direction = None
                    return

                if price >= self.stop_price:
                    print(f"🌙🛑 STOP LOSS SHORT | Price={price:.2f} | Stop={self.stop_price:.2f}")
                    self.position.close()
                    self.stop_price = None
                    self.trade_direction = None
                    return

                if golden_cross:
                    print(f"🌙⚠️ TREND INVALIDATION SHORT | EMA50>EMA200 | Price={price:.2f}")
                    self.position.close()
                    self.stop_price = None
                    self.trade_direction = None
                    return

            return

        # Entry logic
        # Long entry
        if golden_cross and adx_now > self.adx_threshold and adx_rising:
            stop_dist = self.atr_multiplier * atr_now
            if stop_dist <= 0:
                return
            risk_amount = self.equity * self.risk_pct
            position_size = risk_amount / stop_dist
            position_size = int(round(position_size))
            if position_size < 1:
                position_size = 1

            self.stop_price = price - stop_dist
            self.trade_direction = 'long'
            print(f"🌙🚀 LONG ENTRY | Price={price:.2f} | EMA50={ema_fast:.2f} EMA200={ema_slow:.2f} | ADX={adx_now:.2f} | ATR={atr_now:.2f} | Size={position_size}")
            self.buy(size=position_size)
            return

        # Short entry
        if death_cross and adx_now > self.adx_threshold and adx_rising:
            stop_dist = self.atr_multiplier * atr_now
            if stop_dist <= 0:
                return
            risk_amount = self.equity * self.risk_pct
            position_size = risk_amount / stop_dist
            position_size = int(round(position_size))
            if position_size < 1:
                position_size = 1

            self.stop_price = price + stop_dist
            self.trade_direction = 'short'
            print(f"🌙🔻 SHORT ENTRY | Price={price:.2f} | EMA50={ema_fast:.2f} EMA200={ema_slow:.2f} | ADX={adx_now:.2f} | ATR={atr_now:.2f} | Size={position_size}")
            self.sell(size=position_size)
            return


# Initialize backtest
bt = Backtest(
    data,
    GoldenSurge,
    cash=1_000_000,
    commission=0.001
)

print("🌙✨ Running GoldenSurge Backtest... 🚀")
stats = bt.run()
print(stats)
print(stats._strategy)