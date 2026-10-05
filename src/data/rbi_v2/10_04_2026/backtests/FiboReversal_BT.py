import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Load data
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

print("🌙✨ Moon Dev FiboReversal Backtest Loading... 🚀")
print(f"📊 Data shape: {data.shape}")
print(f"📈 Date range: {data.index[0]} to {data.index[-1]}")


class FiboReversal(Strategy):
    # Strategy parameters
    ema_period = 50
    rsi_period = 14
    atr_period = 14
    swing_lookback = 50
    fib_low = 0.618
    fib_high = 0.786
    rsi_bearish_threshold = 40
    rsi_reversal_threshold = 35
    atr_tp_mult = 2.5
    atr_sl_mult = 1.5
    risk_pct = 0.015  # 1.5% risk per trade

    def init(self):
        print("🌙 Initializing indicators... ✨")
        self.ema = self.I(talib.EMA, self.data.Close, timeperiod=self.ema_period)
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=self.rsi_period)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period)
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=self.swing_lookback)
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=self.swing_lookback)
        print("🚀 Indicators ready!")

    def next(self):
        if len(self.data) < self.swing_lookback + 5:
            return

        price = self.data.Close[-1]
        ema_now = self.ema[-1]
        ema_prev = self.ema[-5]
        rsi_now = self.rsi[-1]
        rsi_prev = self.rsi[-2]
        atr_now = self.atr[-1]
        sh = self.swing_high[-1]
        sl = self.swing_low[-1]

        # Downtrend context: EMA sloping down
        downtrend = ema_now < ema_prev

        # Fibonacci levels from swing high to swing low
        if sh <= sl:
            return
        diff = sh - sl
        fib_618 = sh - diff * self.fib_low
        fib_786 = sh - diff * self.fib_high

        # Zone between 0.618 and 0.786
        in_fib_zone = (fib_786 <= price <= fib_618) or (fib_618 <= price <= fib_786)

        # Bearish momentum: RSI was below threshold recently
        bearish_momentum = rsi_prev < self.rsi_bearish_threshold

        # Reversal trigger: RSI turning up
        rsi_turning_up = rsi_now > rsi_prev and rsi_prev < self.rsi_reversal_threshold

        # Entry
        if not self.position:
            if downtrend and in_fib_zone and bearish_momentum and rsi_turning_up:
                # Risk management: position size based on risk
                stop_price = sl - atr_now * 0.5
                risk_per_unit = price - stop_price
                if risk_per_unit <= 0:
                    return

                equity = self.equity
                risk_amount = equity * self.risk_pct
                position_size = int(round(risk_amount / risk_per_unit))
                if position_size < 1:
                    position_size = 1

                tp_price = price + atr_now * self.atr_tp_mult

                self.buy(size=position_size, sl=stop_price, tp=tp_price)
                print(f"🌙✨ FiboReversal LONG! Price: {price:.2f} | Fib zone: {fib_786:.2f}-{fib_618:.2f} | RSI: {rsi_now:.2f} | Size: {position_size} 🚀")
        else:
            # Secondary exit: RSI bearish divergence signal (RSI drops sharply)
            if rsi_now < rsi_prev and rsi_now > 70:
                self.position.close()
                print(f"🌙 RSI exhaustion exit at {price:.2f} ✨")


# Run backtest
print("🌙 Starting backtest run... 🚀")
bt = Backtest(data, FiboReversal, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙✨ Backtest complete! 🚀")