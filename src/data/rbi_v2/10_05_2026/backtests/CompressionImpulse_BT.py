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
    'datetime': 'Date',
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

# Set datetime index
data['Date'] = pd.to_datetime(data['Date'])
data = data.set_index('Date')

print("🌙 Moon Dev CompressionImpulse Backtest Loading... ✨")
print(f"📊 Data shape: {data.shape}")
print(f"🚀 Data head:\n{data.head()}")


class CompressionImpulse(Strategy):
    bb_period = 20
    bb_std = 2.0
    rsi_period = 14
    atr_period = 14
    squeeze_lookback = 20
    risk_pct = 0.02
    time_stop_bars = 15

    def init(self):
        print("🌙 Initializing Moon Dev CompressionImpulse indicators... ✨")
        close = pd.Series(self.data.Close)

        # Bollinger Bands
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS, close, timeperiod=self.bb_period,
            nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0
        )

        # Bollinger Band Width
        bb_width = (self.bb_upper - self.bb_lower) / self.bb_middle
        self.bb_width = self.I(lambda: bb_width, name='BB_Width')

        # Rolling min of BB width for squeeze detection
        self.bb_width_min = self.I(
            talib.MIN, bb_width, timeperiod=self.squeeze_lookback,
            name='BB_Width_Min'
        )

        # RSI
        self.rsi = self.I(talib.RSI, close, timeperiod=self.rsi_period, name='RSI')

        # ATR
        self.atr = self.I(
            talib.ATR, pd.Series(self.data.High), pd.Series(self.data.Low),
            close, timeperiod=self.atr_period, name='ATR'
        )

        print("🌙 Indicators initialized! Ready to hunt squeezes 🚀")

    def next(self):
        price = self.data.Close[-1]

        # Skip if indicators not ready
        if (np.isnan(self.bb_width[-1]) or np.isnan(self.bb_width_min[-1]) or
                np.isnan(self.rsi[-1]) or np.isnan(self.atr[-1]) or
                np.isnan(self.rsi[-2])):
            return

        # Manage open position
        if self.position:
            entry = self.trades[-1].entry_price
            atr_val = self.atr[-1]

            # Time stop
            bars_held = len(self.data) - self.trades[-1].entry_bar
            if bars_held >= self.time_stop_bars:
                print(f"⏰ Moon Dev Time Stop hit at {price:.2f} after {bars_held} bars 🌙")
                self.position.close()
                return

            # Profit target: 2 * ATR above entry
            target = entry + 2 * atr_val
            # Stop loss: 1 * ATR below entry
            stop = entry - 1 * atr_val

            if self.data.High[-1] >= target:
                print(f"🎯 Moon Dev Profit Target hit! Entry: {entry:.2f} | Target: {target:.2f} | Price: {price:.2f} 💰")
                self.position.close()
            elif self.data.Low[-1] <= stop:
                print(f"🛑 Moon Dev Stop Loss hit! Entry: {entry:.2f} | Stop: {stop:.2f} | Price: {price:.2f} 😢")
                self.position.close()
            return

        # Entry conditions
        squeeze = self.bb_width[-1] <= self.bb_width_min[-1]
        rsi_cross_up = self.rsi[-2] <= 50 and self.rsi[-1] > 50

        if squeeze and rsi_cross_up:
            atr_val = self.atr[-1]
            stop = price - atr_val
            risk_per_unit = price - stop

            if risk_per_unit <= 0:
                return

            # Position sizing: risk 2% of equity
            equity = self.equity
            risk_amount = equity * self.risk_pct
            position_size = int(round(risk_amount / risk_per_unit))

            if position_size <= 0:
                return

            print(f"🌙✨ COMPRESSION IMPULSE SIGNAL DETECTED! ✨🌙")
            print(f"   💥 Squeeze: BB_Width {self.bb_width[-1]:.4f} <= Min {self.bb_width_min[-1]:.4f}")
            print(f"   📈 RSI Cross: {self.rsi[-2]:.2f} -> {self.rsi[-1]:.2f}")
            print(f"   🎯 Entry: {price:.2f} | Stop: {stop:.2f} | ATR: {atr_val:.2f}")
            print(f"   🚀 Size: {position_size} units")

            self.buy(size=position_size)


# Run backtest
bt = Backtest(data, CompressionImpulse, cash=1_000_000, commission=0.002)
stats = bt.run()
print(stats)
print(stats._strategy)