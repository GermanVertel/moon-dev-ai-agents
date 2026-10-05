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

# Ensure datetime index
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')

print("🌙 Moon Dev OscillatorBandSurge backtest initializing... ✨")
print(f"📊 Data loaded: {len(data)} bars from {data.index[0]} to {data.index[-1]}")


class OscillatorBandSurge(Strategy):
    # Parameters
    bb_period = 20
    bb_std = 2.0
    rsi_period = 14
    rsi_bb_period = 20
    rsi_bb_std = 2.0
    atr_period = 14
    atr_mult = 2.0
    risk_pct = 0.02
    vol_period = 20

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        # Price Bollinger Bands
        self.bb_mid = self.I(talib.SMA, close, timeperiod=self.bb_period)
        self.bb_stddev = self.I(talib.STDDEV, close, timeperiod=self.bb_period, nbdev=1)
        self.bb_upper = self.I(lambda: self.bb_mid + self.bb_std * self.bb_stddev)
        self.bb_lower = self.I(lambda: self.bb_mid - self.bb_std * self.bb_stddev)

        # RSI
        self.rsi = self.I(talib.RSI, close, timeperiod=self.rsi_period)

        # RSI Bollinger Bands
        self.rsi_mid = self.I(talib.SMA, self.rsi, timeperiod=self.rsi_bb_period)
        self.rsi_stddev = self.I(talib.STDDEV, self.rsi, timeperiod=self.rsi_bb_period, nbdev=1)
        self.rsi_upper = self.I(lambda: self.rsi_mid + self.rsi_bb_std * self.rsi_stddev)
        self.rsi_lower = self.I(lambda: self.rsi_mid - self.rsi_bb_std * self.rsi_stddev)

        # ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        # Volume average
        self.vol_avg = self.I(talib.SMA, volume, timeperiod=self.vol_period)

        # Swing low (20-period)
        self.swing_low = self.I(talib.MIN, low, timeperiod=20)

        print("🌙✨ Indicators initialized successfully! 🚀")

    def next(self):
        price = self.data.Close[-1]

        # Skip if indicators not ready
        if (np.isnan(self.bb_mid[-1]) or np.isnan(self.rsi_upper[-1]) or
                np.isnan(self.atr[-1]) or np.isnan(self.rsi_mid[-1]) or
                np.isnan(self.vol_avg[-1]) or np.isnan(self.swing_low[-1])):
            return

        if self.position:
            # Primary Exit: Close below price Middle BB
            if price < self.bb_mid[-1]:
                print(f"🌙 EXIT (below mid BB) @ {price:.2f} | RSI={self.rsi[-1]:.2f}")
                self.position.close()
                return

            # Momentum Exit: RSI crosses below RSI Middle Band
            if self.rsi[-1] < self.rsi_mid[-1] and self.rsi[-2] >= self.rsi_mid[-2]:
                print(f"🌙 EXIT (RSI momentum fade) @ {price:.2f} | RSI={self.rsi[-1]:.2f}")
                self.position.close()
                return

            # Take Profit at price Upper BB
            if price >= self.bb_upper[-1]:
                print(f"🚀 TAKE PROFIT (upper BB) @ {price:.2f}")
                self.position.close()
                return

            return

        # === ENTRY LOGIC ===
        # 1. RSI crosses above its upper BB (fresh cross)
        rsi_cross_up = (self.rsi[-1] > self.rsi_upper[-1] and
                        self.rsi[-2] <= self.rsi_upper[-2])

        # 2. Close breaks above price middle BB
        price_above_mid = price > self.bb_mid[-1] and self.data.Close[-2] <= self.bb_mid[-2]

        # Optional filter: RSI > 50
        rsi_bullish = self.rsi[-1] > 50

        # Volume confirmation
        vol_confirm = self.data.Volume[-1] > self.vol_avg[-1]

        if rsi_cross_up and price_above_mid and rsi_bullish:
            # Stop loss: tighter of swing low or lower BB, but use ATR-based for adaptation
            atr_stop = price - self.atr_mult * self.atr[-1]
            swing_stop = self.swing_low[-1]
            bb_stop = self.bb_lower[-1]

            # Use the tightest (highest) stop that's below price
            candidates = [s for s in [atr_stop, swing_stop, bb_stop] if s < price]
            stop_price = max(candidates) if candidates else price * 0.98

            risk_per_unit = price - stop_price
            if risk_per_unit <= 0:
                return

            # Position sizing: risk 2% of equity
            equity = self.equity
            risk_amount = equity * self.risk_pct
            position_size = int(round(risk_amount / risk_per_unit))

            if position_size <= 0:
                return

            vol_tag = "📈 VOL OK" if vol_confirm else "⚠️ LOW VOL"
            print(f"🌙✨ ENTRY SIGNAL {vol_tag} @ {price:.2f} | "
                  f"RSI={self.rsi[-1]:.2f} > RSI_UB={self.rsi_upper[-1]:.2f} | "
                  f"SL={stop_price:.2f} | Size={position_size} 🚀")

            self.buy(size=position_size, sl=stop_price)


# Run backtest
bt = Backtest(data, OscillatorBandSurge, cash=1_000_000, commission=0.001)

stats = bt.run()
print(stats)
print(stats._strategy)