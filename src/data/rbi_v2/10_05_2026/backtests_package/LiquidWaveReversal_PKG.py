import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# 🌙 Moon Dev's LiquidWave Reversal Strategy ✨
print("🌙 Initializing LiquidWave Reversal Strategy...")
print("🚀 Loading Moon Dev data pipeline...")

# Load data
data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# 🌙 Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# 🚀 Proper column mapping
data.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')
data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

print(f"✨ Data loaded: {len(data)} candles")
print(f"🌙 Date range: {data.index[0]} to {data.index[-1]}")


class LiquidWaveReversal(Strategy):
    # Strategy parameters
    swing_lookback = 20
    vol_ma_period = 20
    rsi_period = 14
    atr_period = 14
    risk_pct = 0.01  # 1% risk per trade
    rr_ratio = 2.0   # Minimum 2:1 R:R

    def init(self):
        print("🌙 Initializing indicators...")
        # Volume MA
        self.vol_ma = self.I(talib.SMA, self.data.Volume, timeperiod=self.vol_ma_period)
        # RSI
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=self.rsi_period)
        # ATR
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close,
                          timeperiod=self.atr_period)
        # Swing highs/lows
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=self.swing_lookback)
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=self.swing_lookback)
        print("✨ Indicators ready! 🚀")

    def next(self):
        if len(self.data) < self.swing_lookback + 5:
            return

        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        open_ = self.data.Open[-1]
        volume = self.data.Volume[-1]
        vol_ma = self.vol_ma[-1]
        rsi = self.rsi[-1]
        rsi_prev = self.rsi[-2]
        atr = self.atr[-1]

        prior_swing_high = self.swing_high[-2]
        prior_swing_low = self.swing_low[-2]

        if np.isnan(vol_ma) or np.isnan(rsi) or np.isnan(atr):
            return

        volume_spike = volume > vol_ma * 1.5

        # 🌙 LONG ENTRY: Sweep of prior swing low + close back above
        if not self.position:
            sweep_low = low < prior_swing_low and price > prior_swing_low
            bullish_div = rsi > rsi_prev and self.data.Close[-1] < self.data.Close[-2]

            if sweep_low and volume_spike:
                print(f"🌙✨ LONG SIGNAL! Sweep low detected at {price:.2f} | Vol spike: {volume:.2f}")
                stop = low - atr * 0.5
                risk = price - stop
                if risk <= 0:
                    return
                target = price + risk * self.rr_ratio
                # Position sizing: risk 1% of equity
                equity = self.equity
                risk_amount = equity * self.risk_pct
                position_size = int(round(risk_amount / risk))
                if position_size <= 0:
                    return
                print(f"🚀 Entering LONG size={position_size} | Entry={price:.2f} SL={stop:.2f} TP={target:.2f}")
                self.buy(size=position_size, sl=stop, tp=target)

            # 🌙 SHORT ENTRY: Sweep of prior swing high + close back below
            sweep_high = high > prior_swing_high and price < prior_swing_high
            bearish_div = rsi < rsi_prev and self.data.Close[-1] > self.data.Close[-2]

            if sweep_high and volume_spike:
                print(f"🌙✨ SHORT SIGNAL! Sweep high detected at {price:.2f} | Vol spike: {volume:.2f}")
                stop = high + atr * 0.5
                risk = stop - price
                if risk <= 0:
                    return
                target = price - risk * self.rr_ratio
                equity = self.equity
                risk_amount = equity * self.risk_pct
                position_size = int(round(risk_amount / risk))
                if position_size <= 0:
                    return
                print(f"🚀 Entering SHORT size={position_size} | Entry={price:.2f} SL={stop:.2f} TP={target:.2f}")
                self.sell(size=position_size, sl=stop, tp=target)


print("🌙 Setting up backtest...")
bt = Backtest(data, LiquidWaveReversal, cash=1_000_000, commission=0.001)

print("🚀 Running backtest...")
stats = bt.run()
print(stats)
print(stats._strategy)