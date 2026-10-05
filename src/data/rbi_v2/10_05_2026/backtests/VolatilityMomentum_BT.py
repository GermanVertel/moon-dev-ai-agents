import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# 🌙 Moon Dev's VolatilityMomentum Backtest 🚀

data_path = '/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv'

data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})
data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

print("🌙✨ Moon Dev data loaded:", data.shape)
print(data.head())


class VolatilityMomentum(Strategy):
    ma_fast = 50
    ma_slow = 200
    rsi_period = 14
    atr_period = 14
    rsi_oversold = 30
    volume_lookback = 5
    volume_surge_pct = 0.10
    atr_sl_mult = 2.0
    atr_tp_mult = 1.0  # scale out at 1 ATR profit
    risk_pct = 0.02

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        self.ma50 = self.I(talib.SMA, close, timeperiod=self.ma_fast)
        self.ma200 = self.I(talib.SMA, close, timeperiod=self.ma_slow)
        self.rsi = self.I(talib.RSI, close, timeperiod=self.rsi_period)
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        # volume surge: current volume > 1.10 * volume N bars ago
        vol_prev = self.I(talib.SMA, volume, timeperiod=self.volume_lookback)
        self.vol_surge = volume > (vol_prev * (1 + self.volume_surge_pct))

        print("🌙 Indicators initialized ✨")

    def next(self):
        price = self.data.Close[-1]

        # Need enough history
        if len(self.data) < self.ma_slow + 2:
            return

        # Golden cross detection
        ma50_now = self.ma50[-1]
        ma200_now = self.ma200[-1]
        ma50_prev = self.ma50[-2]
        ma200_prev = self.ma200[-2]

        if np.isnan(ma50_now) or np.isnan(ma200_now) or np.isnan(ma50_prev) or np.isnan(ma200_prev):
            return

        golden_cross = (ma50_prev <= ma200_prev) and (ma50_now > ma200_now)

        rsi_now = self.rsi[-1]
        atr_now = self.atr[-1]
        vol_surge = self.vol_surge[-1]

        # --- Exit management for open position ---
        if self.position:
            entry = self.trades[-1].entry_price
            sl = self.trades[-1].sl
            tp = self.trades[-1].tp
            # Opposite signal exit: MA50 crosses below MA200
            if (ma50_prev >= ma200_prev) and (ma50_now < ma200_now):
                print(f"🌙 Bearish cross detected — exiting long at {price:.2f} 🚀")
                self.position.close()
            return

        # --- Entry logic ---
        if np.isnan(rsi_now) or np.isnan(atr_now):
            return

        if golden_cross and rsi_now < self.rsi_oversold and vol_surge:
            # Extension filter: avoid if price > 3 ATR from MA50
            if abs(price - ma50_now) > 3 * atr_now:
                print("🌙 Skipping — price too extended from MA50 ✨")
                return

            sl_price = price - self.atr_sl_mult * atr_now
            tp_price = price + self.atr_tp_mult * atr_now

            risk_per_unit = price - sl_price
            if risk_per_unit <= 0:
                return

            # Fixed position size per instructions
            size = 1_000_000

            print(f"🌙✨🚀 GOLDEN CROSS + OVERSOLD + VOL SURGE detected!")
            print(f"   Price: {price:.2f} | RSI: {rsi_now:.2f} | ATR: {atr_now:.2f}")
            print(f"   SL: {sl_price:.2f} | TP: {tp_price:.2f} | Size: {size}")
            self.buy(size=size, sl=sl_price, tp=tp_price)


bt = Backtest(data, VolatilityMomentum, cash=10_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)