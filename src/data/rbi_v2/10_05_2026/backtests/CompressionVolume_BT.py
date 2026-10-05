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
print("🌙✨ Moon Dev Data Loaded! Shape:", data.shape, "🚀")


class CompressionVolume(Strategy):
    bb_period = 20
    bb_std = 2.0
    lookback = 30
    vol_ma_period = 20
    vol_mult = 1.2
    time_stop = 15
    risk_pct = 0.02

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS, close, timeperiod=self.bb_period,
            nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0
        )
        self.band_width = self.I(
            lambda u, m, l: (u - l) / m, self.bb_upper, self.bb_middle, self.bb_lower
        )
        self.bw_min = self.I(talib.MIN, self.band_width, timeperiod=self.lookback)
        self.vol_ma = self.I(talib.SMA, volume, timeperiod=self.vol_ma_period)

        self.entry_bar = None
        self.entry_bw = None
        print("🌙 CompressionVolume indicators initialized! 🚀")

    def next(self):
        price = self.data.Close[-1]
        upper = self.bb_upper[-1]
        lower = self.bb_lower[-1]
        middle = self.bb_middle[-1]
        bw = self.band_width[-1]
        bw_min = self.bw_min[-1]
        vol = self.data.Volume[-1]
        vol_ma = self.vol_ma[-1]

        if np.isnan(bw) or np.isnan(bw_min) or np.isnan(vol_ma):
            return

        squeeze = bw <= bw_min * 1.05
        vol_spike = vol > vol_ma * self.vol_mult

        # Manage open position
        if self.position:
            bars_held = len(self.data) - self.entry_bar
            if self.position.is_long:
                if price < middle:
                    print(f"🌙 Long exit: price {price:.2f} crossed below middle {middle:.2f} ✨")
                    self.position.close()
                elif price < lower:
                    print(f"🌙 Long exit: price re-entered bands 🚀")
                    self.position.close()
                elif bars_held >= self.time_stop:
                    print(f"🌙 Long time-stop exit after {bars_held} bars ⏰")
                    self.position.close()
            else:
                if price > middle:
                    print(f"🌙 Short exit: price {price:.2f} crossed above middle {middle:.2f} ✨")
                    self.position.close()
                elif price > upper:
                    print(f"🌙 Short exit: price re-entered bands 🚀")
                    self.position.close()
                elif bars_held >= self.time_stop:
                    print(f"🌙 Short time-stop exit after {bars_held} bars ⏰")
                    self.position.close()
            return

        # Long entry
        if squeeze and vol_spike and price > upper:
            risk = price - lower
            if risk <= 0:
                return
            risk_amt = self.equity * self.risk_pct
            size = int(round(risk_amt / risk))
            if size > 0:
                print(f"🚀🌙 LONG breakout! price={price:.2f} upper={upper:.2f} bw={bw:.4f} bw_min={bw_min:.4f} vol={vol:.2f} vol_ma={vol_ma:.2f} size={size}")
                self.buy(size=size)
                self.entry_bar = len(self.data)
                self.entry_bw = bw

        # Short entry
        elif squeeze and vol_spike and price < lower:
            risk = upper - price
            if risk <= 0:
                return
            risk_amt = self.equity * self.risk_pct
            size = int(round(risk_amt / risk))
            if size > 0:
                print(f"🚀🌙 SHORT breakout! price={price:.2f} lower={lower:.2f} bw={bw:.4f} bw_min={bw_min:.4f} vol={vol:.2f} vol_ma={vol_ma:.2f} size={size}")
                self.sell(size=size)
                self.entry_bar = len(self.data)
                self.entry_bw = bw


bt = Backtest(data, CompressionVolume, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)