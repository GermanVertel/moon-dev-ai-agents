import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

print("🌙✨ Moon Dev Backtest AI initializing... VolumetricSqueeze 🚀")

# Load data
data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper column mapping
data.rename(columns={
    'datetime': 'Date',
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

print(f"🌙 Data loaded: {len(data)} bars ✨")
print(f"🚀 Columns: {list(data.columns)}")


class VolumetricSqueeze(Strategy):
    cmf_period = 20
    bb_period = 20
    bb_std = 2.0
    squeeze_lookback = 100
    squeeze_pct = 0.20
    atr_period = 14
    atr_mult = 1.5
    risk_pct = 0.02

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        # Chaikin Money Flow
        self.cmf = self.I(talib.ADOSC, high, low, close, volume,
                          fastperiod=3, slowperiod=self.cmf_period, name="CMF")

        # Bollinger Bands
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS, close, timeperiod=self.bb_period,
            nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0,
            name="BB"
        )

        # BandWidth = (upper - lower) / middle
        bw = (self.bb_upper - self.bb_lower) / self.bb_middle
        self.bandwidth = self.I(lambda: bw, name="BandWidth")

        # Squeeze threshold: lowest 20th percentile of BandWidth over lookback
        self.bw_threshold = self.I(
            lambda: pd.Series(bw).rolling(self.squeeze_lookback).quantile(self.squeeze_pct).values,
            name="BW_Threshold"
        )

        # ATR for stop
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period, name="ATR")

        print("🌙✨ Indicators initialized: CMF, BB, BandWidth, ATR 🚀")

    def next(self):
        if len(self.data) < max(self.squeeze_lookback, self.bb_period, self.cmf_period) + 2:
            return

        price = self.data.Close[-1]
        cmf_now = self.cmf[-1]
        cmf_prev = self.cmf[-2]
        bw_now = self.bandwidth[-1]
        bw_thresh = self.bw_threshold[-1]
        atr_now = self.atr[-1]

        if np.isnan(cmf_now) or np.isnan(cmf_prev) or np.isnan(bw_thresh) or np.isnan(atr_now):
            return

        # Entry: CMF crosses above zero AND squeeze active
        cmf_cross_up = cmf_prev <= 0 and cmf_now > 0
        squeeze_active = bw_now <= bw_thresh

        if not self.position:
            if cmf_cross_up and squeeze_active:
                stop_price = price - self.atr_mult * atr_now
                risk_per_unit = price - stop_price
                if risk_per_unit <= 0:
                    return
                risk_amount = self.equity * self.risk_pct
                size = int(round(risk_amount / risk_per_unit))
                if size < 1:
                    size = 1
                print(f"🌙🚀 LONG SIGNAL | Price={price:.2f} CMF={cmf_now:.4f} BW={bw_now:.4f} < Thresh={bw_thresh:.4f} | Size={size}")
                self.buy(size=size, sl=stop_price)
        else:
            # Exit on CMF cross below zero
            cmf_cross_down = cmf_prev >= 0 and cmf_now < 0
            # Exit on upper band touch
            upper_touch = self.data.High[-1] >= self.bb_upper[-1]

            if cmf_cross_down:
                print(f"🌙✨ EXIT: CMF cross down ({cmf_prev:.4f} -> {cmf_now:.4f}) at {price:.2f}")
                self.position.close()
            elif upper_touch:
                print(f"🌙🎯 EXIT: Upper BB touch at {price:.2f} (upper={self.bb_upper[-1]:.2f})")
                self.position.close()


bt = Backtest(data, VolumetricSqueeze, cash=1_000_000, commission=0.001)

print("🌙✨ Running backtest... 🚀")
stats = bt.run()
print(stats)
print(stats._strategy)