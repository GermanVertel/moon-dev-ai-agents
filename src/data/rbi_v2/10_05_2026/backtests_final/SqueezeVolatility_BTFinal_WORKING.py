import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

print("🌙 Moon Dev's SqueezeVolatility Backtest Loading... ✨🚀")

data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper column mapping
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume',
    'datetime': 'Date'
})

data['Date'] = pd.to_datetime(data['Date'])
data = data.set_index('Date')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

print(f"🌙 Data loaded: {len(data)} bars ✨")


class SqueezeVolatility(Strategy):
    bb_period = 20
    bb_std = 2.0
    bbw_lookback = 20
    squeeze_window = 4
    vol_mult = 1.5
    vol_period = 20
    atr_period = 14
    atr_mult = 3.0
    risk_pct = 0.02

    def init(self):
        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        volume = pd.Series(self.data.Volume)

        # Bollinger Bands
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS, close, timeperiod=self.bb_period,
            nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0
        )

        # Bandwidth
        def compute_bbw(upper, middle, lower):
            return (upper - lower) / middle

        self.bbw = self.I(compute_bbw, self.bb_upper, self.bb_middle, self.bb_lower)

        # BBW 20-day low
        def rolling_min(arr, window):
            return pd.Series(arr).rolling(window).min().values

        self.bbw_min = self.I(rolling_min, self.bbw, self.bbw_lookback)

        # Volume SMA
        self.vol_sma = self.I(talib.SMA, volume, timeperiod=self.vol_period)

        # ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        # Parabolic SAR
        self.sar = self.I(talib.SAR, high, low, acceleration=0.02, maximum=0.20)

        print("🌙 Indicators initialized ✨")

    def next(self):
        price = self.data.Close[-1]

        # Track highest high since entry
        if not hasattr(self, 'highest_high'):
            self.highest_high = 0

        if self.position:
            self.highest_high = max(self.highest_high, self.data.High[-1])

            # Exit 1: SAR flip above price
            if self.sar[-1] > price:
                print(f"🌙 SAR Flip Exit at {price:.2f} ✨")
                self.position.close()
                return

            # Exit 2: 3x ATR trailing stop
            if self.atr[-1] > 0:
                trail_stop = self.highest_high - self.atr_mult * self.atr[-1]
                if price < trail_stop:
                    print(f"🌙 ATR Trail Exit at {price:.2f} (stop {trail_stop:.2f}) ✨")
                    self.position.close()
                    return
            return

        # Entry conditions
        if len(self.data) < self.bb_period + 5:
            return

        if np.isnan(self.bbw[-1]) or np.isnan(self.bbw_min[-1]) or np.isnan(self.vol_sma[-1]):
            return

        # 1. Squeeze: BBW was at 20-day low within last N bars
        squeeze_recent = False
        for i in range(1, self.squeeze_window + 1):
            if len(self.bbw) > i and not np.isnan(self.bbw[-i]) and not np.isnan(self.bbw_min[-i]):
                if self.bbw[-i] <= self.bbw_min[-i] + 1e-12:
                    squeeze_recent = True
                    break

        # 2. Close > upper band
        breakout = price > self.bb_upper[-1]

        # 3. Volume >= 1.5x avg
        vol_confirm = self.data.Volume[-1] >= self.vol_mult * self.vol_sma[-1]

        if squeeze_recent and breakout and vol_confirm:
            # Initial stop: below breakout low or middle band, whichever tighter
            stop_price = max(self.data.Low[-1], self.bb_middle[-1])
            risk = price - stop_price
            if risk <= 0:
                return

            risk_amount = self.equity * self.risk_pct
            position_size = int(round(risk_amount / risk))
            if position_size < 1:
                position_size = 1

            print(f"🚀 SQUEEZE BREAKOUT! Price={price:.2f} BBW={self.bbw[-1]:.4f} Vol={self.data.Volume[-1]:.0f} Size={position_size} 🚀")
            self.highest_high = self.data.High[-1]
            self.buy(size=position_size)


bt = Backtest(data, SqueezeVolatility, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)