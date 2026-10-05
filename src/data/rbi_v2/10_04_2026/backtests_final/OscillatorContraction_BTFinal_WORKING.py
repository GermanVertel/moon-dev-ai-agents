import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

print("🌙 Moon Dev Backtest AI initializing... ✨")
print("🚀 Loading OscillatorContraction strategy...")

data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"

data = pd.read_csv(data_path)
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

if 'datetime' in data.columns:
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')

data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
print(f"🌙 Data loaded: {len(data)} bars ✨")


class OscillatorContraction(Strategy):
    bb_period = 20
    bb_std = 2.0
    rsi_period = 14
    vol_period = 20
    vol_mult = 1.5
    atr_period = 14
    atr_sl_mult = 1.2
    atr_tp_mult = 1.8
    risk_pct = 0.01
    size = 0.99
    pivot_window = 3

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        self.bb_upper, self.bb_mid, self.bb_lower = self.I(
            talib.BBANDS, close, timeperiod=self.bb_period,
            nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0
        )
        self.rsi = self.I(talib.RSI, close, timeperiod=self.rsi_period)
        self.vol_sma = self.I(talib.SMA, volume, timeperiod=self.vol_period)
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        print("🌙 Indicators initialized: BB, RSI, Vol SMA, ATR ✨")

    def next(self):
        if len(self.data) < max(self.bb_period, self.rsi_period, self.atr_period) + 10:
            return

        price = self.data.Close[-1]
        upper = self.bb_upper[-1]
        lower = self.bb_lower[-1]
        mid = self.bb_mid[-1]
        rsi = self.rsi[-1]
        vol = self.data.Volume[-1]
        vol_avg = self.vol_sma[-1]
        atr = self.atr[-1]

        if np.isnan(upper) or np.isnan(rsi) or np.isnan(vol_avg) or np.isnan(atr):
            return

        vol_ratio = vol / vol_avg if vol_avg > 0 else 0

        if self.position:
            return

        # Bearish divergence: price at/above upper band, RSI < 70 but lower high
        bearish_div = False
        bullish_div = False

        lookback = 10
        if len(self.data) > lookback + 2:
            prior_high_price = max(self.data.High[-lookback:-1])
            prior_rsi_max = max(self.rsi[-lookback:-1])
            prior_low_price = min(self.data.Low[-lookback:-1])
            prior_rsi_min = min(self.rsi[-lookback:-1])

            if self.data.High[-1] > prior_high_price and rsi < prior_rsi_max and rsi > 60:
                bearish_div = True
            if self.data.Low[-1] < prior_low_price and rsi > prior_rsi_min and rsi < 40:
                bullish_div = True

        # SHORT setup
        if price >= upper and rsi > 60 and bearish_div and vol_ratio > self.vol_mult:
            sl = price + self.atr_sl_mult * atr
            tp = price - self.atr_tp_mult * atr
            self.sell(size=self.size, sl=sl, tp=tp)
            print(f"🔻 SHORT entry @ {price:.2f} | RSI={rsi:.1f} | VolRatio={vol_ratio:.2f} | SL={sl:.2f} TP={tp:.2f} 🌙")

        # LONG setup
        elif price <= lower and rsi < 40 and bullish_div and vol_ratio > self.vol_mult:
            sl = price - self.atr_sl_mult * atr
            tp = price + self.atr_tp_mult * atr
            self.buy(size=self.size, sl=sl, tp=tp)
            print(f"🔺 LONG entry @ {price:.2f} | RSI={rsi:.1f} | VolRatio={vol_ratio:.2f} | SL={sl:.2f} TP={tp:.2f} 🚀")


print("🌙 Running backtest... ✨")
bt = Backtest(data, OscillatorContraction, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙 Backtest complete! ✨🚀")