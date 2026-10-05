import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

print("🌙 Moon Dev Backtest AI initializing...")
print("✨ Loading VolatilityContraction strategy...")

data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
print(f"🚀 Data loaded: {len(data)} bars from {data.index[0]} to {data.index[-1]}")


class VolatilityContraction(Strategy):
    bb_period = 20
    bb_std = 2.0
    atr_period = 14
    exit_mult = 2.0
    vol_lookback = 5
    risk_pct = 0.01

    def init(self):
        print("🌙 Initializing indicators...")
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        def _bb_upper(c):
            return talib.BBANDS(np.asarray(c, dtype=np.float64), timeperiod=self.bb_period, nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0)[0]

        def _bb_mid(c):
            return talib.BBANDS(np.asarray(c, dtype=np.float64), timeperiod=self.bb_period, nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0)[1]

        def _bb_lower(c):
            return talib.BBANDS(np.asarray(c, dtype=np.float64), timeperiod=self.bb_period, nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0)[2]

        def _atr(h, l, c):
            return talib.ATR(np.asarray(h, dtype=np.float64),
                             np.asarray(l, dtype=np.float64),
                             np.asarray(c, dtype=np.float64),
                             timeperiod=self.atr_period)

        def _vol_sma(v):
            return talib.SMA(np.asarray(v, dtype=np.float64), timeperiod=self.vol_lookback)

        self.bb_upper = self.I(_bb_upper, close, name='BB_Upper')
        self.bb_mid = self.I(_bb_mid, close, name='BB_Mid')
        self.bb_lower = self.I(_bb_lower, close, name='BB_Lower')
        self.atr = self.I(_atr, high, low, close, name='ATR')
        self.vol_sma = self.I(_vol_sma, volume, name='Vol_SMA')
        print("✨ Indicators ready!")

    def next(self):
        # Need enough history for BB, ATR, and vol_sma
        min_bars = max(self.bb_period, self.atr_period, self.vol_lookback) + 5
        if len(self.data.Close) < min_bars:
            return

        price = self.data.Close[-1]
        upper = self.bb_upper[-1]
        lower = self.bb_lower[-1]
        atr = self.atr[-1]
        vol = self.data.Volume[-1]
        vol_sma = self.vol_sma[-1]

        if np.isnan(atr) or np.isnan(upper) or np.isnan(lower) or np.isnan(vol_sma) or atr <= 0:
            return

        # Volatility filter: skip extreme low/high ATR
        atr_hist = self.atr[-50:] if len(self.atr) >= 50 else self.atr[:]
        atr_hist = np.asarray(atr_hist, dtype=np.float64)
        atr_hist = atr_hist[~np.isnan(atr_hist)]
        if len(atr_hist) > 10:
            atr_mean = np.mean(atr_hist)
            if atr < 0.3 * atr_mean or atr > 3.0 * atr_mean:
                return

        vol_declining = vol < vol_sma

        # Manage open position
        if self.position:
            if len(self.trades) > 0:
                entry = self.trades[-1].entry_price
            else:
                entry = price
            if self.position.is_long:
                stop = entry - self.exit_mult * atr
                if self.data.Low[-1] <= stop:
                    print(f"🌙 Long exit @ {stop:.2f} (ATR stop) | Price: {price:.2f}")
                    self.position.close()
            elif self.position.is_short:
                stop = entry + self.exit_mult * atr
                if self.data.High[-1] >= stop:
                    print(f"🌙 Short exit @ {stop:.2f} (ATR stop) | Price: {price:.2f}")
                    self.position.close()
            return

        # Entry logic - use fractional sizing (percent of equity)
        if price > upper and vol_declining:
            print(f"🚀 LONG signal | Price: {price:.2f} > Upper: {upper:.2f} | Vol: {vol:.2f} < SMA: {vol_sma:.2f}")
            self.buy(size=0.95)

        elif price < lower and vol_declining:
            print(f"🌙 SHORT signal | Price: {price:.2f} < Lower: {lower:.2f} | Vol: {vol:.2f} < SMA: {vol_sma:.2f}")
            self.sell(size=0.95)


print("🌙 Running backtest...")
bt = Backtest(data, VolatilityContraction, cash=1_000_000, commission=0.0005)
stats = bt.run()
print(stats)
print(stats._strategy)
print("✨ Backtest complete! Moon Dev out. 🌙")