import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

print("🌙 Moon Dev's LiquiditySqueeze Backtest Loading... ✨")

data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
# 🌙 Ensure all OHLCV columns are float64 for talib compatibility
for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    data[col] = data[col].astype(float)
print(f"🚀 Data loaded: {len(data)} bars 🌙")


class LiquiditySqueeze(Strategy):
    bb_period = 20
    bb_std = 2.0
    vol_period = 20
    atr_period = 14
    squeeze_lookback = 100
    squeeze_percentile = 20
    vol_mult = 1.5
    risk_pct = 0.01
    swing_lookback = 20

    def init(self):
        # 🌙 Bollinger Bands via talib (returns upper, middle, lower)
        def bb_upper(close):
            u, m, l = talib.BBANDS(close, timeperiod=self.bb_period,
                                   nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0)
            return u

        def bb_middle(close):
            u, m, l = talib.BBANDS(close, timeperiod=self.bb_period,
                                   nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0)
            return m

        def bb_lower(close):
            u, m, l = talib.BBANDS(close, timeperiod=self.bb_period,
                                   nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0)
            return l

        self.bb_upper = self.I(bb_upper, self.data.Close)
        self.bb_middle = self.I(bb_middle, self.data.Close)
        self.bb_lower = self.I(bb_lower, self.data.Close)

        # 🌙 Bandwidth = (upper - lower) / middle * 100
        self.bandwidth = self.I(
            lambda u, m, l: (u - l) / m * 100,
            self.bb_upper, self.bb_middle, self.bb_lower
        )
        # 🌙 Volume moving average — cast Volume to float64 array for talib
        self.vol_ma = self.I(talib.SMA, self.data.Volume.astype(float), timeperiod=self.vol_period)
        # 🌙 ATR for risk sizing
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period)
        # 🌙 Swing highs / lows
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=self.swing_lookback)
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=self.swing_lookback)
        print("✨ Indicators initialized: BB, Bandwidth, VolMA, ATR, Swing 🌙")

    def next(self):
        if len(self.data) < self.squeeze_lookback + 5:
            return

        price = self.data.Close[-1]
        bw = self.bandwidth[-1]
        bw_window = np.array(self.bandwidth[-self.squeeze_lookback:], dtype=float)
        bw_window = bw_window[~np.isnan(bw_window)]
        if len(bw_window) < 10:
            return
        bw_thresh = np.percentile(bw_window, self.squeeze_percentile)
        in_squeeze = bw < bw_thresh

        vol = self.data.Volume[-1]
        vol_ma = self.vol_ma[-1]
        if vol_ma is None or np.isnan(vol_ma) or vol_ma == 0:
            return
        vol_surge = vol > self.vol_mult * vol_ma

        upper = self.bb_upper[-1]
        lower = self.bb_lower[-1]
        atr = self.atr[-1]
        sh = self.swing_high[-1]
        sl = self.swing_low[-1]

        if any(v is None or np.isnan(v) for v in [upper, lower, atr, sh, sl]):
            return

        if not self.position:
            if in_squeeze and vol_surge:
                # 🌙 Bullish breakout: close above upper band and rising
                if price > upper and price > self.data.Close[-2]:
                    stop = max(sl, price - atr)
                    risk = price - stop
                    if risk > 0:
                        size = int(round((self.equity * self.risk_pct) / risk))
                        if size > 0:
                            print(f"🚀 LONG BREAKOUT! Price={price:.2f} Upper={upper:.2f} BW={bw:.3f} VolSurge={vol/vol_ma:.2f}x 🌙")
                            self.buy(size=size, sl=stop, tp=price + 2 * atr)
                # 🌙 Bearish breakout: close below lower band and falling
                elif price < lower and price < self.data.Close[-2]:
                    stop = min(sh, price + atr)
                    risk = stop - price
                    if risk > 0:
                        size = int(round((self.equity * self.risk_pct) / risk))
                        if size > 0:
                            print(f"🔻 SHORT BREAKOUT! Price={price:.2f} Lower={lower:.2f} BW={bw:.3f} VolSurge={vol/vol_ma:.2f}x 🌙")
                            self.sell(size=size, sl=stop, tp=price - 2 * atr)
        else:
            bw_80 = np.percentile(bw_window, 80)
            if self.position.is_long:
                if price < self.bb_middle[-1] or bw > bw_80:
                    print(f"🌙 Exit LONG: BW={bw:.3f} Price={price:.2f} ✨")
                    self.position.close()
            elif self.position.is_short:
                if price > self.bb_middle[-1] or bw > bw_80:
                    print(f"🌙 Exit SHORT: BW={bw:.3f} Price={price:.2f} ✨")
                    self.position.close()


bt = Backtest(data, LiquiditySqueeze, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)