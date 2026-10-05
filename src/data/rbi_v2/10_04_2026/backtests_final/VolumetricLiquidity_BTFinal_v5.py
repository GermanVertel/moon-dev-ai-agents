import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Load and clean data
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={'datetime': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
data['Date'] = pd.to_datetime(data['Date'])
data = data.set_index('Date')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

print("🌙 Moon Dev VolumetricLiquidity Backtest Initializing... ✨")


class VolumetricLiquidity(Strategy):
    lookback = 30
    vol_ma_period = 20
    atr_period = 14
    bb_period = 20
    bb_std = 2.0
    rsi_period = 14
    vol_mult = 1.5
    rr_ratio = 2.0
    risk_pct = 0.01
    atr_trail_mult = 2.0
    time_exit_bars = 10
    bb_width_threshold = 0.0

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        self.vol_ma = self.I(talib.SMA, volume, timeperiod=self.vol_ma_period)
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)
        self.atr_ma = self.I(talib.SMA, self.atr, timeperiod=self.atr_period)
        self.rsi = self.I(talib.RSI, close, timeperiod=self.rsi_period)

        self.bb_upper, self.bb_mid, self.bb_lower = self.I(
            talib.BBANDS, close, timeperiod=self.bb_period,
            nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0
        )

        self.bb_width = self.I(lambda u, l, m: (u - l) / m, self.bb_upper, self.bb_lower, self.bb_mid)
        self.bb_width_ma = self.I(talib.SMA, self.bb_width, timeperiod=self.bb_period)

        self.cluster_high = self.I(talib.MAX, high, timeperiod=self.lookback)
        self.cluster_low = self.I(talib.MIN, low, timeperiod=self.lookback)

        self.entry_bar = 0

        print("🌙 Indicators loaded: Volume MA, ATR, RSI, Bollinger Bands, Volume Clusters ✨")

    def next(self):
        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        vol = self.data.Volume[-1]

        upper = self.cluster_high[-1]
        lower = self.cluster_low[-1]
        vol_ma = self.vol_ma[-1]
        atr = self.atr[-1]
        atr_ma = self.atr_ma[-1]
        rsi = self.rsi[-1]
        bb_w = self.bb_width[-1]
        bb_w_ma = self.bb_width_ma[-1]

        if np.isnan(upper) or np.isnan(vol_ma) or np.isnan(atr) or np.isnan(rsi) or np.isnan(atr_ma) or np.isnan(bb_w) or np.isnan(bb_w_ma):
            return

        if atr <= 0 or vol_ma <= 0 or bb_w_ma <= 0:
            return

        vol_ok = vol > self.vol_mult * vol_ma
        atr_ok = atr > atr_ma
        bb_ok = bb_w > bb_w_ma

        if not (vol_ok and atr_ok and bb_ok):
            return

        # Long signal
        if price > upper and rsi > 50:
            if not self.position:
                stop = upper - 0.5 * atr
                risk = price - stop
                if risk <= 0:
                    return
                tp = price + self.rr_ratio * risk
                size_frac = self.risk_pct * (price / risk)
                if size_frac <= 0 or size_frac >= 1:
                    size_frac = min(max(size_frac, 0.01), 0.99)
                print(f"🚀 MOON DEV LONG SIGNAL | Price: {price:.2f} | Cluster High: {upper:.2f} | RSI: {rsi:.2f} | Size: {size_frac:.4f}")
                self.buy(size=size_frac, sl=stop, tp=tp)
                self.entry_bar = len(self.data)

        # Short signal
        elif price < lower and rsi < 50:
            if not self.position:
                stop = lower + 0.5 * atr
                risk = stop - price
                if risk <= 0:
                    return
                tp = price - self.rr_ratio * risk
                size_frac = self.risk_pct * (price / risk)
                if size_frac <= 0 or size_frac >= 1:
                    size_frac = min(max(size_frac, 0.01), 0.99)
                print(f"🌑 MOON DEV SHORT SIGNAL | Price: {price:.2f} | Cluster Low: {lower:.2f} | RSI: {rsi:.2f} | Size: {size_frac:.4f}")
                self.sell(size=size_frac, sl=stop, tp=tp)
                self.entry_bar = len(self.data)

        # Trailing stop & time exit
        if self.position:
            bars_held = len(self.data) - self.entry_bar
            if bars_held >= self.time_exit_bars:
                print(f"⏰ MOON DEV TIME EXIT | Bars held: {bars_held}")
                self.position.close()
                return
            if self.position.is_long:
                trail = high - self.atr_trail_mult * atr
                current_sl = self.trades[-1].sl if self.trades and self.trades[-1].sl is not None else -np.inf
                if trail > current_sl:
                    self.trades[-1].sl = trail
            else:
                trail = low + self.atr_trail_mult * atr
                current_sl = self.trades[-1].sl if self.trades and self.trades[-1].sl is not None else np.inf
                if trail < current_sl:
                    self.trades[-1].sl = trail


bt = Backtest(data, VolumetricLiquidity, cash=1_000_000, commission=0.001, exclusive_orders=True)
stats = bt.run()
print(stats)
print(stats._strategy)