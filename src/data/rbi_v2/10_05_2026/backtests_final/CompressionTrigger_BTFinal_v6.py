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
    'datetime': 'Datetime',
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

data['Datetime'] = pd.to_datetime(data['Datetime'])
data = data.set_index('Datetime')

print("🌙 Moon Dev: Data loaded successfully! ✨")
print(f"🚀 Data shape: {data.shape}")
print(data.head())


class CompressionTrigger(Strategy):
    bb_period = 20
    bb_std = 2.0
    hv_period = 20
    hv_lookback = 500
    hv_percentile = 20
    atr_period = 14
    vol_ma_period = 20
    squeeze_bars = 3
    volume_mult = 1.5
    risk_pct = 0.02
    atr_stop_mult = 2.0
    atr_tp_mult = 3.0

    def init(self):
        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        volume = pd.Series(self.data.Volume)

        # Bollinger Bands
        self.sma = self.I(talib.SMA, close, timeperiod=self.bb_period)
        self.std = self.I(talib.STDDEV, close, timeperiod=self.bb_period, nbdev=1)
        self.upper = self.I(lambda s, sd: s + self.bb_std * sd, self.sma, self.std)
        self.lower = self.I(lambda s, sd: s - self.bb_std * sd, self.sma, self.std)
        self.bandwidth = self.I(lambda u, l, m: (u - l) / m, self.upper, self.lower, self.sma)

        # Historical Volatility (annualized)
        log_ret = np.log(close / close.shift(1))
        self.hv = self.I(lambda x: pd.Series(x).rolling(self.hv_period).std() * np.sqrt(252) * 100, log_ret)

        # ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        # Volume MA
        self.vol_ma = self.I(talib.SMA, volume, timeperiod=self.vol_ma_period)

        # Trend filter: 200 SMA
        self.sma200 = self.I(talib.SMA, close, timeperiod=200)

        # Rolling percentile thresholds
        self.bw_threshold = self.I(lambda x: pd.Series(x).rolling(self.hv_lookback, min_periods=50).quantile(0.2), self.bandwidth)
        self.hv_threshold = self.I(lambda x: pd.Series(x).rolling(self.hv_lookback, min_periods=50).quantile(self.hv_percentile / 100.0), self.hv)

        # squeeze counter
        self._squeeze_count = 0

        print("🌙 Moon Dev: All indicators initialized! ✨")

    def next(self):
        price = self.data.Close[-1]
        upper = self.upper[-1]
        lower = self.lower[-1]
        bw = self.bandwidth[-1]
        bw_th = self.bw_threshold[-1]
        hv = self.hv[-1]
        hv_th = self.hv_threshold[-1]
        atr = self.atr[-1]
        vol = self.data.Volume[-1]
        vol_ma = self.vol_ma[-1]
        sma200 = self.sma200[-1]

        if (np.isnan(bw_th) or np.isnan(hv_th) or np.isnan(atr) or np.isnan(sma200)
                or np.isnan(bw) or np.isnan(hv) or np.isnan(vol_ma) or np.isnan(upper) or np.isnan(lower)):
            return

        # Squeeze condition
        squeeze = (bw < bw_th) and (hv < hv_th)

        # Track consecutive squeeze bars
        if squeeze:
            self._squeeze_count += 1
        else:
            self._squeeze_count = 0

        had_squeeze = self._squeeze_count >= self.squeeze_bars

        # Manage open positions
        if self.position:
            entry_price = self.trades[-1].entry_price
            if self.position.is_long:
                # Exit if close back inside bands
                if price < upper:
                    print(f"🌙 Moon Dev: LONG exit - price back inside bands at {price:.2f} ✨")
                    self.position.close()
                # Trailing stop check
                elif price < entry_price - self.atr_stop_mult * atr:
                    print(f"🌙 Moon Dev: LONG stop hit at {price:.2f} 🚀")
                    self.position.close()
            elif self.position.is_short:
                if price > lower:
                    print(f"🌙 Moon Dev: SHORT exit - price back inside bands at {price:.2f} ✨")
                    self.position.close()
                elif price > entry_price + self.atr_stop_mult * atr:
                    print(f"🌙 Moon Dev: SHORT stop hit at {price:.2f} 🚀")
                    self.position.close()
            return

        # Entry logic - only if we had a squeeze
        if not had_squeeze:
            return

        # Volume confirmation
        if vol_ma == 0 or vol < self.volume_mult * vol_ma:
            return

        # Long entry
        if price > upper and price > sma200:
            risk_amount = self.equity * self.risk_pct
            stop_dist = self.atr_stop_mult * atr
            if stop_dist <= 0 or price - stop_dist <= 0:
                return
            position_size = int(round(risk_amount / stop_dist))
            if position_size < 1:
                position_size = 1
            max_units = int(self.equity // price)
            if position_size > max_units:
                position_size = max_units
            if position_size < 1:
                return
            print(f"🌙 Moon Dev: 🚀 LONG BREAKOUT! Price={price:.2f} Upper={upper:.2f} Size={position_size} ✨")
            self.buy(size=position_size, sl=price - stop_dist, tp=price + self.atr_tp_mult * atr)
            self._squeeze_count = 0

        # Short entry
        elif price < lower and price < sma200:
            risk_amount = self.equity * self.risk_pct
            stop_dist = self.atr_stop_mult * atr
            if stop_dist <= 0:
                return
            position_size = int(round(risk_amount / stop_dist))
            if position_size < 1:
                position_size = 1
            max_units = int(self.equity // price)
            if position_size > max_units:
                position_size = max_units
            if position_size < 1:
                return
            print(f"🌙 Moon Dev: 🔻 SHORT BREAKOUT! Price={price:.2f} Lower={lower:.2f} Size={position_size} ✨")
            self.sell(size=position_size, sl=price + stop_dist, tp=price - self.atr_tp_mult * atr)
            self._squeeze_count = 0


bt = Backtest(data, CompressionTrigger, cash=1_000_000, commission=0.001)

print("🌙 Moon Dev: Starting backtest... 🚀")
stats = bt.run()
print(stats)
print(stats._strategy)