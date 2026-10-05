import pandas as pd
import numpy as np
import talib
from backtesting import Backtest, Strategy

# Load data
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Ensure proper column mapping
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

# Convert datetime and set index
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')

# 🌙 Ensure numeric dtypes for all OHLCV columns (fixes talib "input array type is not double")
for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    data[col] = pd.to_numeric(data[col], errors='coerce').astype('float64')

data = data.dropna()

print("🌙 Moon Dev VolumetricBandRider backtest initializing...")
print(f"📊 Data loaded: {len(data)} bars")
print(f"🚀 Data range: {data.index[0]} to {data.index[-1]}")


class VolumetricBandRider(Strategy):
    """
    VolumetricBandRider Strategy 🌙
    Bollinger Band breakouts + Volume confirmation
    """

    # Parameters
    bb_period = 20
    bb_std = 2.0
    vol_period = 20
    atr_period = 14
    risk_pct = 0.02  # 2% risk per trade
    rr_ratio = 2.0   # 2:1 reward-to-risk
    position_size = 1_000_000

    def init(self):
        print("🌙 Initializing indicators...")

        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        # 🌙 Convert to numpy arrays with float64 to satisfy talib
        close_arr = np.asarray(close, dtype=np.float64)
        high_arr = np.asarray(high, dtype=np.float64)
        low_arr = np.asarray(low, dtype=np.float64)
        volume_arr = np.asarray(volume, dtype=np.float64)

        # Bollinger Bands
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS, close_arr, timeperiod=self.bb_period,
            nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0,
            name=['BB_Upper', 'BB_Middle', 'BB_Lower']
        )

        # Volume SMA
        self.vol_sma = self.I(
            talib.SMA, volume_arr, timeperiod=self.vol_period,
            name='Vol_SMA'
        )

        # ATR for stop sizing
        self.atr = self.I(
            talib.ATR, high_arr, low_arr, close_arr, timeperiod=self.atr_period,
            name='ATR'
        )

        print("✨ Indicators ready: BB(20,2), VolSMA(20), ATR(14)")

    def next(self):
        price = self.data.Close[-1]
        vol = self.data.Volume[-1]

        upper = self.bb_upper[-1]
        lower = self.bb_lower[-1]
        vol_avg = self.vol_sma[-1]
        atr = self.atr[-1]

        # Skip if indicators not ready
        if np.isnan(upper) or np.isnan(lower) or np.isnan(vol_avg) or np.isnan(atr):
            return

        # Manage open positions
        if self.position:
            # 🌙 Position object has no .entry_price; use last trade entry price
            entry = self.trades[-1].entry_price if len(self.trades) > 0 else price
            if self.position.is_long:
                # Exit if price re-enters bands (mean reversion)
                if price < upper:
                    print(f"🌙 Long exit signal: price {price:.2f} re-entered bands (upper {upper:.2f})")
                    self.position.close()
            else:
                if price > lower:
                    print(f"🌙 Short exit signal: price {price:.2f} re-entered bands (lower {lower:.2f})")
                    self.position.close()
            return

        # Long entry: close above upper band + volume > vol SMA
        if price > upper and vol > vol_avg:
            stop = price - atr
            target = price + (atr * self.rr_ratio)
            size = int(round(self.position_size * self.risk_pct / atr)) if atr > 0 else 0
            if size > 0:
                print(f"🚀 LONG entry: price {price:.2f} > upper {upper:.2f} | vol {vol:.2f} > volSMA {vol_avg:.2f}")
                print(f"   SL={stop:.2f} TP={target:.2f} size={size}")
                self.buy(size=size, sl=stop, tp=target)

        # Short entry: close below lower band + volume < vol SMA
        elif price < lower and vol < vol_avg:
            stop = price + atr
            target = price - (atr * self.rr_ratio)
            size = int(round(self.position_size * self.risk_pct / atr)) if atr > 0 else 0
            if size > 0:
                print(f"🔻 SHORT entry: price {price:.2f} < lower {lower:.2f} | vol {vol:.2f} < volSMA {vol_avg:.2f}")
                print(f"   SL={stop:.2f} TP={target:.2f} size={size}")
                self.sell(size=size, sl=stop, tp=target)


# Run backtest
print("🌙 Starting VolumetricBandRider backtest...")
bt = Backtest(data, VolumetricBandRider, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)