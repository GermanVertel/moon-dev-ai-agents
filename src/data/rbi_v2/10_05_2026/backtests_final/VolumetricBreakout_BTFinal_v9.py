import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Load data
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

# Ensure all OHLCV are float64 (talib requires double)
for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    data[col] = data[col].astype(np.float64)

# Resample to weekly
data = data.resample('W').agg({
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'Close': 'last',
    'Volume': 'sum'
}).dropna()

# Re-cast after resample (safety)
for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    data[col] = data[col].astype(np.float64)

print("🌙✨ Moon Dev: Data loaded and resampled to weekly! 🚀")
print(f"🌙 Data shape: {data.shape}")
print(f"🌙 Dtypes:\n{data.dtypes}")


class VolumetricBreakout(Strategy):
    bb_period = 20
    bb_std = 2.0
    atr_period = 20
    vol_threshold = 0.02
    obv_high_period = 10
    consolidation_min = 3
    vol_avg_period = 20
    ema_period = 20
    risk_pct = 0.02
    atr_mult_stop = 1.5

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        # Bollinger Bands
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS, close, timeperiod=self.bb_period,
            nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0
        )

        # ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        # OBV
        self.obv = self.I(talib.OBV, close, volume)

        # OBV 10-week high
        self.obv_high = self.I(talib.MAX, self.obv, timeperiod=self.obv_high_period)

        # Volume SMA
        self.vol_sma = self.I(talib.SMA, volume, timeperiod=self.vol_avg_period)

        # 20-week EMA for trailing stop
        self.ema = self.I(talib.EMA, close, timeperiod=self.ema_period)

        # Weekly volatility (ATR / Close)
        def _weekly_vol():
            atr_arr = np.array(self.atr)
            close_arr = np.array(self.data.Close)
            return atr_arr / close_arr
        self.weekly_vol = self.I(_weekly_vol)

        print("🌙✨ Indicators initialized! 🚀")

    def next(self):
        if len(self.data) < max(self.bb_period, self.atr_period, self.obv_high_period, self.vol_avg_period) + self.consolidation_min + 2:
            return

        price = self.data.Close[-1]
        upper = self.bb_upper[-1]
        middle = self.bb_middle[-1]
        lower = self.bb_lower[-1]
        vol = self.weekly_vol[-1]
        obv_now = self.obv[-1]
        obv_high = self.obv_high[-1]
        volume = self.data.Volume[-1]
        vol_avg = self.vol_sma[-1]
        ema = self.ema[-1]
        atr = self.atr[-1]

        # Check volatility filter
        vol_ok = vol > self.vol_threshold

        # Check consolidation in lower BBL for last N weeks
        consolidation_ok = True
        for i in range(2, self.consolidation_min + 2):
            c = self.data.Close[-i]
            lo = self.bb_lower[-i]
            mid = self.bb_middle[-i]
            if not (lo <= c <= mid):
                consolidation_ok = False
                break

        # OBV rising during consolidation
        obv_rising = self.obv[-1] >= self.obv[-2] if len(self.obv) > 1 else False

        # Breakout conditions
        breakout = price > upper
        obv_breakout = obv_now >= obv_high
        volume_confirm = volume > vol_avg

        # Entry
        if not self.position:
            if vol_ok and consolidation_ok and breakout and obv_breakout and volume_confirm:
                # Position sizing based on risk
                stop_price = price - self.atr_mult_stop * atr
                risk_per_unit = price - stop_price
                if risk_per_unit > 0:
                    equity = self.equity
                    risk_amount = equity * self.risk_pct
                    units = risk_amount / risk_per_unit
                    # Convert to fraction of equity for backtesting.py compatibility
                    size_frac = (units * price) / equity
                    # Ensure size is a valid fraction between 0 and 1
                    size_frac = float(min(max(size_frac, 0.01), 0.99))
                    self.buy(size=size_frac)
                    print(f"🌙✨🚀 MOON DEV BREAKOUT ENTRY! Price: {price:.2f}, Upper BB: {upper:.2f}, SizeFrac: {size_frac:.4f}, OBV Breakout: {obv_breakout}, Vol: {vol:.2%}")

        # Exit logic
        else:
            # Exit if close below middle band
            if price < middle:
                self.position.close()
                print(f"🌙 EXIT: Price closed below middle band at {price:.2f}")
            # Trailing stop via EMA
            elif price < ema:
                self.position.close()
                print(f"🌙 EXIT: Price closed below EMA at {price:.2f}")


bt = Backtest(data, VolumetricBreakout, cash=1_000_000, commission=0.002)
stats = bt.run()
print(stats)
print(stats._strategy)