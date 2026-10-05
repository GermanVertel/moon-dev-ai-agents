import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Load data
data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Map to proper case
data.columns = [col.capitalize() for col in data.columns]

# Ensure datetime
if 'Datetime' in data.columns:
    data['Datetime'] = pd.to_datetime(data['Datetime'])
    data = data.set_index('Datetime')

print("🌙✨ Moon Dev KineticVolatility Backtest Loading... 🚀")
print(f"📊 Data shape: {data.shape}")
print(f"📈 Columns: {list(data.columns)}")


class KineticVolatility(Strategy):
    # Strategy parameters
    bb_period = 20
    bb_std = 2.0
    atr_period = 14
    vol_ma_period = 20
    lookback_high = 20
    compression_lookback = 50
    compression_pct = 0.25  # lowest 25%
    volume_mult = 1.3
    atr_stop_mult = 1.5
    atr_trail_mult = 2.0
    risk_pct = 0.01
    min_squeeze_bars = 5

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        # Bollinger Bands
        self.bb_upper, self.bb_mid, self.bb_lower = self.I(
            talib.BBANDS, close, timeperiod=self.bb_period,
            nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0
        )
        self.bb_width = self.I(lambda u, l: u - l, self.bb_upper, self.bb_lower)

        # ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        # Volume MA - cast volume to double for talib
        volume_arr = np.asarray(volume, dtype=np.float64)
        self.vol_ma = self.I(talib.SMA, volume_arr, timeperiod=self.vol_ma_period)

        # Swing highs/lows
        self.hh = self.I(talib.MAX, high, timeperiod=self.lookback_high)
        self.ll = self.I(talib.MIN, low, timeperiod=self.lookback_high)

        # Compression: current BB width in lowest pct of last N bars
        def compression(bbw):
            out = np.zeros(len(bbw))
            for i in range(len(bbw)):
                if i < self.compression_lookback:
                    continue
                window = bbw[i - self.compression_lookback:i]
                window = window[~np.isnan(window)]
                if len(window) < 10:
                    continue
                thresh = np.percentile(window, self.compression_pct * 100)
                out[i] = 1.0 if bbw[i] <= thresh else 0.0
            return out

        self.compressed = self.I(compression, self.bb_width)

        # ATR rising check: current ATR > previous ATR
        def atr_rising(atr):
            out = np.zeros(len(atr))
            for i in range(1, len(atr)):
                if not np.isnan(atr[i]) and not np.isnan(atr[i-1]):
                    out[i] = 1.0 if atr[i] > atr[i-1] else 0.0
            return out

        self.atr_up = self.I(atr_rising, self.atr)

        # Track squeeze bar count
        self.squeeze_count = 0
        self.trail_stop = None

    def next(self):
        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        openp = self.data.Open[-1]

        # Update squeeze counter
        if self.compressed[-1] == 1.0:
            self.squeeze_count += 1
        else:
            self.squeeze_count = 0

        # Manage existing position with trailing stop
        if self.position:
            if self.position.is_long and self.trail_stop is not None:
                new_trail = self.data.Close[-1] - self.atr_trail_mult * self.atr[-1]
                if new_trail > self.trail_stop:
                    self.trail_stop = new_trail
                if price <= self.trail_stop:
                    print(f"🌙 Long trail exit @ {price:.2f} | trail={self.trail_stop:.2f}")
                    self.position.close()
                    self.trail_stop = None
            elif self.position.is_short and self.trail_stop is not None:
                new_trail = self.data.Close[-1] + self.atr_trail_mult * self.atr[-1]
                if new_trail < self.trail_stop:
                    self.trail_stop = new_trail
                if price >= self.trail_stop:
                    print(f"🌙 Short trail exit @ {price:.2f} | trail={self.trail_stop:.2f}")
                    self.position.close()
                    self.trail_stop = None
            return

        # Need enough data
        if len(self.data) < self.compression_lookback + 5:
            return

        # Candle body strength
        rng = high - low
        if rng <= 0:
            return
        body = abs(price - openp)
        body_pct = body / rng

        # Volume confirmation
        vol_ok = self.data.Volume[-1] > self.vol_ma[-1] * self.volume_mult

        # Require squeeze lasted min bars
        squeeze_ok = self.squeeze_count >= self.min_squeeze_bars

        # ATR expanding
        atr_expanding = self.atr_up[-1] == 1.0

        # Breakout levels
        upper_trigger = self.hh[-2]  # prior N-bar high (excluding current)
        lower_trigger = self.ll[-2]

        if np.isnan(upper_trigger) or np.isnan(lower_trigger):
            return

        # Long entry
        if (squeeze_ok and atr_expanding and body_pct > 0.6 and vol_ok
                and price > upper_trigger and price > self.bb_upper[-1]):
            stop = min(low, price - self.atr_stop_mult * self.atr[-1])
            risk = price - stop
            if risk <= 0:
                return
            size = int(round(1000000 / price))
            if size < 1:
                return
            print(f"🚀🌙 LONG BREAKOUT @ {price:.2f} | trigger={upper_trigger:.2f} | body={body_pct:.2f} | vol_ok={vol_ok} | squeeze={self.squeeze_count}")
            self.buy(size=size)
            self.trail_stop = stop

        # Short entry
        elif (squeeze_ok and atr_expanding and body_pct > 0.6 and vol_ok
                and price < lower_trigger and price < self.bb_lower[-1]):
            stop = max(high, price + self.atr_stop_mult * self.atr[-1])
            risk = stop - price
            if risk <= 0:
                return
            size = int(round(1000000 / price))
            if size < 1:
                return
            print(f"🔻🌙 SHORT BREAKOUT @ {price:.2f} | trigger={lower_trigger:.2f} | body={body_pct:.2f} | vol_ok={vol_ok} | squeeze={self.squeeze_count}")
            self.sell(size=size)
            self.trail_stop = stop


print("🌙 Initializing KineticVolatility backtest... ✨")
bt = Backtest(data, KineticVolatility, cash=1_000_000, commission=0.001, exclusive_orders=True)
stats = bt.run()
print("\n" + "="*60)
print("🌙✨ KINETIC VOLATILITY BACKTEST RESULTS ✨🌙")
print("="*60)
print(stats)
print("\n" + "="*60)
print("🌙 STRATEGY DETAILS 🌙")
print("="*60)
print(stats._strategy)