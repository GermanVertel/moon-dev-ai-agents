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
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

# Set datetime index
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')

# Ensure numeric types (fixes talib "input array type is not double")
for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    data[col] = pd.to_numeric(data[col], errors='coerce').astype('float64')

data = data.dropna()

print("🌙✨ Moon Dev Backtest Initialized ✨🌙")
print(f"📊 Data loaded: {len(data)} bars")
print(f"📅 Range: {data.index[0]} to {data.index[-1]}")


class SqueezeSurge(Strategy):
    bb_period = 20
    bb_std = 2.0
    kc_ema_period = 20
    kc_atr_period = 20
    kc_atr_mult = 1.5
    vol_sma_period = 20
    vol_surge_mult = 1.5
    adx_period = 14
    adx_exit = 25
    adx_entry = 20
    vol_band_period = 480  # 5 days of 15m bars (5*24*4)
    atr_period = 14
    atr_stop_mult = 1.5
    time_stop_bars = 15
    risk_pct = 0.01
    slippage_atr = 0.1

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

        # Keltner Channels
        self.kc_ema = self.I(talib.EMA, close, timeperiod=self.kc_ema_period)
        self.kc_atr = self.I(talib.ATR, high, low, close, timeperiod=self.kc_atr_period)
        self.kc_upper = self.I(lambda: self.kc_ema + self.kc_atr_mult * self.kc_atr)
        self.kc_lower = self.I(lambda: self.kc_ema - self.kc_atr_mult * self.kc_atr)

        # Volume SMA
        self.vol_sma = self.I(talib.SMA, volume, timeperiod=self.vol_sma_period)

        # ADX and DI
        self.adx = self.I(talib.ADX, high, low, close, timeperiod=self.adx_period)
        self.plus_di = self.I(talib.PLUS_DI, high, low, close, timeperiod=self.adx_period)
        self.minus_di = self.I(talib.MINUS_DI, high, low, close, timeperiod=self.adx_period)

        # ATR for stops
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        # 5-day volatility band
        self.vol_high = self.I(talib.MAX, high, timeperiod=self.vol_band_period)
        self.vol_low = self.I(talib.MIN, low, timeperiod=self.vol_band_period)

        # Squeeze state tracking
        self.squeeze_count = 0
        self.entry_bar = None
        self.entry_price = None
        self.stop_price = None
        self.breakeven_moved = False

    def next(self):
        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        vol = self.data.Volume[-1]

        # Skip if indicators not ready
        if (np.isnan(self.bb_upper[-1]) or np.isnan(self.kc_upper[-1]) or
            np.isnan(self.vol_sma[-1]) or np.isnan(self.adx[-1]) or
            np.isnan(self.vol_high[-1])):
            return

        # Squeeze detection
        squeeze = (self.bb_upper[-1] < self.kc_upper[-1]) and (self.bb_lower[-1] > self.kc_lower[-1])
        if squeeze:
            self.squeeze_count += 1
        else:
            self.squeeze_count = 0

        # Volume surge
        vol_surge = vol >= self.vol_surge_mult * self.vol_sma[-1]

        # Manage open position
        if self.position:
            bars_held = len(self.data) - self.entry_bar if self.entry_bar else 0

            # Move stop to breakeven
            if not self.breakeven_moved and self.entry_price is not None:
                gain = price - self.entry_price if self.position.is_long else self.entry_price - price
                if gain >= self.atr[-1]:
                    self.stop_price = self.entry_price
                    self.breakeven_moved = True
                    print(f"🌙 Breakeven stop moved! Price: {price:.2f}, Entry: {self.entry_price:.2f}")

            # Primary exit: ADX < 25
            if self.adx[-1] < self.adx_exit:
                print(f"🚀 EXIT (ADX decay {self.adx[-1]:.1f} < 25) at {price:.2f}")
                self.position.close()
                self.entry_bar = None
                return

            # Secondary exit: volatility band breach
            if self.position.is_long and price < self.vol_low[-1]:
                print(f"🚀 EXIT (5-day low breach {self.vol_low[-1]:.2f}) at {price:.2f}")
                self.position.close()
                self.entry_bar = None
                return
            if self.position.is_short and price > self.vol_high[-1]:
                print(f"🚀 EXIT (5-day high breach {self.vol_high[-1]:.2f}) at {price:.2f}")
                self.position.close()
                self.entry_bar = None
                return

            # Stop loss check
            if self.stop_price is not None:
                if self.position.is_long and price <= self.stop_price:
                    print(f"🛑 STOP HIT (long) at {price:.2f}")
                    self.position.close()
                    self.entry_bar = None
                    return
                if self.position.is_short and price >= self.stop_price:
                    print(f"🛑 STOP HIT (short) at {price:.2f}")
                    self.position.close()
                    self.entry_bar = None
                    return

            # Time stop
            if bars_held >= self.time_stop_bars:
                print(f"⏰ TIME STOP ({bars_held} bars) at {price:.2f}")
                self.position.close()
                self.entry_bar = None
                return

            return

        # Entry logic — need squeeze on prior bar
        if self.squeeze_count >= 3 and vol_surge:
            slip = self.slippage_atr * self.atr[-1]

            # Long entry
            if price > self.bb_upper[-1] + slip:
                if self.adx[-1] > self.adx_entry and self.plus_di[-1] > self.minus_di[-1]:
                    stop = price - self.atr_stop_mult * self.atr[-1]
                    risk = price - stop
                    if risk > 0:
                        size = int(round((self.equity * self.risk_pct) / risk))
                        if size > 0:
                            print(f"🌙✨ LONG SIGNAL! Price: {price:.2f}, BB_upper: {self.bb_upper[-1]:.2f}, ADX: {self.adx[-1]:.1f}, Size: {size}")
                            self.buy(size=size)
                            self.entry_bar = len(self.data)
                            self.entry_price = price
                            self.stop_price = stop
                            self.breakeven_moved = False

            # Short entry
            elif price < self.bb_lower[-1] - slip:
                if self.adx[-1] > self.adx_entry and self.minus_di[-1] > self.plus_di[-1]:
                    stop = price + self.atr_stop_mult * self.atr[-1]
                    risk = stop - price
                    if risk > 0:
                        size = int(round((self.equity * self.risk_pct) / risk))
                        if size > 0:
                            print(f"🌙✨ SHORT SIGNAL! Price: {price:.2f}, BB_lower: {self.bb_lower[-1]:.2f}, ADX: {self.adx[-1]:.1f}, Size: {size}")
                            self.sell(size=size)
                            self.entry_bar = len(self.data)
                            self.entry_price = price
                            self.stop_price = stop
                            self.breakeven_moved = False


bt = Backtest(data, SqueezeSurge, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)