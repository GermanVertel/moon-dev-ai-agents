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
data.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']

data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')

data.columns = [col.capitalize() for col in data.columns]

print("🌙 Moon Dev Backtest Initializing... Loading VolatilityIgnition Strategy ✨")
print(f"📊 Data loaded: {len(data)} bars")
print(f"🚀 Starting backtest...\n")


class VolatilityIgnition(Strategy):
    atr_period = 14
    atr_ma_period = 20
    atr_contraction_threshold = 0.75
    range_period = 15
    volume_ma_period = 20
    volume_multiplier = 1.5
    rsi_period = 14
    rsi_long_threshold = 60
    rsi_short_threshold = 40
    ema_trend_period = 200
    risk_pct = 0.02
    atr_sl_mult = 1.0
    atr_tp_mult = 2.0
    max_bars_in_trade = 10

    def init(self):
        print("🌙 Initializing indicators... ✨")
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)
        self.atr_ma = self.I(talib.SMA, self.atr, timeperiod=self.atr_ma_period)
        self.range_high = self.I(talib.MAX, high, timeperiod=self.range_period)
        self.range_low = self.I(talib.MIN, low, timeperiod=self.range_period)
        self.vol_ma = self.I(talib.SMA, volume, timeperiod=self.volume_ma_period)
        self.rsi = self.I(talib.RSI, close, timeperiod=self.rsi_period)
        self.ema_trend = self.I(talib.EMA, close, timeperiod=self.ema_trend_period)

        # Initialize trade state attributes
        self.entry_bar = 0
        self.entry_price = 0.0
        self.entry_atr = 0.0
        self.trail_active = False
        self.trail_sl = 0.0

        print("🚀 Indicators ready!\n")

    def next(self):
        if len(self.data) < max(self.ema_trend_period, self.atr_ma_period, self.range_period) + 5:
            return

        price = self.data.Close[-1]
        atr_now = self.atr[-1]
        atr_prev = self.atr[-2]
        atr_ma_now = self.atr_ma[-1]
        vol_now = self.data.Volume[-1]
        vol_ma_now = self.vol_ma[-1]
        rsi_now = self.rsi[-1]
        ema_now = self.ema_trend[-1]
        range_high = self.range_high[-2]  # previous bar's range to avoid self-inclusion
        range_low = self.range_low[-2]

        if np.isnan(atr_now) or np.isnan(atr_ma_now) or np.isnan(vol_ma_now) or np.isnan(rsi_now) or np.isnan(ema_now):
            return

        # Manage existing position
        if self.position:
            self.manage_position(price, atr_now)
            return

        # Volatility contraction
        contraction = atr_now < self.atr_contraction_threshold * atr_ma_now
        vol_expansion = vol_now > self.volume_multiplier * vol_ma_now
        atr_rising = atr_now > atr_prev

        if not (contraction and vol_expansion and atr_rising):
            return

        # Long entry
        if price > range_high and rsi_now > self.rsi_long_threshold and price > ema_now:
            sl = price - self.atr_sl_mult * atr_now
            tp = price + self.atr_tp_mult * atr_now
            risk_per_unit = price - sl
            if risk_per_unit <= 0:
                return
            risk_amount = self.equity * self.risk_pct
            size = int(round(risk_amount / risk_per_unit))
            if size < 1:
                size = 1
            print(f"🌙✨ LONG SIGNAL! Price={price:.2f} RangeHigh={range_high:.2f} RSI={rsi_now:.1f} ATR={atr_now:.2f} Size={size}")
            self.buy(size=size, sl=sl, tp=tp)
            self.entry_bar = len(self.data)
            self.entry_price = price
            self.entry_atr = atr_now
            self.trail_active = False
            self.trail_sl = sl

        # Short entry
        elif price < range_low and rsi_now < self.rsi_short_threshold and price < ema_now:
            sl = price + self.atr_sl_mult * atr_now
            tp = price - self.atr_tp_mult * atr_now
            risk_per_unit = sl - price
            if risk_per_unit <= 0:
                return
            risk_amount = self.equity * self.risk_pct
            size = int(round(risk_amount / risk_per_unit))
            if size < 1:
                size = 1
            print(f"🌙✨ SHORT SIGNAL! Price={price:.2f} RangeLow={range_low:.2f} RSI={rsi_now:.1f} ATR={atr_now:.2f} Size={size}")
            self.sell(size=size, sl=sl, tp=tp)
            self.entry_bar = len(self.data)
            self.entry_price = price
            self.entry_atr = atr_now
            self.trail_active = False
            self.trail_sl = sl

    def manage_position(self, price, atr_now):
        bars_in_trade = len(self.data) - self.entry_bar

        # Time-based exit
        if bars_in_trade >= self.max_bars_in_trade:
            print(f"⏰ Time-based exit after {bars_in_trade} bars at {price:.2f}")
            self.position.close()
            return

        if self.position.is_long:
            # Activate trailing once in profit by 1 ATR
            if not self.trail_active and price >= self.entry_price + self.entry_atr:
                self.trail_active = True
                print(f"🎯 Trailing activated for LONG at {price:.2f}")
            if self.trail_active:
                new_sl = price - self.atr_sl_mult * atr_now
                if new_sl > self.trail_sl:
                    self.trail_sl = new_sl
                    print(f"🔄 Trailing SL updated to {new_sl:.2f}")
        elif self.position.is_short:
            if not self.trail_active and price <= self.entry_price - self.entry_atr:
                self.trail_active = True
                print(f"🎯 Trailing activated for SHORT at {price:.2f}")
            if self.trail_active:
                new_sl = price + self.atr_sl_mult * atr_now
                if new_sl < self.trail_sl:
                    self.trail_sl = new_sl
                    print(f"🔄 Trailing SL updated to {new_sl:.2f}")


bt = Backtest(data, VolatilityIgnition, cash=1_000_000, commission=0.001, exclusive_orders=True)
stats = bt.run()
print(stats)
print(stats._strategy)