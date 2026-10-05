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
    'datetime': 'Date',
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

data['Date'] = pd.to_datetime(data['Date'])
data = data.set_index('Date')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

print("🌙✨ Moon Dev VortexSqueeze Backtest Initializing... 🚀")
print(f"📊 Data loaded: {len(data)} bars from {data.index[0]} to {data.index[-1]}")


class VortexSqueeze(Strategy):
    vi_period = 14
    vol_sma_period = 20
    bb_period = 20
    bb_std = 2.0
    bbw_lookback = 100
    bbw_percentile = 20
    atr_period = 14
    atr_mult = 1.5
    risk_pct = 0.01
    time_stop_bars = 20

    def init(self):
        high = self.data.High
        low = self.data.Low
        close = self.data.Close
        volume = self.data.Volume

        # Vortex Indicator
        def vortex_plus(h, l, c, period):
            vm_plus = np.abs(h - np.roll(l, 1))
            tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
            vm_plus_s = pd.Series(vm_plus).rolling(period).sum()
            tr_s = pd.Series(tr).rolling(period).sum()
            return (vm_plus_s / tr_s).values

        def vortex_minus(h, l, c, period):
            vm_minus = np.abs(l - np.roll(h, 1))
            tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
            vm_minus_s = pd.Series(vm_minus).rolling(period).sum()
            tr_s = pd.Series(tr).rolling(period).sum()
            return (vm_minus_s / tr_s).values

        self.vi_plus = self.I(vortex_plus, high, low, close, self.vi_period, name='VI+')
        self.vi_minus = self.I(vortex_minus, high, low, close, self.vi_period, name='VI-')

        # Volume SMA
        self.vol_sma = self.I(talib.SMA, volume, timeperiod=self.vol_sma_period, name='VolSMA')

        # Bollinger Bands
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS, close, timeperiod=self.bb_period, nbdevup=self.bb_std,
            nbdevdn=self.bb_std, matype=0, name='BB'
        )

        # Bollinger Band Width
        def bbw_func(upper, middle, lower):
            return (upper - lower) / middle

        self.bbw = self.I(bbw_func, self.bb_upper, self.bb_middle, self.bb_lower, name='BBW')

        # BBW rolling percentile (20th percentile over last 100 bars)
        def bbw_pct(bw):
            s = pd.Series(bw)
            return s.rolling(self.bbw_lookback).apply(
                lambda x: np.nanpercentile(x, self.bbw_percentile), raw=True
            ).values

        self.bbw_threshold = self.I(bbw_pct, self.bbw, name='BBW_P20')

        # ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period, name='ATR')

        # SMA 20 for bias
        self.sma20 = self.I(talib.SMA, close, timeperiod=20, name='SMA20')

        # State tracking
        self.entry_bar = None
        self.entry_price = None
        self.stop_price = None
        self.middle_reached = False

    def next(self):
        price = self.data.Close[-1]

        # Manage open position
        if self.position:
            bars_held = len(self.data) - 1 - self.entry_bar

            # Move stop to breakeven once middle band reached
            if not self.middle_reached and price >= self.bb_middle[-1]:
                self.middle_reached = True
                self.stop_price = max(self.stop_price, self.entry_price)
                print(f"🌙✨ Moon Dev: Middle band reached! Stop moved to breakeven @ {self.stop_price:.2f}")

            # Trail below middle band
            if self.middle_reached:
                new_stop = max(self.stop_price, self.bb_middle[-1] * 0.999)
                if new_stop > self.stop_price:
                    self.stop_price = new_stop

            # Update stop loss order
            if self.stop_price:
                self.orders.set_stop(self.stop_price)

            # Primary exit: price touches upper BB
            if price >= self.bb_upper[-1]:
                print(f"🚀 Moon Dev EXIT: Upper BB touch @ {price:.2f} — Take Profit!")
                self.position.close()
                return

            # Secondary exit: VI+ crosses below VI- (bearish crossover)
            if self.vi_plus[-2] > self.vi_minus[-2] and self.vi_plus[-1] < self.vi_minus[-1]:
                print(f"⚠️ Moon Dev EXIT: VI+ crossed below VI- — trend invalidation")
                self.position.close()
                return

            # Time stop
            if bars_held >= self.time_stop_bars:
                print(f"⏰ Moon Dev EXIT: Time stop after {bars_held} bars")
                self.position.close()
                return

            return

        # Entry logic
        if len(self.data) < self.bbw_lookback + 2:
            return

        # Check for valid values
        if (np.isnan(self.vi_plus[-1]) or np.isnan(self.vi_minus[-1]) or
                np.isnan(self.vol_sma[-1]) or np.isnan(self.bbw_threshold[-1]) or
                np.isnan(self.bb_upper[-1]) or np.isnan(self.atr[-1])):
            return

        # VI+ crosses above VI- (bullish crossover)
        vi_cross_up = (self.vi_plus[-2] <= self.vi_minus[-2] and
                       self.vi_plus[-1] > self.vi_minus[-1])

        # Volume surge
        vol_surge = self.data.Volume[-1] >= 1.5 * self.vol_sma[-1]

        # BBW squeeze
        squeeze = self.bbw[-1] <= self.bbw_threshold[-1]

        # Bias: close > SMA20
        bias_ok = price > self.sma20[-1]

        if vi_cross_up and vol_surge and squeeze and bias_ok:
            print(f"🌙✨ Moon Dev SIGNAL: VortexSqueeze Long! VI+={self.vi_plus[-1]:.3f} VI-={self.vi_minus[-1]:.3f} "
                  f"Vol={self.data.Volume[-1]:.2f} VolSMA={self.vol_sma[-1]:.2f} "
                  f"BBW={self.bbw[-1]:.4f} Thresh={self.bbw_threshold[-1]:.4f}")

            # Stop loss: tighter of lower BB or 1.5*ATR
            lower_bb_stop = self.bb_lower[-1]
            atr_stop = price - self.atr_mult * self.atr[-1]
            stop = max(lower_bb_stop, atr_stop)

            risk = price - stop
            if risk <= 0:
                return

            equity = self.equity
            risk_amount = equity * self.risk_pct
            position_size = int(round(risk_amount / risk))

            if position_size < 1:
                position_size = 1

            print(f"🎯 Moon Dev ENTRY: Long {position_size} units @ {price:.2f} | Stop: {stop:.2f} | Risk: {risk:.2f}")

            self.buy(size=position_size)
            self.entry_bar = len(self.data) - 1
            self.entry_price = price
            self.stop_price = stop
            self.middle_reached = False

            # Place stop loss order
            self.orders.set_stop(stop)


bt = Backtest(data, VortexSqueeze, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)