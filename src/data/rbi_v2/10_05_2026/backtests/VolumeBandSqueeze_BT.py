import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Load and clean data
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
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

print("🌙 Moon Dev: Data loaded and cleaned! Shape:", data.shape, "✨")
print("🚀 First few rows:\n", data.head())


class VolumeBandSqueeze(Strategy):
    # Bollinger Bands
    bb_period = 20
    bb_std = 2.0
    # BandWidth squeeze lookback
    bw_lookback = 50
    bw_percentile = 0.20
    # Chaikin Oscillator
    co_fast = 3
    co_slow = 10
    # ATR
    atr_period = 14
    atr_mult = 1.5
    # Volume
    vol_ma_period = 20
    vol_mult = 1.5
    # Risk
    risk_pct = 0.01
    rr_min = 1.5
    max_trades_per_day = 5

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        # Bollinger Bands
        self.bb_mid = self.I(talib.SMA, close, timeperiod=self.bb_period)
        self.bb_std = self.I(talib.STDDEV, close, timeperiod=self.bb_period, nbdev=1)
        self.bb_upper = self.I(lambda m, s: m + self.bb_std * s, self.bb_mid, self.bb_std)
        self.bb_lower = self.I(lambda m, s: m - self.bb_std * s, self.bb_mid, self.bb_std)

        # BandWidth
        self.bw = self.I(lambda u, l, m: (u - l) / m, self.bb_upper, self.bb_lower, self.bb_mid)

        # Accumulation/Distribution Line
        self.adl = self.I(talib.AD, high, low, close, volume)

        # Chaikin Oscillator = EMA(3) of ADL - EMA(10) of ADL
        self.co = self.I(lambda adl: talib.EMA(adl, timeperiod=self.co_fast) - talib.EMA(adl, timeperiod=self.co_slow), self.adl)

        # ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        # Volume MA
        self.vol_ma = self.I(talib.SMA, volume, timeperiod=self.vol_ma_period)

        # Trackers
        self.trades_today = 0
        self.current_day = None
        self.entry_bar = None
        self.entry_price = None
        self.stop_price = None
        self.target_price = None

        print("🌙 Moon Dev: All indicators initialized! ✨")

    def _reset_day_if_needed(self):
        day = self.data.index[-1].date()
        if self.current_day != day:
            self.current_day = day
            self.trades_today = 0

    def next(self):
        self._reset_day_if_needed()

        if len(self.data) < max(self.bw_lookback, self.bb_period, self.atr_period, self.vol_ma_period) + 2:
            return

        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        volume = self.data.Volume[-1]

        bw = self.bw[-1]
        bw_window = self.bw[-self.bw_lookback:]
        bw_threshold = np.percentile(bw_window, self.bw_percentile * 100)
        squeeze = bw <= bw_threshold

        co = self.co[-1]
        co_prev = self.co[-2]
        co_prev2 = self.co[-3] if len(self.co) > 3 else co_prev

        vol_ok = volume > self.vol_mult * self.vol_ma[-1]

        # Position management
        if self.position:
            # Momentum exit: CO crosses back through zero against position
            if self.position.is_long:
                # Exit if CO turns negative
                if co < 0 and co_prev >= 0:
                    print(f"🌙 Moon Dev: CO crossed below zero, exiting LONG at {price} ✨")
                    self.position.close()
                    return
                # Stop loss
                if low <= self.stop_price:
                    print(f"🌙 Moon Dev: Stop loss hit on LONG at {price} 🛑")
                    self.position.close()
                    return
                # Take profit
                if high >= self.target_price:
                    print(f"🌙 Moon Dev: Take profit hit on LONG at {price} 🎯")
                    self.position.close()
                    return
                # Re-entry stop: close back inside band within 2 bars
                if len(self.data) - self.entry_bar <= 2 and price < self.bb_upper[-1]:
                    print(f"🌙 Moon Dev: Re-entry stop - price back inside band, exiting LONG 🚫")
                    self.position.close()
                    return
            elif self.position.is_short:
                if co > 0 and co_prev <= 0:
                    print(f"🌙 Moon Dev: CO crossed above zero, exiting SHORT at {price} ✨")
                    self.position.close()
                    return
                if high >= self.stop_price:
                    print(f"🌙 Moon Dev: Stop loss hit on SHORT at {price} 🛑")
                    self.position.close()
                    return
                if low <= self.target_price:
                    print(f"🌙 Moon Dev: Take profit hit on SHORT at {price} 🎯")
                    self.position.close()
                    return
                if len(self.data) - self.entry_bar <= 2 and price > self.bb_lower[-1]:
                    print(f"🌙 Moon Dev: Re-entry stop - price back inside band, exiting SHORT 🚫")
                    self.position.close()
                    return
            return

        # Trade limits
        if self.trades_today >= self.max_trades_per_day:
            return

        # Volatility filter: skip if BandWidth at top 20% (already expanded)
        if bw >= np.percentile(bw_window, 80):
            return

        if not squeeze:
            return

        # Long entry
        long_signal = (
            price > self.bb_upper[-1] and
            vol_ok and
            ((co > 0 and co > co_prev) or (co > 0 and co_prev <= 0 and co_prev2 <= 0))
        )

        # Short entry
        short_signal = (
            price < self.bb_lower[-1] and
            vol_ok and
            ((co < 0 and co < co_prev) or (co < 0 and co_prev >= 0 and co_prev2 >= 0))
        )

        if long_signal:
            atr = self.atr[-1]
            stop = min(self.bb_lower[-1], price - self.atr_mult * atr)
            risk = price - stop
            if risk <= 0:
                return
            target = price + 2 * bw * self.bb_mid[-1] if self.bb_mid[-1] else price + 2 * risk
            reward = target - price
            if reward < self.rr_min * risk:
                return

            size = int(round((self.equity * self.risk_pct) / risk))
            if size <= 0:
                return
            size = min(size, int(self.equity / price))

            self.buy(size=size)
            self.entry_bar = len(self.data)
            self.entry_price = price
            self.stop_price = stop
            self.target_price = target
            self.trades_today += 1
            print(f"🚀 Moon Dev: LONG entry at {price}, stop={stop:.2f}, target={target:.2f}, size={size} 🌙")

        elif short_signal:
            atr = self.atr[-1]
            stop = max(self.bb_upper[-1], price + self.atr_mult * atr)
            risk = stop - price
            if risk <= 0:
                return
            target = price - 2 * bw * self.bb_mid[-1] if self.bb_mid[-1] else price - 2 * risk
            reward = price - target
            if reward < self.rr_min * risk:
                return

            size = int(round((self.equity * self.risk_pct) / risk))
            if size <= 0:
                return
            size = min(size, int(self.equity / price))

            self.sell(size=size)
            self.entry_bar = len(self.data)
            self.entry_price = price
            self.stop_price = stop
            self.target_price = target
            self.trades_today += 1
            print(f"🚀 Moon Dev: SHORT entry at {price}, stop={stop:.2f}, target={target:.2f}, size={size} 🌙")


bt = Backtest(data, VolumeBandSqueeze, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)