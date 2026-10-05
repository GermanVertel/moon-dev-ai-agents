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

print("🌙✨ Moon Dev CompressionSurge Backtest Loading ✨🌙")
print(f"🚀 Data loaded: {len(data)} bars")
print(f"📊 Columns: {list(data.columns)}")


class CompressionSurge(Strategy):
    bb_period = 20
    bb_std = 2.0
    bbw_lookback = 30
    atr_period = 14
    atr_mult = 2.0
    vol_period = 20
    time_stop_bars = 10
    max_dd_guard = 0.15

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        # Bollinger Bands
        self.upper, self.middle, self.lower = self.I(
            talib.BBANDS, close, timeperiod=self.bb_period,
            nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0
        )

        # Bollinger Band Width
        def bbw_calc(u, m, l):
            return (u - l) / m
        self.bbw = self.I(bbw_calc, self.upper, self.middle, self.lower)

        # ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        # Volume average - ensure float64 for talib
        volume_arr = np.asarray(volume, dtype=np.float64)
        self.vol_avg = self.I(talib.SMA, volume_arr, timeperiod=self.vol_period)

        # BBW lowest in lookback
        self.bbw_min = self.I(talib.MIN, self.bbw, timeperiod=self.bbw_lookback)

        # State
        self.trail_stop = None
        self.entry_bar = None
        self.highest_close = None
        self.peak_equity = None

        print("🌙 CompressionSurge indicators initialized 🚀")

    def next(self):
        price = self.data.Close[-1]
        high = self.data.High[-1]

        # Track peak equity for drawdown guard
        equity = self.equity
        if self.peak_equity is None or equity > self.peak_equity:
            self.peak_equity = equity
        dd = (self.peak_equity - equity) / self.peak_equity if self.peak_equity else 0

        if self.position:
            # Update highest close
            if self.highest_close is None or price > self.highest_close:
                self.highest_close = price

            # Update trailing stop (never move down)
            atr_val = self.atr[-1]
            new_stop = self.highest_close - self.atr_mult * atr_val
            if self.trail_stop is None or new_stop > self.trail_stop:
                self.trail_stop = new_stop
                print(f"🌙 Trailing stop updated: {self.trail_stop:.2f} | Price: {price:.2f}")

            # Exit conditions
            bars_held = len(self.data) - 1 - self.entry_bar if self.entry_bar else 0

            # Close below trailing stop
            if price < self.trail_stop:
                print(f"🚀 EXIT: Close {price:.2f} < Trail Stop {self.trail_stop:.2f} 🌙")
                self.position.close()
                self.trail_stop = None
                self.highest_close = None
                self.entry_bar = None
                return

            # Close below middle band (early warning)
            if price < self.middle[-1]:
                print(f"✨ EXIT: Close {price:.2f} < Middle Band {self.middle[-1]:.2f} 🌙")
                self.position.close()
                self.trail_stop = None
                self.highest_close = None
                self.entry_bar = None
                return

            # Time stop
            if bars_held >= self.time_stop_bars:
                if price <= self.trail_stop:
                    print(f"⏰ TIME STOP exit at {price:.2f} 🌙")
                    self.position.close()
                    self.trail_stop = None
                    self.highest_close = None
                    self.entry_bar = None
                    return

        else:
            # Drawdown guard
            if dd > self.max_dd_guard:
                return

            # Entry logic
            if len(self.data) < self.bbw_lookback + 2:
                return

            bbw_now = self.bbw[-1]
            bbw_min_val = self.bbw_min[-1]

            # Squeeze: current BBW is lowest in lookback
            squeeze = bbw_now <= bbw_min_val + 1e-12

            # Breakout: close above upper band
            breakout = price > self.upper[-1]

            # Volume confirmation
            vol_ok = self.data.Volume[-1] > self.vol_avg[-1]

            if squeeze and breakout and vol_ok:
                atr_val = self.atr[-1]
                if atr_val <= 0 or np.isnan(atr_val):
                    return

                entry_price = price
                initial_stop = entry_price - self.atr_mult * atr_val
                risk_per_unit = entry_price - initial_stop

                if risk_per_unit <= 0:
                    return

                # Position sizing: risk 1% of equity
                risk_amount = self.equity * 0.01
                position_size = risk_amount / risk_per_unit
                position_size = int(round(position_size))

                if position_size < 1:
                    position_size = 1

                # Cap by available cash
                max_affordable = int(self.equity // entry_price)
                if position_size > max_affordable:
                    position_size = max(1, max_affordable)

                print(f"🌙✨ SURGE ENTRY! Price: {price:.2f} | BBW: {bbw_now:.4f} | ATR: {atr_val:.2f}")
                print(f"🚀 Size: {position_size} | Initial Stop: {initial_stop:.2f}")

                self.buy(size=position_size)
                self.trail_stop = initial_stop
                self.entry_bar = len(self.data) - 1
                self.highest_close = entry_price


bt = Backtest(data, CompressionSurge, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)