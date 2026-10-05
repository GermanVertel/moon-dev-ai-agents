import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

print("🌙✨ Moon Dev's VolumetricSurge Backtest Initializing... 🚀")

data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"

data = pd.read_csv(data_path)
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'datetime': 'Datetime',
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})
data = data.set_index(pd.to_datetime(data['Datetime']))
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
data = data.astype(np.float64)
print(f"🌙 Data loaded: {len(data)} bars from {data.index[0]} to {data.index[-1]} ✨")


class VolumetricSurge(Strategy):
    bb_period = 20
    bb_std = 2.0
    vol_period = 20
    vol_mult = 2.0
    rsi_period = 14
    rsi_entry_max = 70
    rsi_exit_level = 70
    atr_period = 14
    atr_mult = 2.0
    risk_pct = 0.02
    max_holding = 15

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        def bb_upper_func(close):
            u, m, l = talib.BBANDS(close, timeperiod=self.bb_period,
                                   nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0)
            return u

        def bb_mid_func(close):
            u, m, l = talib.BBANDS(close, timeperiod=self.bb_period,
                                   nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0)
            return m

        def bb_lower_func(close):
            u, m, l = talib.BBANDS(close, timeperiod=self.bb_period,
                                   nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0)
            return l

        self.bb_upper = self.I(bb_upper_func, close)
        self.bb_mid = self.I(bb_mid_func, close)
        self.bb_lower = self.I(bb_lower_func, close)
        self.vol_sma = self.I(talib.SMA, volume.astype(np.float64), timeperiod=self.vol_period)
        self.rsi = self.I(talib.RSI, close, timeperiod=self.rsi_period)
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        self.entry_price = None
        self.entry_low = None
        self.entry_bar = None
        self.stop_price = None
        self.partial_taken = False
        self.rsi_was_above_70 = False

        print("🌙 Indicators ready: BB, Vol SMA, RSI, ATR ✨")

    def next(self):
        if len(self.data) < self.bb_period + 2:
            return

        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        vol = self.data.Volume[-1]

        upper = self.bb_upper[-1]
        mid = self.bb_mid[-1]
        vol_avg = self.vol_sma[-1]
        rsi = self.rsi[-1]
        atr = self.atr[-1]

        if np.isnan(upper) or np.isnan(vol_avg) or np.isnan(rsi) or np.isnan(atr):
            return

        if self.position:
            self._manage_position(price, mid, rsi, atr)
            return

        breakout = price > upper
        vol_confirm = vol >= self.vol_mult * vol_avg
        rsi_ok = rsi < self.rsi_entry_max

        if breakout and vol_confirm and rsi_ok:
            entry = price
            stop_atr = entry - self.atr_mult * atr
            stop_low = low
            stop = max(stop_atr, stop_low)
            risk_per_unit = entry - stop
            if risk_per_unit <= 0:
                print("🌙 ⚠️ Invalid risk, skipping entry")
                return

            risk_amount = self.equity * self.risk_pct
            pos_size = int(round(risk_amount / risk_per_unit))
            if pos_size <= 0:
                pos_size = 1

            # Cap position size to available cash
            max_size = int(self.equity // entry)
            if max_size < 1:
                max_size = 1
            if pos_size > max_size:
                pos_size = max_size

            self.buy(size=pos_size)
            self.entry_price = entry
            self.entry_low = low
            self.entry_bar = len(self.data)
            self.stop_price = stop
            self.partial_taken = False
            self.rsi_was_above_70 = False

            print(f"🌙 🚀 ENTRY VolumetricSurge! Price={price:.2f} Upper={upper:.2f} "
                  f"Vol={vol:.2f} vs Avg={vol_avg:.2f} RSI={rsi:.2f} "
                  f"Stop={stop:.2f} Size={pos_size} ✨")

    def _manage_position(self, price, mid, rsi, atr):
        if self.entry_bar is None:
            return

        bars_held = len(self.data) - self.entry_bar

        if rsi > 80:
            self.rsi_was_above_70 = True

        if rsi > 70:
            self.rsi_was_above_70 = True

        if self.rsi_was_above_70:
            new_stop = mid
            if self.stop_price is None or new_stop > self.stop_price:
                self.stop_price = new_stop
                print(f"🌙 📈 Trailing stop raised to mid-band {new_stop:.2f}")

        if (not self.partial_taken) and (self.entry_price is not None):
            if price >= self.entry_price + 2 * atr:
                try:
                    current_size = self.position.size
                    half_size = max(1, int(current_size // 2))
                    if half_size >= current_size:
                        half_size = max(1, int(current_size) - 1)
                    if half_size > 0 and current_size > 1:
                        self.sell(size=half_size)
                        self.partial_taken = True
                        print(f"🌙 💰 Partial profit 50% at +2xATR price={price:.2f}")
                except Exception as e:
                    print(f"🌙 ⚠️ Partial exit issue: {e}")

        if self.stop_price is not None and price <= self.stop_price:
            self.position.close()
            print(f"🌙 🛑 STOP-LOSS hit at {price:.2f} (stop={self.stop_price:.2f})")
            self._reset()
            return

        if self.rsi_was_above_70 and rsi < self.rsi_exit_level:
            self.position.close()
            print(f"🌙 ✅ RSI exit: RSI crossed below 70 at {rsi:.2f}, price={price:.2f}")
            self._reset()
            return

        if (not self.rsi_was_above_70) and price < mid:
            self.position.close()
            print(f"🌙 ✅ Mid-band exit: price {price:.2f} < mid {mid:.2f}")
            self._reset()
            return

        if bars_held >= self.max_holding:
            self.position.close()
            print(f"🌙 ⏰ Max holding period ({self.max_holding} bars) reached, exit at {price:.2f}")
            self._reset()
            return

    def _reset(self):
        self.entry_price = None
        self.entry_low = None
        self.entry_bar = None
        self.stop_price = None
        self.partial_taken = False
        self.rsi_was_above_70 = False


bt = Backtest(data, VolumetricSurge, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)