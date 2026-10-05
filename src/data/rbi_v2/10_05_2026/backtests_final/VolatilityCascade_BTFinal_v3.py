import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

print("🌙 Moon Dev VolatilityCascade Backtest Loading... ✨")

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
print(f"🌙 Data loaded: {len(data)} bars ✨")


class VolatilityCascade(Strategy):
    # Parameters
    atr_ratio_threshold = 0.5
    atr_ratio_exit = 0.7
    swing_lookback = 20
    sl_buffer_atr = 0.35
    risk_reward = 2.5
    risk_pct = 0.01
    time_stop_bars = 8
    vol_mult = 1.5

    def init(self):
        print("🌙 Initializing VolatilityCascade indicators... ✨")
        high = self.data.High.astype(np.float64)
        low = self.data.Low.astype(np.float64)
        close = self.data.Close.astype(np.float64)
        volume = self.data.Volume.astype(np.float64)

        self.atr = self.I(talib.ATR, high, low, close, timeperiod=14)
        self.atr_long = self.I(talib.ATR, high, low, close, timeperiod=280)
        self.atr_ratio = self.I(lambda a, b: a / np.where(b == 0, np.nan, b), self.atr, self.atr_long)

        self.swing_high = self.I(talib.MAX, high, timeperiod=self.swing_lookback)
        self.swing_low = self.I(talib.MIN, low, timeperiod=self.swing_lookback)

        self.vol_avg = self.I(talib.SMA, volume, timeperiod=20)

        self.rsi = self.I(talib.RSI, close, timeperiod=14)

        # Track trade state
        self.entry_bar = None
        self.entry_price_val = None
        self.stop_price = None
        self.tp_price = None
        self.trade_dir = None

    def next(self):
        if len(self.data) < 300:
            return

        price = self.data.Close[-1]
        atr_val = self.atr[-1]
        ratio = self.atr_ratio[-1]

        if np.isnan(ratio) or np.isnan(atr_val) or atr_val <= 0:
            return

        # Manage open trade
        if self.position:
            bars_held = len(self.data) - self.entry_bar
            # Time stop
            if bars_held >= self.time_stop_bars:
                if self.trade_dir == 'long':
                    reached_1r = self.data.High[-1] >= self.entry_price_val + (self.entry_price_val - self.stop_price)
                else:
                    reached_1r = self.data.Low[-1] <= self.entry_price_val - (self.stop_price - self.entry_price_val)
                if not reached_1r:
                    print(f"⏰ Moon Dev Time Stop hit after {bars_held} bars 🌙")
                    self.position.close()
                    self._reset_state()
                    return

            # Volatility regime flip
            if ratio > self.atr_ratio_exit:
                print(f"⚠️ Volatility rose above {self.atr_ratio_exit}, exiting early 🌙")
                self.position.close()
                self._reset_state()
                return

            # Check SL/TP for open position
            if self.trade_dir == 'long':
                if self.data.Low[-1] <= self.stop_price:
                    print(f"🛑 Moon Dev LONG SL hit at {self.stop_price:.2f} 🌙")
                    self.position.close()
                    self._reset_state()
                elif self.data.High[-1] >= self.tp_price:
                    print(f"🎯 Moon Dev LONG TP hit at {self.tp_price:.2f} ✨")
                    self.position.close()
                    self._reset_state()
            else:
                if self.data.High[-1] >= self.stop_price:
                    print(f"🛑 Moon Dev SHORT SL hit at {self.stop_price:.2f} 🌙")
                    self.position.close()
                    self._reset_state()
                elif self.data.Low[-1] <= self.tp_price:
                    print(f"🎯 Moon Dev SHORT TP hit at {self.tp_price:.2f} ✨")
                    self.position.close()
                    self._reset_state()
            return

        # Entry logic
        if ratio >= self.atr_ratio_threshold:
            return

        # Detect liquidation cascade: sharp wick + high volume
        high_vol = self.data.Volume[-1] > self.vol_avg[-1] * self.vol_mult
        if not high_vol:
            return

        prev_low = self.data.Low[-2]
        prev_high = self.data.High[-2]
        candle_range = self.data.High[-2] - self.data.Low[-2]

        if candle_range <= 0:
            return

        lower_wick = min(self.data.Open[-2], self.data.Close[-2]) - prev_low
        upper_wick = prev_high - max(self.data.Open[-2], self.data.Close[-2])

        long_liq = lower_wick > candle_range * 0.5 and self.data.Close[-2] < self.data.Open[-2]
        short_liq = upper_wick > candle_range * 0.5 and self.data.Close[-2] > self.data.Open[-2]

        curr_bull = self.data.Close[-1] > self.data.Open[-1]
        curr_bear = self.data.Close[-1] < self.data.Open[-1]

        no_new_low = self.data.Low[-1] > prev_low
        no_new_high = self.data.High[-1] < prev_high

        equity = self.equity

        if long_liq and curr_bull and no_new_low:
            entry = price
            stop = self.swing_low[-1] - atr_val * self.sl_buffer_atr
            risk = entry - stop
            if risk <= 0:
                return
            tp = entry + risk * self.risk_reward
            # Use fractional sizing (percentage of equity) to avoid unit-size issues
            risk_amount = equity * self.risk_pct
            size_units = risk_amount / risk
            size_frac = size_units * entry / equity
            if size_frac <= 0 or size_frac >= 1:
                return
            size_frac = max(0.01, min(0.99, size_frac))
            print(f"🚀 MOON DEV LONG SIGNAL | Entry: {entry:.2f} | SL: {stop:.2f} | TP: {tp:.2f} | Size: {size_frac:.4f} 🌙")
            self.buy(size=size_frac)
            self.entry_bar = len(self.data)
            self.entry_price_val = entry
            self.stop_price = stop
            self.tp_price = tp
            self.trade_dir = 'long'

        elif short_liq and curr_bear and no_new_high:
            entry = price
            stop = self.swing_high[-1] + atr_val * self.sl_buffer_atr
            risk = stop - entry
            if risk <= 0:
                return
            tp = entry - risk * self.risk_reward
            risk_amount = equity * self.risk_pct
            size_units = risk_amount / risk
            size_frac = size_units * entry / equity
            if size_frac <= 0 or size_frac >= 1:
                return
            size_frac = max(0.01, min(0.99, size_frac))
            print(f"🔻 MOON DEV SHORT SIGNAL | Entry: {entry:.2f} | SL: {stop:.2f} | TP: {tp:.2f} | Size: {size_frac:.4f} 🌙")
            self.sell(size=size_frac)
            self.entry_bar = len(self.data)
            self.entry_price_val = entry
            self.stop_price = stop
            self.tp_price = tp
            self.trade_dir = 'short'

    def _reset_state(self):
        self.entry_bar = None
        self.entry_price_val = None
        self.stop_price = None
        self.tp_price = None
        self.trade_dir = None


print("🌙 Launching Moon Dev Backtest... 🚀")
bt = Backtest(data, VolatilityCascade, cash=1_000_000, commission=0.001, exclusive_orders=True)
stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙 Moon Dev Backtest Complete! ✨")