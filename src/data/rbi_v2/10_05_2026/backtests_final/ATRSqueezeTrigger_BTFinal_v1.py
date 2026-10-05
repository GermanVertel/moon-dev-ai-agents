import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

print("🌙✨ Moon Dev ATRSqueezeTrigger Backtest Initializing... 🚀")

data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Map to proper case
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

if 'datetime' in data.columns:
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')

print(f"🌙 Data loaded: {len(data)} bars 🚀")
print(f"✨ Columns: {list(data.columns)}")


class ATRSqueezeTrigger(Strategy):
    bb_period = 20
    bb_std = 2.0
    atr_period = 20
    squeeze_mult = 0.20
    squeeze_persist = 2
    trailing_pct = 0.50
    vol_mult = 1.5
    risk_pct = 0.02
    time_stop_bars = 10

    def init(self):
        print("🌙 Moon Dev: Initializing indicators... ✨")
        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        volume = pd.Series(self.data.Volume)

        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS, close, timeperiod=self.bb_period,
            nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0
        )
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)
        self.vol_sma = self.I(talib.SMA, volume, timeperiod=20)

        # Band width
        self.bb_width = self.I(
            lambda u, m, l: (np.array(u) - np.array(l)),
            self.bb_upper, self.bb_middle, self.bb_lower
        )

        self.trade_entry_bar = None
        self.trade_entry_price = None
        self.breakout_range = None
        self.highest_since_entry = None
        self.lowest_since_entry = None
        print("🌙 Moon Dev: Indicators ready! 🚀")

    def next(self):
        i = len(self.data) - 1
        if i < max(self.bb_period, self.atr_period, 20) + self.squeeze_persist + 2:
            return

        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        vol = self.data.Volume[-1]

        atr_val = self.atr[-1]
        bb_w = self.bb_width[-1]
        upper = self.bb_upper[-1]
        lower = self.bb_lower[-1]
        mid = self.bb_middle[-1]

        if np.isnan(atr_val) or np.isnan(bb_w) or np.isnan(upper):
            return

        squeeze_threshold = self.squeeze_mult * atr_val
        squeeze_now = bb_w < squeeze_threshold

        # Check persistence
        squeeze_ok = True
        for k in range(1, self.squeeze_persist + 1):
            if len(self.bb_width) < k + 1:
                squeeze_ok = False
                break
            w_k = self.bb_width[-k - 1]
            a_k = self.atr[-k - 1]
            if np.isnan(w_k) or np.isnan(a_k):
                squeeze_ok = False
                break
            if w_k >= self.squeeze_mult * a_k:
                squeeze_ok = False
                break

        vol_ok = vol > self.vol_mult * self.vol_sma[-1]

        # Manage open position
        if self.position:
            bars_held = i - self.trade_entry_bar if self.trade_entry_bar else 0

            if self.position.is_long:
                self.highest_since_entry = max(self.highest_since_entry, high)
                trail_stop = self.highest_since_entry - self.trailing_pct * self.breakout_range

                # Re-entry into bands exit
                if price < mid:
                    print(f"🌙 Exit LONG: price re-entered bands @ {price:.2f} ✨")
                    self.position.close()
                    self._reset_trade()
                    return

                if low <= trail_stop:
                    print(f"🌙 Trail STOP LONG hit @ {trail_stop:.2f} 🚀")
                    self.position.close()
                    self._reset_trade()
                    return

                if bars_held >= self.time_stop_bars and price < self.trade_entry_price:
                    print(f"🌙 Time stop LONG @ {price:.2f} ⏰")
                    self.position.close()
                    self._reset_trade()
                    return

            elif self.position.is_short:
                self.lowest_since_entry = min(self.lowest_since_entry, low)
                trail_stop = self.lowest_since_entry + self.trailing_pct * self.breakout_range

                if price > mid:
                    print(f"🌙 Exit SHORT: price re-entered bands @ {price:.2f} ✨")
                    self.position.close()
                    self._reset_trade()
                    return

                if high >= trail_stop:
                    print(f"🌙 Trail STOP SHORT hit @ {trail_stop:.2f} 🚀")
                    self.position.close()
                    self._reset_trade()
                    return

                if bars_held >= self.time_stop_bars and price > self.trade_entry_price:
                    print(f"🌙 Time stop SHORT @ {price:.2f} ⏰")
                    self.position.close()
                    self._reset_trade()
                    return
            return

        if not squeeze_ok:
            return

        # Long entry
        if price > upper and price > (high + low) / 2:
            if vol_ok:
                self._enter_long(price, high, low, i)
                return

        # Short entry
        if price < lower and price < (high + low) / 2:
            if vol_ok:
                self._enter_short(price, high, low, i)
                return

    def _enter_long(self, price, high, low, i):
        r = high - low
        if r <= 0:
            return
        stop = price - self.trailing_pct * r
        risk_per_unit = price - stop
        if risk_per_unit <= 0:
            return
        equity = self.equity
        risk_amount = equity * self.risk_pct
        size = int(round(risk_amount / risk_per_unit))
        if size < 1:
            size = 1
        # Cap size to available cash to avoid invalid orders
        max_size = int(self.equity / price)
        if max_size < 1:
            print(f"🌙 Moon Dev: Insufficient cash for LONG entry @ {price:.2f} 💸")
            return
        if size > max_size:
            size = max_size
        print(f"🚀 LONG ENTRY @ {price:.2f} | R={r:.2f} | Stop={stop:.2f} | Size={size} 🌙")
        self.buy(size=size)
        self.trade_entry_bar = i
        self.trade_entry_price = price
        self.breakout_range = r
        self.highest_since_entry = high

    def _enter_short(self, price, high, low, i):
        r = high - low
        if r <= 0:
            return
        stop = price + self.trailing_pct * r
        risk_per_unit = stop - price
        if risk_per_unit <= 0:
            return
        equity = self.equity
        risk_amount = equity * self.risk_pct
        size = int(round(risk_amount / risk_per_unit))
        if size < 1:
            size = 1
        max_size = int(self.equity / price)
        if max_size < 1:
            print(f"🌙 Moon Dev: Insufficient cash for SHORT entry @ {price:.2f} 💸")
            return
        if size > max_size:
            size = max_size
        print(f"🚀 SHORT ENTRY @ {price:.2f} | R={r:.2f} | Stop={stop:.2f} | Size={size} 🌙")
        self.sell(size=size)
        self.trade_entry_bar = i
        self.trade_entry_price = price
        self.breakout_range = r
        self.lowest_since_entry = low

    def _reset_trade(self):
        self.trade_entry_bar = None
        self.trade_entry_price = None
        self.breakout_range = None
        self.highest_since_entry = None
        self.lowest_since_entry = None


bt = Backtest(
    data, ATRSqueezeTrigger,
    cash=1_000_000,
    commission=0.002,
    exclusive_orders=True
)

stats = bt.run()
print(stats)
print(stats._strategy)