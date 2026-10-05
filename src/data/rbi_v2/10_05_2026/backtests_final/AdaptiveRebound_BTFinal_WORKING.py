import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

print("🌙✨ AdaptiveRebound Strategy - Moon Dev Backtest Engine ✨🌙")

# Load data
data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Rename columns properly
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

# Ensure datetime handling
if 'datetime' in data.columns:
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')
    data.index.name = 'Datetime'

data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

print(f"🚀 Data loaded: {len(data)} bars, columns: {list(data.columns)}")


class AdaptiveRebound(Strategy):
    lookback = 252          # rolling window for HH
    entry_pct = 0.75        # entry threshold as pct of HH
    tp_pct = 1.25           # take-profit as pct of HH
    stop_pct = 0.50         # hard stop as pct of entry
    bb_period = 20
    bb_std = 2.0
    rsi_period = 14
    time_stop = 60
    base_size = 0.5         # fraction of equity
    target_bbw = 0.05       # reference BBW for sizing
    max_size = 0.95         # max fraction of equity

    def init(self):
        print("🌙 Initializing AdaptiveRebound indicators...")
        self.hh = self.I(talib.MAX, self.data.High, timeperiod=self.lookback)

        # BBANDS returns 3 arrays - wrap with a helper to split outputs
        self.bb_upper = self.I(lambda c: talib.BBANDS(c, timeperiod=self.bb_period, nbdevup=self.bb_std, nbdevdn=self.bb_std)[0], self.data.Close)
        self.bb_middle = self.I(lambda c: talib.BBANDS(c, timeperiod=self.bb_period, nbdevup=self.bb_std, nbdevdn=self.bb_std)[1], self.data.Close)
        self.bb_lower = self.I(lambda c: talib.BBANDS(c, timeperiod=self.bb_period, nbdevup=self.bb_std, nbdevdn=self.bb_std)[2], self.data.Close)

        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=self.rsi_period)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14)
        print("✨ Indicators ready: HH, BBANDS, RSI, ATR")

    def next(self):
        price = self.data.Close[-1]

        # Skip if indicators not ready
        if np.isnan(self.hh[-1]) or np.isnan(self.bb_middle[-1]) or np.isnan(self.rsi[-1]):
            return

        hh = self.hh[-1]
        bbw = (self.bb_upper[-1] - self.bb_lower[-1]) / self.bb_middle[-1] if self.bb_middle[-1] > 0 else 0

        entry_level = self.entry_pct * hh
        tp_level = self.tp_pct * hh

        # ================== ENTRY LOGIC ==================
        if not self.position:
            bullish_confirm = self.data.Close[-1] > self.data.High[-2] if len(self.data) > 1 else False
            oversold = self.rsi[-1] < 40 and self.rsi[-1] > self.rsi[-2]  # turning up from low

            if price <= entry_level and (bullish_confirm or oversold):
                # Position sizing inversely proportional to BBW - use fraction of equity
                if bbw > 0:
                    size_raw = self.base_size * (self.target_bbw / bbw)
                else:
                    size_raw = self.base_size
                size = min(max(size_raw, 0.01), self.max_size)

                # Stop: max of 50% entry or volatility-adjusted
                entry_price = price
                vol_stop = entry_price - 1.5 * bbw * entry_price
                hard_stop = self.stop_pct * entry_price
                effective_stop = max(hard_stop, vol_stop)

                # Cap stop below entry
                if effective_stop >= entry_price:
                    effective_stop = entry_price * 0.95

                print(f"🌙🚀 ENTRY | Price={price:.2f} | HH={hh:.2f} | EntryLvl={entry_level:.2f} | "
                      f"BBW={bbw:.4f} | Size={size:.4f} | Stop={effective_stop:.2f} | TP={tp_level:.2f}")

                self.buy(size=size)
                self._entry_price = entry_price
                self._stop_price = effective_stop
                self._tp_price = tp_level
                self._bars_held = 0

        # ================== EXIT LOGIC ==================
        else:
            self._bars_held += 1
            stop_price = self._stop_price
            tp_price = self._tp_price

            # Take profit
            if price >= tp_price:
                print(f"🌙💰 TAKE PROFIT | Price={price:.2f} >= TP={tp_price:.2f}")
                self.position.close()
                return

            # Hard stop
            if price <= stop_price:
                print(f"🌙🛑 STOP LOSS | Price={price:.2f} <= Stop={stop_price:.2f}")
                self.position.close()
                return

            # Time stop
            if self._bars_held >= self.time_stop:
                print(f"🌙⏰ TIME STOP | Held {self._bars_held} bars | Price={price:.2f}")
                self.position.close()
                return


print("🌙 Setting up backtest...")
bt = Backtest(
    data,
    AdaptiveRebound,
    cash=1_000_000,
    commission=0.002,
    exclusive_orders=True
)

print("🚀 Running backtest...")
stats = bt.run()
print(stats)
print(stats._strategy)
print("✨🌙 Moon Dev Backtest Complete 🌙✨")