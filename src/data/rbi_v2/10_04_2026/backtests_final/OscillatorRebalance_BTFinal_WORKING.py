import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

print("🌙✨ Moon Dev's OscillatorRebalance Backtest Loading... 🚀")

data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"

data = pd.read_csv(data_path)
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
print(f"🌙 Data loaded: {len(data)} candles ✨")


class OscillatorRebalance(Strategy):
    rsi_period = 14
    derisk_threshold = 30
    profit_threshold = 60
    derisk_pct = 0.70
    reentry_leverage_mult = 1.20
    profit_take_pct = 0.50
    max_leverage = 3.0
    base_leverage = 2.0
    stop_lookback = 3
    time_stop_candles = 5

    def init(self):
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=self.rsi_period)
        self.state = 'IDLE'
        self.reentry_price = None
        self.reentry_bar = None
        self.lowest_since_reentry = None
        self.bars_since_reentry = 0
        self.core_size = 0
        self.leverage = self.base_leverage
        print("🌙 RSI indicator initialized ✨")

    def next(self):
        if len(self.data) < self.rsi_period + 2:
            return

        rsi_val = self.rsi[-1]
        price = self.data.Close[-1]

        if np.isnan(rsi_val):
            return

        if self.state == 'IDLE':
            if rsi_val < self.derisk_threshold:
                if self.position:
                    current_size = self.position.size
                    if current_size > 0:
                        sell_size = int(round(current_size * self.derisk_pct))
                        if sell_size > 0:
                            self.sell(size=sell_size)
                            self.core_size = current_size - sell_size
                            print(f"🌙 DE-RISK! RSI={rsi_val:.2f} < 30 → Sold {sell_size} units, keeping {self.core_size} core 🚀")
                else:
                    self.core_size = int(round(self.equity * 0.95 / price))

                self.leverage = min(self.base_leverage * self.reentry_leverage_mult, self.max_leverage)
                reentry_size = int(round(self.core_size * (self.derisk_pct / (1 - self.derisk_pct)) * self.reentry_leverage_mult))
                if reentry_size > 0:
                    self.buy(size=reentry_size)
                    self.reentry_price = price
                    self.reentry_bar = len(self.data)
                    self.lowest_since_reentry = price
                    self.bars_since_reentry = 0
                    self.state = 'RE_LEVERED'
                    print(f"✨ RE-LEVER! Bought {reentry_size} units @ {price:.2f}, leverage={self.leverage:.2f}x 🚀")

        elif self.state == 'RE_LEVERED':
            self.bars_since_reentry += 1
            if price < self.lowest_since_reentry:
                self.lowest_since_reentry = price

            if self.bars_since_reentry == 1 and rsi_val > self.profit_threshold:
                if self.position:
                    current_size = self.position.size
                    take_size = int(round(current_size * self.profit_take_pct))
                    if take_size > 0:
                        self.sell(size=take_size)
                        print(f"🌙 PROFIT-TAKE! RSI={rsi_val:.2f} > 60 → Took {take_size} units profit ✨")
                if self.position and self.position.size <= 0:
                    self.state = 'IDLE'
                    self.core_size = 0
                else:
                    self.state = 'CORE'
                    self.core_size = self.position.size if self.position else 0
                return

            if rsi_val < self.derisk_threshold and self.bars_since_reentry >= self.stop_lookback:
                if price <= self.lowest_since_reentry and self.position:
                    self.position.close()
                    print(f"🛑 STOP OUT! New low {price:.2f}, RSI={rsi_val:.2f} stayed < 30 for {self.stop_lookback} bars")
                    self.state = 'IDLE'
                    self.core_size = 0
                    return

            if self.bars_since_reentry >= self.time_stop_candles:
                if rsi_val <= self.profit_threshold and self.position:
                    self.position.close()
                    print(f"⏰ TIME STOP! {self.time_stop_candles} candles elapsed, RSI={rsi_val:.2f} → Flattened")
                    self.state = 'IDLE'
                    self.core_size = 0
                    return

            if rsi_val > self.derisk_threshold and self.bars_since_reentry > 1:
                self.state = 'CORE'
                self.core_size = self.position.size if self.position else 0
                print(f"🌙 RSI reset above 30 ({rsi_val:.2f}) → back to CORE state ✨")

        elif self.state == 'CORE':
            if rsi_val < self.derisk_threshold:
                if self.position:
                    current_size = self.position.size
                    sell_size = int(round(current_size * self.derisk_pct))
                    if sell_size > 0:
                        self.sell(size=sell_size)
                        self.core_size = current_size - sell_size
                        print(f"🌙 DE-RISK again! RSI={rsi_val:.2f} → Sold {sell_size} units 🚀")

                self.leverage = min(self.base_leverage * self.reentry_leverage_mult, self.max_leverage)
                reentry_size = int(round(self.core_size * (self.derisk_pct / (1 - self.derisk_pct)) * self.reentry_leverage_mult))
                if reentry_size > 0:
                    self.buy(size=reentry_size)
                    self.reentry_price = price
                    self.reentry_bar = len(self.data)
                    self.lowest_since_reentry = price
                    self.bars_since_reentry = 0
                    self.state = 'RE_LEVERED'
                    print(f"✨ RE-LEVER again! Bought {reentry_size} units @ {price:.2f} 🚀")

        if not self.position and self.state == 'IDLE':
            init_size = int(round(self.equity * 0.95 / price))
            if init_size > 0:
                self.buy(size=init_size)
                self.core_size = init_size
                print(f"🌙 Initial long position opened: {init_size} units @ {price:.2f} ✨")


bt = Backtest(
    data,
    OscillatorRebalance,
    cash=1_000_000,
    commission=0.002,
    exclusive_orders=True
)

stats = bt.run()
print(stats)
print(stats._strategy)