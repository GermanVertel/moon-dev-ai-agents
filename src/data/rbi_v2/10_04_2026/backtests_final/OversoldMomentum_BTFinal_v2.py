import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# 🌙 Moon Dev's OversoldMomentum Backtest 🌙
print("🚀 Moon Dev is warming up the engines...")
print("✨ Loading data from the lunar base...")

data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"

data = pd.read_csv(data_path)

# 🧹 Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# 🗺️ Map columns to backtesting.py requirements
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume',
})

# Set datetime index
if 'datetime' in data.columns:
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')
    data.index.name = 'Datetime'

# 🌙 Ensure numeric dtypes for OHLCV
for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')

data = data.dropna(subset=['Open', 'High', 'Low', 'Close'])

print(f"✅ Data loaded: {len(data)} rows from {data.index[0]} to {data.index[-1]}")
print(f"📊 Columns: {list(data.columns)}")


class OversoldMomentum(Strategy):
    """
    🌙 OversoldMomentum Strategy
    - Entry: 50-period RSI dips below 30, confirmed by crossing back above 30
    - Exit: Price reaches 20-period MA (mean reversion target)
    - Stop: 20% of ATR below entry
    - Risk: 1% of equity per trade
    """
    rsi_period = 50
    ma_period = 20
    atr_period = 14
    rsi_oversold = 30
    trend_ma_period = 200
    risk_pct = 0.01
    stop_atr_mult = 0.20
    max_bars_held = 100

    def init(self):
        print("🌙 Initializing Moon Dev indicators...")
        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)

        self.rsi = self.I(talib.RSI, close, timeperiod=self.rsi_period)
        self.ma = self.I(talib.SMA, close, timeperiod=self.ma_period)
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)
        self.trend_ma = self.I(talib.SMA, close, timeperiod=self.trend_ma_period)
        self.bars_held = 0
        print("✨ Indicators ready: RSI(50), SMA(20), ATR(14), SMA(200)")

    def next(self):
        price = self.data.Close[-1]

        # 🛡️ Manage open position
        if self.position:
            self.bars_held += 1

            # 🎯 Primary exit: price reaches 20-MA (mean reversion target)
            if price >= self.ma[-1]:
                print(f"🎯 [EXIT-MA] Price {price:.2f} hit 20-MA {self.ma[-1]:.2f} — taking profits! 🌙")
                self.position.close()
                self.bars_held = 0
                return

            # ⏰ Time-based exit
            if self.bars_held >= self.max_bars_held:
                print(f"⏰ [EXIT-TIME] Held {self.bars_held} bars — freeing capital! 🚀")
                self.position.close()
                self.bars_held = 0
                return
            return

        # 🔍 Entry logic: RSI crossed back above 30 (reversal confirmation)
        if len(self.rsi) < 2:
            return

        rsi_prev = self.rsi[-2]
        rsi_now = self.rsi[-1]

        if np.isnan(rsi_prev) or np.isnan(rsi_now):
            return

        # Oversold dip followed by recovery cross
        if rsi_prev < self.rsi_oversold and rsi_now >= self.rsi_oversold:
            # Optional trend filter: price above 200-MA
            if not np.isnan(self.trend_ma[-1]) and price < self.trend_ma[-1]:
                print(f"🚫 [SKIP] RSI reversal but price {price:.2f} below 200-MA {self.trend_ma[-1]:.2f}")
                return

            atr_val = self.atr[-1]
            if np.isnan(atr_val) or atr_val <= 0:
                return

            stop_distance = self.stop_atr_mult * atr_val
            if stop_distance <= 0:
                return

            # 💰 Position sizing: risk 1% of equity
            risk_amount = self.equity * self.risk_pct
            position_size = int(round(risk_amount / stop_distance))
            if position_size < 1:
                position_size = 1

            # Cap by equity
            max_size = int(self.equity / price)
            if position_size > max_size:
                position_size = max_size
            if position_size < 1:
                return

            stop_price = price - stop_distance
            print(f"🌙 [ENTRY] RSI reversal {rsi_prev:.2f}->{rsi_now:.2f} | Price {price:.2f} | "
                  f"ATR {atr_val:.2f} | Stop {stop_price:.2f} | Size {position_size} 🚀")

            self.buy(size=position_size, sl=stop_price)
            self.bars_held = 0


print("🌙✨ Launching Moon Dev backtest...")
bt = Backtest(data, OversoldMomentum, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)