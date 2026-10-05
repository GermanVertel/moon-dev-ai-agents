import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

print("🌙✨ Moon Dev's ContrarianCrossover Backtest Loading... ✨🌙")

# Load data
data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Map to proper case
data.columns = [col.capitalize() for col in data.columns]

# Ensure datetime
if 'Datetime' in data.columns:
    data['Datetime'] = pd.to_datetime(data['Datetime'])
    data = data.set_index('Datetime')

# Required columns
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

print(f"🚀 Data loaded: {len(data)} rows")
print(f"📊 Columns: {list(data.columns)}")


class ContrarianCrossover(Strategy):
    """
    ContrarianCrossover: 3-day SMA momentum crossover on base asset.
    Since we trade the inverse ETF (approximated here by BTC-USD as proxy),
    a bearish base signal = bullish inverse -> go long.
    A bullish base signal = bearish inverse -> exit / go short.
    """
    sma_period = 3
    risk_pct = 0.02  # 2% risk per trade
    stop_loss_pct = 0.05  # 5% stop loss
    take_profit_pct = 0.10  # 10% take profit (premium capture analog)

    def init(self):
        print("🌙 Initializing indicators...")
        # ✅ talib.SMA wrapped in self.I() — NO backtesting.lib used
        self.sma = self.I(talib.SMA, self.data.Close, timeperiod=self.sma_period, name="SMA_3")
        print(f"✨ SMA({self.sma_period}) computed via talib")

    def next(self):
        # Guard against insufficient data (SMA warmup + need [-2] access)
        if len(self.data) < self.sma_period + 1:
            return

        price = self.data.Close[-1]
        sma_now = self.sma[-1]
        sma_prev = self.sma[-2]
        price_prev = self.data.Close[-2]

        # NaN guard (talib warmup produces NaNs)
        if np.isnan(sma_now) or np.isnan(sma_prev):
            return

        # ✅ Crossover detection via array indexing — NO backtesting.lib.crossover
        bullish_base = price_prev < sma_prev and price > sma_now  # base up -> inverse down
        bearish_base = price_prev > sma_prev and price < sma_now  # base down -> inverse up

        if not self.position:
            if bearish_base:
                # Sell cash-secured puts analog -> LONG inverse (long here)
                risk_amount = self.equity * self.risk_pct
                stop_price = price * (1 - self.stop_loss_pct)
                risk_per_unit = price - stop_price
                if risk_per_unit > 0:
                    size = int(round(risk_amount / risk_per_unit))
                    if size > 0:
                        print(f"🌙✨ BEARISH BASE SIGNAL -> LONG inverse @ {price:.2f} | size={size}")
                        self.buy(size=size, sl=stop_price, tp=price * (1 + self.take_profit_pct))
        else:
            if bullish_base:
                print(f"🚀 BULLISH BASE SIGNAL -> CLOSING inverse long @ {price:.2f}")
                self.position.close()


print("🌙 Running initial backtest...")
bt = Backtest(data, ContrarianCrossover, cash=1_000_000, commission=0.002)
stats = bt.run()
print(stats)
print(stats._strategy)
print("✨ Moon Dev backtest complete! ✨")