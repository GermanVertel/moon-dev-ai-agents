import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

print("🌙✨ Initializing Moon Dev's ContrarianAnalyst Backtest ✨🌙")

# Load data
data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
print(f"🚀 Loading data from: {data_path}")
data = pd.read_csv(data_path)

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper case mapping
data.columns = [col.capitalize() for col in data.columns]
print(f"📊 Data columns: {list(data.columns)}")

# Ensure datetime index
if 'Datetime' in data.columns:
    data['Datetime'] = pd.to_datetime(data['Datetime'])
    data = data.set_index('Datetime')
    data.index.name = 'Datetime'

print(f"🌙 Data shape: {data.shape}")
print(f"✨ First few rows:\n{data.head()}")


class ContrarianAnalyst(Strategy):
    """
    ContrarianAnalyst Strategy 🌙
    - Buy laggards above 50-DMA (trend intact)
    - Exit on new all-time high (profit-taking)
    - Stop-loss below 50-DMA
    """
    # Parameters
    sma_period = 50
    lookback_return = 30      # ~1 month of 15m bars (approx)
    stop_loss_pct = 0.03      # 3% stop below 50-DMA proxy
    take_profit_pct = 0.10    # 10% take profit
    position_size = 0.95      # Fraction of equity (0 < size < 1)

    def init(self):
        print("🌙 Initializing indicators...")
        close = self.data.Close
        high = self.data.High

        # 50-period SMA (trend filter) — talib wrapped in self.I
        self.sma50 = self.I(talib.SMA, close, timeperiod=self.sma_period)

        # Trailing return (underperformance proxy) — numpy/pandas, no backtesting.lib
        self.ret = self.I(
            lambda x: pd.Series(x).pct_change(self.lookback_return).values * 100,
            close
        )

        # All-time high tracker = running (expanding) max of highs — no talib needed
        self.ath = self.I(
            lambda x: pd.Series(x).expanding().max().values,
            high
        )

        # Year-to-date high proxy using rolling 96-bar high (1 day on 15m)
        self.ytd_high = self.I(talib.MAX, high, timeperiod=96)

        print("✨ Indicators ready! 🚀")

    def next(self):
        price = self.data.Close[-1]
        sma = self.sma50[-1]
        ret = self.ret[-1]
        ath = self.ath[-1]
        ytd = self.ytd_high[-1]

        # Skip if indicators not ready
        if np.isnan(sma) or np.isnan(ret) or np.isnan(ath):
            return

        # ENTRY: laggard (negative trailing return) + above 50-DMA (trend intact)
        if not self.position:
            if ret < 0 and price > sma:
                print(f"🌙✨ BUY SIGNAL! Price={price:.2f}, SMA50={sma:.2f}, Ret={ret:.2f}%")
                sl = price * (1 - self.stop_loss_pct)
                tp = price * (1 + self.take_profit_pct)
                self.buy(size=self.position_size, sl=sl, tp=tp)

        # EXIT: new all-time high breach (sell winners)
        else:
            if price >= ath * 0.999:  # Near ATH
                print(f"🚀💰 ATH BREACH EXIT! Price={price:.2f}, ATH={ath:.2f}")
                self.position.close()
            # Stop-loss: close below 50-DMA
            elif price < sma * (1 - self.stop_loss_pct):
                print(f"🛑 STOP LOSS! Price={price:.2f}, SMA50={sma:.2f}")
                self.position.close()


print("🌙🚀 Running backtest...")
bt = Backtest(
    data,
    ContrarianAnalyst,
    cash=1_000_000,
    commission=0.002,
    exclusive_orders=True
)

stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙✨ Backtest complete! ✨🌙")