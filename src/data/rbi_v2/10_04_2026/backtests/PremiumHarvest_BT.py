import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

print("🌙 Moon Dev PremiumHarvest Backtest Initializing... ✨🚀")

# Load data
data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper column mapping
data.rename(columns={
    'datetime': 'datetime',
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

print(f"🌙 Data loaded: {len(data)} bars ✨")
print(f"🚀 Date range: {data.index[0]} to {data.index[-1]}")


class PremiumHarvest(Strategy):
    """
    PremiumHarvest Strategy:
    - Trend filter: price above 200-period MA for entire trailing ~21 sessions
    - Entry: sell OTM put equivalent (simulated via long entry in bullish regime)
    - Exit: 10% profit target OR 20% stop loss OR trend break
    Since we don't have options data, we simulate the directional bias
    (bullish premium harvest) using the futures price with the same risk profile.
    """
    ma_period = 200
    trend_lookback = 21  # ~21 sessions of above-MA confirmation
    profit_target_pct = 0.10  # 10% profit target
    stop_loss_pct = 0.20      # 20% stop loss
    risk_pct = 0.02           # 2% risk per trade
    size = 1_000_000

    def init(self):
        print("🌙 Initializing PremiumHarvest indicators... ✨")
        self.ma200 = self.I(talib.SMA, self.data.Close, timeperiod=self.ma_period)
        # Realized volatility (20-period) for volatility guard
        self.rv = self.I(talib.STDDEV, self.data.Close, timeperiod=20)
        print("🚀 Indicators ready: MA200 + Realized Volatility 🌙")

    def next(self):
        # Need enough data
        if len(self.data) < self.ma_period + self.trend_lookback + 1:
            return

        price = self.data.Close[-1]
        ma = self.ma200[-1]

        if np.isnan(ma):
            return

        # Trend filter: price above MA for entire trailing window
        window = self.data.Close[-self.trend_lookback:]
        ma_window = self.ma200[-self.trend_lookback:]
        above_ma_all = np.all(window > ma_window)

        # Volatility guard: skip if realized vol is extreme (top 5% historically)
        rv_now = self.rv[-1]
        rv_hist = self.rv[-500:] if len(self.rv) >= 500 else self.rv
        rv_hist = rv_hist[~np.isnan(rv_hist)]
        vol_ok = True
        if len(rv_hist) > 50:
            vol_ok = rv_now < np.percentile(rv_hist, 95)

        # Entry: bullish regime confirmed
        if not self.position and above_ma_all and vol_ok:
            # Position sizing based on risk
            risk_amount = self.equity * self.risk_pct
            risk_per_unit = price * self.stop_loss_pct
            if risk_per_unit > 0:
                position_size = int(round(risk_amount / risk_per_unit))
                # Cap to fixed size parameter
                position_size = min(position_size, self.size)
                if position_size > 0:
                    self.entry_price = price
                    self.buy(size=position_size)
                    print(f"🌙✨ PREMIUM HARVEST ENTRY ✨🌙 | Price: {price:.2f} | MA200: {ma:.2f} | Size: {position_size} 🚀")

        # Exit logic
        elif self.position:
            entry = self.position.entry_price if hasattr(self.position, 'entry_price') else self.entry_price
            pnl_pct = (price - entry) / entry

            # Profit target hit
            if pnl_pct >= self.profit_target_pct:
                self.position.close()
                print(f"🌙💰 PROFIT TARGET HIT 💰🌙 | PnL: {pnl_pct*100:.2f}% | Price: {price:.2f} ✨")

            # Stop loss hit
            elif pnl_pct <= -self.stop_loss_pct:
                self.position.close()
                print(f"🌙🛑 STOP LOSS HIT 🛑🌙 | PnL: {pnl_pct*100:.2f}% | Price: {price:.2f}")

            # Trend break: price closes below MA200
            elif price < ma:
                self.position.close()
                print(f"🌙⚠️ TREND BREAK EXIT ⚠️🌙 | Price: {price:.2f} < MA200: {ma:.2f} 🚀")


print("🌙 Setting up backtest... ✨")
bt = Backtest(
    data,
    PremiumHarvest,
    cash=1_000_000,
    commission=0.002,
    exclusive=False
)

print("🚀 Running PremiumHarvest backtest... 🌙")
stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙✨ PremiumHarvest Backtest Complete! ✨🌙")