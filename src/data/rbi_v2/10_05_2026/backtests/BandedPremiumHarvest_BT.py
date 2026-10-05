import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

print("🌙✨ Moon Dev Backtest AI - BandedPremiumHarvest ✨🌙")

# Load data
data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper column mapping
data.columns = [col.capitalize() for col in data.columns]
data = data.rename(columns={'Datetime': 'Date'})

# Ensure required columns
required = ['Open', 'High', 'Low', 'Close', 'Volume']
for col in required:
    if col not in data.columns:
        raise ValueError(f"Missing column: {col}")

data['Date'] = pd.to_datetime(data['Date'])
data = data.set_index('Date')
data = data[required]

print(f"🚀 Data loaded: {len(data)} rows from {data.index[0]} to {data.index[-1]}")


class BandedPremiumHarvest(Strategy):
    """
    🌙 BandedPremiumHarvest Strategy
    Sell overbought conditions when upper Bollinger Band intersects 5-day high.
    Since we can't trade actual options, we simulate the contrarian short
    bias by shorting the underlying when the overbought signal fires,
    with 50% profit target and stop loss at 2x risk.
    """
    bb_period = 20
    bb_std = 2.0
    high_lookback = 5
    risk_pct = 0.02
    reward_ratio = 1.5  # 1.5R target (analogous to 50% premium decay)

    def init(self):
        print("🌙 Initializing BandedPremiumHarvest indicators...")
        close = self.data.Close
        high = self.data.High

        # Bollinger Bands
        self.bb_upper = self.I(talib.BBANDS, close, timeperiod=self.bb_period,
                               nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0,
                               name='BB_Upper', index=0)
        self.bb_mid = self.I(talib.BBANDS, close, timeperiod=self.bb_period,
                             nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0,
                             name='BB_Mid', index=1)
        self.bb_lower = self.I(talib.BBANDS, close, timeperiod=self.bb_period,
                               nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0,
                               name='BB_Lower', index=2)

        # Rolling 5-day highest high
        self.hh5 = self.I(talib.MAX, high, timeperiod=self.high_lookback,
                          name='HH5')

        # ATR for risk sizing
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, close,
                          timeperiod=14, name='ATR')

        print("✨ Indicators ready: BB(20,2), HH(5), ATR(14)")

    def next(self):
        price = self.data.Close[-1]

        # Skip if indicators not ready
        if np.isnan(self.bb_upper[-1]) or np.isnan(self.hh5[-1]) or np.isnan(self.atr[-1]):
            return

        upper = self.bb_upper[-1]
        hh5 = self.hh5[-1]
        atr = self.atr[-1]

        # Entry: overbought — price high reaches upper band AND equals 5-day high
        overbought = (self.data.High[-1] >= upper) and (self.data.High[-1] >= hh5 * 0.999)

        if not self.position and overbought:
            # Risk-based sizing
            risk_amount = self.equity * self.risk_pct
            stop_distance = 2.0 * atr
            if stop_distance <= 0:
                return
            position_size = int(round(risk_amount / stop_distance))
            if position_size <= 0:
                return

            sl = price + stop_distance
            tp = price - (stop_distance * self.reward_ratio)

            print(f"🌙✨ SHORT SIGNAL | Price={price:.2f} | Upper={upper:.2f} | "
                  f"HH5={hh5:.2f} | Size={position_size} | SL={sl:.2f} | TP={tp:.2f}")
            self.sell(size=position_size, sl=sl, tp=tp)

        # Exit on time / mean reversion back to mid band
        elif self.position and self.position.is_short:
            if price <= self.bb_mid[-1]:
                print(f"🚀 Mean reversion exit at {price:.2f}")
                self.position.close()


# Run backtest
print("🌙 Running initial backtest...")
bt = Backtest(data, BandedPremiumHarvest, cash=1_000_000, commission=0.0002,
              exclusive_orders=True)
stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙✨ Moon Dev Backtest Complete ✨🌙")