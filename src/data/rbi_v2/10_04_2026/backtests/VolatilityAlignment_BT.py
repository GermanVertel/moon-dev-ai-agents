import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Load data
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')

# Clean columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper column mapping
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

print("🌙✨ Moon Dev VolatilityAlignment Backtest Loading... 🚀")
print(f"📊 Data shape: {data.shape}")
print(f"📈 Date range: {data.index[0]} to {data.index[-1]}")


class VolatilityAlignment(Strategy):
    bb_period = 50
    bb_std = 2.0
    sma_period = 20
    risk_pct = 0.02

    def init(self):
        close = self.data.Close

        # Bollinger Bands - 50 period, 2 std
        self.bb_upper, self.bb_mid, self.bb_lower = self.I(
            talib.BBANDS, close, timeperiod=self.bb_period,
            nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0
        )
        # BBMP is the middle band (50-period SMA)
        self.bbmp = self.bb_mid

        # 20-period SMA
        self.sma = self.I(talib.SMA, close, timeperiod=self.sma_period)

        # For trailing stop: 2 std below BBMP
        self.bb_std_dev = self.I(
            lambda x: pd.Series(x).rolling(self.bb_period).std().values,
            close
        )

        self.trailing_stop = None
        print("🌙 VolatilityAlignment indicators initialized ✨")

    def next(self):
        price = self.data.Close[-1]
        bbmp = self.bbmp[-1]
        sma = self.sma[-1]
        std_dev = self.bb_std_dev[-1]

        if np.isnan(bbmp) or np.isnan(sma) or np.isnan(std_dev):
            return

        # Compute trailing stop candidate: 2 std below BBMP
        candidate_stop = bbmp - (self.bb_std * std_dev)

        if not self.position:
            # Long entry: BBMP crosses above SMA
            if len(self.bbmp) > 2 and len(self.sma) > 2:
                prev_bbmp = self.bbmp[-2]
                prev_sma = self.sma[-2]
                if prev_bbmp <= prev_sma and bbmp > sma:
                    # Risk-based position sizing
                    risk_per_unit = price - candidate_stop
                    if risk_per_unit <= 0:
                        return
                    equity = self.equity
                    risk_amount = equity * self.risk_pct
                    size = risk_amount / risk_per_unit
                    size = int(round(size))
                    if size < 1:
                        size = 1
                    self.trailing_stop = candidate_stop
                    self.buy(size=size)
                    print(f"🚀🌙 LONG ENTRY | Price: {price:.2f} | BBMP: {bbmp:.2f} > SMA: {sma:.2f} | Size: {size} | Stop: {candidate_stop:.2f}")
        else:
            # Update trailing stop (only move upward)
            if candidate_stop > self.trailing_stop:
                self.trailing_stop = candidate_stop
                print(f"📈 Trailing stop raised to {self.trailing_stop:.2f}")

            # Exit if BBMP crosses below SMA
            if len(self.bbmp) > 2 and len(self.sma) > 2:
                prev_bbmp = self.bbmp[-2]
                prev_sma = self.sma[-2]
                if prev_bbmp >= prev_sma and bbmp < sma:
                    self.position.close()
                    print(f"🔻🌙 EXIT SIGNAL | BBMP: {bbmp:.2f} < SMA: {sma:.2f} | Price: {price:.2f}")
                    self.trailing_stop = None
                    return

            # Trailing stop hit
            if price <= self.trailing_stop:
                self.position.close()
                print(f"🛑 Trailing Stop Hit | Price: {price:.2f} <= Stop: {self.trailing_stop:.2f}")
                self.trailing_stop = None


bt = Backtest(data, VolatilityAlignment, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)