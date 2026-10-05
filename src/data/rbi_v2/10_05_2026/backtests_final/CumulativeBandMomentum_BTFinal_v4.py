import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

print("🌙 Moon Dev's CumulativeBandMomentum Backtest Loading... ✨🚀")

data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')

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

print(f"🌙 Data loaded: {len(data)} bars ✨")
print(f"🚀 Date range: {data.index[0]} to {data.index[-1]}")


class CumulativeBandMomentum(Strategy):
    bb_period = 20
    bb_k = 2.0
    cp_window = 50
    risk_pct = 0.02

    def init(self):
        close = pd.Series(self.data.Close)

        sma = close.rolling(self.bb_period).mean()
        sd = close.rolling(self.bb_period).std()

        bb_lower_vals = (sma - self.bb_k * sd).values
        bb_mid_vals = sma.values
        bb_upper_vals = (sma + self.bb_k * sd).values

        self.lower = self.I(lambda: bb_lower_vals)
        self.mid = self.I(lambda: bb_mid_vals)
        self.upper = self.I(lambda: bb_upper_vals)

        # Normalized lower band: (LB - SMA)/SMA -> can flip sign
        norm_lb = (bb_lower_vals - bb_mid_vals) / bb_mid_vals

        def cp_func(arr):
            s = pd.Series(arr)
            out = np.full(len(s), np.nan)
            for i in range(self.cp_window - 1, len(s)):
                window = s.iloc[i - self.cp_window + 1:i + 1].values
                if np.any(np.isnan(window)):
                    continue
                # Use log-sum to avoid overflow/underflow while preserving sign
                prod = 1.0
                for v in window:
                    prod *= v
                    # Rescale to prevent overflow
                    if abs(prod) > 1e100:
                        prod = np.sign(prod) * 1e100
                    elif abs(prod) < 1e-100 and prod != 0:
                        prod = np.sign(prod) * 1e-100
                out[i] = prod
            return out

        self.cp = self.I(cp_func, norm_lb)

        print("🌙 Indicators initialized: BB Lower/Mid/Upper + Cumulative Product ✨")

    def next(self):
        price = self.data.Close[-1]
        cp = self.cp[-1]
        mid = self.mid[-1]
        lower = self.lower[-1]
        upper = self.upper[-1]

        if np.isnan(cp) or np.isnan(mid) or np.isnan(lower) or np.isnan(upper):
            return

        # Determine signal
        if cp > 0:
            signal = 'long'
        elif cp < 0:
            signal = 'short'
        else:
            signal = None

        if signal is None:
            return

        # Check current position
        current = None
        if self.position:
            if self.position.is_long:
                current = 'long'
            elif self.position.is_short:
                current = 'short'

        # Position flipping
        if current == signal:
            return

        # Close existing position
        if self.position:
            print(f"🌙 Moon Dev Exit: Closing {current} @ {price:.2f} | CP={cp:.6f} ✨")
            self.position.close()

        # Calculate stop
        if signal == 'long':
            stop = lower
            risk_per_unit = price - stop
            if risk_per_unit <= 0:
                return
            risk_amount = self.equity * self.risk_pct
            size = int(round(risk_amount / risk_per_unit))
            if size < 1:
                size = 1
            print(f"🚀 Moon Dev LONG Entry @ {price:.2f} | Stop={stop:.2f} | Size={size} | CP={cp:.6f} 🌙")
            self.buy(size=size, sl=stop)
        else:
            stop = upper
            risk_per_unit = stop - price
            if risk_per_unit <= 0:
                return
            risk_amount = self.equity * self.risk_pct
            size = int(round(risk_amount / risk_per_unit))
            if size < 1:
                size = 1
            print(f"🚀 Moon Dev SHORT Entry @ {price:.2f} | Stop={stop:.2f} | Size={size} | CP={cp:.6f} 🌙")
            self.sell(size=size, sl=stop)


bt = Backtest(data, CumulativeBandMomentum, cash=1_000_000, commission=0.002)

print("🌙 Running initial backtest with default parameters... ✨🚀")
stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙 Moon Dev Backtest Complete! ✨🚀")