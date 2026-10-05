import pandas as pd
import numpy as np
import talib
from backtesting import Backtest, Strategy

# 🌙 Moon Dev's VolatilityCompression Backtest ✨
print("🌙 Moon Dev is initializing the VolatilityCompression backtest... 🚀")

# Load data
data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper column mapping (capitalize)
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume',
    'datetime': 'Datetime'
}, inplace=True)

# Ensure datetime index
if 'Datetime' in data.columns:
    data['Datetime'] = pd.to_datetime(data['Datetime'])
    data = data.set_index('Datetime')

print(f"🌙 Data loaded: {len(data)} rows ✨")
print(f"🌙 Columns: {list(data.columns)} 🚀")


class VolatilityCompression(Strategy):
    """
    🌙 Moon Dev's VolatilityCompression Strategy ✨

    Entry: ATR(10) < SMA(ATR, 20) AND Close > SMA(Close, 200)
    Exit: ATR Stop-Loss OR 5% Trailing Stop
    """

    # Strategy parameters
    atr_period = 10
    atr_sma_period = 20
    sma_period = 200
    atr_stop_mult = 2.0
    trailing_pct = 0.05
    risk_per_trade = 0.02  # 2% risk per trade

    def init(self):
        print("🌙 Initializing indicators... ✨")

        # ATR (10) - use talib
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period)

        # SMA of ATR (20)
        self.atr_sma = self.I(talib.SMA, self.atr, timeperiod=self.atr_sma_period)

        # SMA of Close (200)
        self.sma200 = self.I(talib.SMA, self.data.Close, timeperiod=self.sma_period)

        # Track peak price for trailing stop
        self.peak_price = 0
        self.entry_price = 0
        self.atr_stop_price = 0

        print("🌙 Indicators initialized! 🚀")

    def next(self):
        # Skip if we don't have enough data
        if len(self.data) < self.sma_period + 5:
            return

        # Check for NaN values
        if (pd.isna(self.atr[-1]) or pd.isna(self.atr_sma[-1]) or
            pd.isna(self.sma200[-1])):
            return

        price = self.data.Close[-1]
        atr_val = self.atr[-1]
        atr_sma_val = self.atr_sma[-1]
        sma200_val = self.sma200[-1]

        # 🌙 Manage existing position
        if self.position:
            # Update peak price
            if self.data.High[-1] > self.peak_price:
                self.peak_price = self.data.High[-1]
                print(f"🌙 New peak: {self.peak_price:.2f} 📈")

            # Calculate trailing stop
            trailing_stop = self.peak_price * (1 - self.trailing_pct)

            # Check ATR stop-loss
            if self.data.Low[-1] <= self.atr_stop_price:
                print(f"🛑 ATR STOP-LOSS HIT! Price: {self.data.Low[-1]:.2f}, Stop: {self.atr_stop_price:.2f} 🌙")
                self.position.close()
                return

            # Check trailing stop
            if self.data.Low[-1] <= trailing_stop:
                print(f"🛑 TRAILING STOP HIT! Price: {self.data.Low[-1]:.2f}, Stop: {trailing_stop:.2f} 🌙")
                self.position.close()
                return

        # 🌙 Entry logic
        else:
            # Volatility compression: ATR < SMA of ATR
            compression = atr_val < atr_sma_val

            # Trend filter: close > 200 SMA
            uptrend = price > sma200_val

            if compression and uptrend:
                # Calculate position size based on risk
                risk_per_unit = self.atr_stop_mult * atr_val

                if risk_per_unit <= 0:
                    return

                # Position size as fraction of equity (0 < size < 1)
                # Risk 2% of equity per trade: size = (equity * risk) / risk_per_unit
                equity = self.equity
                units = (equity * self.risk_per_trade) / risk_per_unit

                if units <= 0:
                    return

                # Cap size as fraction of equity (backtesting.py uses fraction 0-1)
                # Convert units to fraction of equity based on current price
                size_fraction = (units * price) / equity
                size_fraction = min(max(size_fraction, 0.01), 0.99)

                # Set ATR stop price
                self.atr_stop_price = price - risk_per_unit
                self.entry_price = price
                self.peak_price = price

                print(f"🚀 LONG ENTRY! Price: {price:.2f} | ATR: {atr_val:.2f} | "
                      f"ATR SMA: {atr_sma_val:.2f} | SMA200: {sma200_val:.2f} 🌙")
                print(f"   🛑 ATR Stop: {self.atr_stop_price:.2f} | Size: {size_fraction:.4f} ✨")

                self.buy(size=size_fraction)


# 🌙 Run the backtest
print("🌙 Running VolatilityCompression backtest... 🚀")
bt = Backtest(
    data,
    VolatilityCompression,
    cash=1_000_000,
    commission=0.002,
    exclusive_orders=True
)

stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙 Moon Dev's VolatilityCompression backtest complete! ✨🚀")