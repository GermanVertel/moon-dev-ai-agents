import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# 🌙 Moon Dev Data Loading & Cleaning
data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Ensure proper column mapping
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

# Set datetime index
if 'datetime' in data.columns:
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')

# Ensure OHLCV columns are float
for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')

data = data.dropna()

print(f"🌙✨ Data loaded: {len(data)} bars from {data.index[0]} to {data.index[-1]}")
print(f"🚀 Moon Dev is ready to launch the DivergentOscillator backtest!")


class DivergentOscillator(Strategy):
    """
    🌙 DivergentOscillator Strategy
    - Long entry: RSI(14) < 30 AND Price > 50-MA (bullish divergence)
    - Exit: CCI(20) crosses below 0
    - Stop: 1.5x ATR below entry
    """
    # Strategy parameters
    rsi_period = 14
    rsi_oversold = 30
    ma_period = 50
    cci_period = 20
    atr_period = 14
    atr_multiplier = 1.5
    risk_pct = 0.02  # 2% risk per trade

    def init(self):
        # 🌙 Calculate indicators using talib via self.I wrapper
        close = self.data.Close
        high = self.data.High
        low = self.data.Low

        self.rsi = self.I(talib.RSI, close, timeperiod=self.rsi_period, name='RSI')
        self.ma = self.I(talib.SMA, close, timeperiod=self.ma_period, name='MA50')
        self.cci = self.I(talib.CCI, high, low, close,
                          timeperiod=self.cci_period, name='CCI')
        self.atr = self.I(talib.ATR, high, low, close,
                          timeperiod=self.atr_period, name='ATR')
        print("🌙✨ Indicators initialized: RSI, MA50, CCI, ATR 🚀")

    def next(self):
        # Skip if indicators not ready
        if len(self.data) < self.ma_period + 2:
            return

        price = self.data.Close[-1]
        rsi_now = self.rsi[-1]
        rsi_prev = self.rsi[-2]
        ma_now = self.ma[-1]
        cci_now = self.cci[-1]
        cci_prev = self.cci[-2]
        atr_now = self.atr[-1]

        # Skip if any indicator is NaN
        if (np.isnan(rsi_now) or np.isnan(rsi_prev) or np.isnan(ma_now) or
                np.isnan(cci_now) or np.isnan(cci_prev) or np.isnan(atr_now)):
            return

        # 🌙 No open position - check entry
        if not self.position:
            # Entry conditions:
            # 1. RSI < 30 (oversold)
            # 2. Price > 50-MA (bullish divergence / underlying uptrend)
            # 3. RSI turning up (confirmation)
            if (rsi_now < self.rsi_oversold and
                price > ma_now and
                rsi_now > rsi_prev):

                # 🚀 Position sizing based on risk
                stop_price = price - (self.atr_multiplier * atr_now)
                risk_per_unit = price - stop_price

                if risk_per_unit <= 0:
                    return

                risk_amount = self.equity * self.risk_pct
                position_size = risk_amount / risk_per_unit

                # 🚀 Convert to fraction of equity for backtesting.py compatibility
                position_fraction = (position_size * price) / self.equity

                # Clamp to valid range (0 < size < 1)
                if position_fraction <= 0:
                    position_fraction = 0.01
                elif position_fraction >= 1:
                    position_fraction = 0.95

                # Ensure size is strictly between 0 and 1 (backtesting.py requires fraction)
                position_fraction = float(position_fraction)
                if not (0 < position_fraction < 1):
                    position_fraction = 0.5

                # 🚀 Round to avoid floating point precision issues with backtesting.py
                position_fraction = round(position_fraction, 6)
                if position_fraction <= 0:
                    position_fraction = 0.01
                if position_fraction >= 1:
                    position_fraction = 0.95

                print(f"🌙✨ ENTRY SIGNAL! Price={price:.2f} | RSI={rsi_now:.2f} "
                      f"| MA50={ma_now:.2f} | Size={position_fraction:.4f} | Stop={stop_price:.2f} 🚀")

                self.buy(size=position_fraction, sl=stop_price)

        # 🌙 Position open - check exit via CCI
        else:
            # Exit when CCI crosses below 0 (bearish momentum flip)
            if cci_prev >= 0 and cci_now < 0:
                print(f"🌙 EXIT SIGNAL! CCI crossed below 0: {cci_prev:.2f} -> {cci_now:.2f} "
                      f"| Price={price:.2f} 💫")
                self.position.close()


# 🌙 Run the backtest
bt = Backtest(
    data,
    DivergentOscillator,
    cash=1_000_000,
    commission=0.002,
    exclusive_orders=True
)

print("🚀 Moon Dev launching backtest... 🌙")
stats = bt.run()
print(stats)
print(stats._strategy)