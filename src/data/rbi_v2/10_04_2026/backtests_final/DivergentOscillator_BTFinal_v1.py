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
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=self.rsi_period, name='RSI')
        self.ma = self.I(talib.SMA, self.data.Close, timeperiod=self.ma_period, name='MA50')
        self.cci = self.I(talib.CCI, self.data.High, self.data.Low, self.data.Close,
                          timeperiod=self.cci_period, name='CCI')
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close,
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
                position_size = int(round(position_size))

                if position_size <= 0:
                    return

                # Cap size at 1,000,000 units per instructions
                position_size = min(position_size, 1_000_000)

                print(f"🌙✨ ENTRY SIGNAL! Price={price:.2f} | RSI={rsi_now:.2f} "
                      f"| MA50={ma_now:.2f} | Size={position_size} | Stop={stop_price:.2f} 🚀")

                self.buy(size=position_size, sl=stop_price)

        # 🌙 Position open - check exit via CCI
        else:
            # Exit when CCI crosses below 0 (bearish momentum flip)
            # Replaced backtesting.lib.crossover with manual array comparison
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