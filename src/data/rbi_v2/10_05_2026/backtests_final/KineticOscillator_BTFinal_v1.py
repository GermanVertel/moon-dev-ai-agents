import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Load and prepare data
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
data = data.dropna()

print("🌙✨ Moon Dev KineticOscillator Backtest Initializing... 🚀")
print(f"📊 Data loaded: {len(data)} bars")
print(f"📈 Date range: {data.index[0]} to {data.index[-1]}")


class KineticOscillator(Strategy):
    # Stochastic parameters
    stoch_k = 14
    stoch_d = 3
    stoch_smooth = 3
    # RSI parameters
    rsi_period = 14
    # Support/resistance lookback
    sr_period = 5
    # Risk management
    risk_pct = 0.01  # 1% risk per trade
    rr_ratio = 1.5   # 1:1.5 risk-reward

    def init(self):
        print("🌙 Initializing indicators... ✨")
        # Stochastic - talib.STOCH returns (slowk, slowd)
        stoch = self.I(
            talib.STOCH,
            self.data.High, self.data.Low, self.data.Close,
            fastk_period=self.stoch_k,
            slowk_period=self.stoch_smooth,
            slowk_matype=0,
            slowd_period=self.stoch_d,
            slowd_matype=0
        )
        self.stoch_k_line = stoch[0]
        self.stoch_d_line = stoch[1]
        # RSI
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=self.rsi_period)
        # Short-term resistance/support
        self.resistance = self.I(talib.MAX, self.data.High, timeperiod=self.sr_period)
        self.support = self.I(talib.MIN, self.data.Low, timeperiod=self.sr_period)
        print("🚀 Indicators ready! Moon Dev is watching the charts... 🌙")

    def next(self):
        price = self.data.Close[-1]

        # Skip if indicators not ready
        if len(self.data) < 3:
            return
        if np.isnan(self.stoch_k_line[-1]) or np.isnan(self.stoch_d_line[-1]) or np.isnan(self.rsi[-1]):
            return
        if np.isnan(self.stoch_k_line[-2]) or np.isnan(self.stoch_d_line[-2]) or np.isnan(self.rsi[-2]):
            return

        # Current and previous values
        k_now = self.stoch_k_line[-1]
        k_prev = self.stoch_k_line[-2]
        d_now = self.stoch_d_line[-1]
        d_prev = self.stoch_d_line[-2]
        rsi_now = self.rsi[-1]
        rsi_prev = self.rsi[-2]

        # If in a position, check exit conditions
        if self.position:
            if self.position.is_long:
                # Long exit: Stoch cross below D from above 80, or RSI crosses below 70
                if (k_prev > d_prev and k_now < d_now and k_prev > 80) or \
                   (rsi_prev > 70 and rsi_now < 70):
                    print(f"🌙 EXIT LONG @ {price:.2f} | Stoch K:{k_now:.1f} D:{d_now:.1f} RSI:{rsi_now:.1f} ✨")
                    self.position.close()
            elif self.position.is_short:
                # Short exit: Stoch cross above D from below 20, or RSI crosses above 30
                if (k_prev < d_prev and k_now > d_now and k_prev < 20) or \
                   (rsi_prev < 30 and rsi_now > 30):
                    print(f"🌙 EXIT SHORT @ {price:.2f} | Stoch K:{k_now:.1f} D:{d_now:.1f} RSI:{rsi_now:.1f} ✨")
                    self.position.close()
            return

        # Long entry conditions
        stoch_bull_cross = k_prev < d_prev and k_now > d_now and k_prev < 20
        rsi_bull_cross = rsi_prev < 30 and rsi_now > 30
        price_breakout_long = price > self.resistance[-2]

        # Short entry conditions
        stoch_bear_cross = k_prev > d_prev and k_now < d_now and k_prev > 80
        rsi_bear_cross = rsi_prev > 70 and rsi_now < 70
        price_breakout_short = price < self.support[-2]

        # Long entry
        if stoch_bull_cross and rsi_bull_cross and price_breakout_long:
            # Stop loss at recent swing low
            sl = self.support[-1]
            risk = price - sl
            if risk > 0:
                tp = price + risk * self.rr_ratio
                # Position sizing: risk 1% of equity -> convert to fraction of equity
                size_units = int(round((self.equity * self.risk_pct) / risk))
                if size_units > 0:
                    # Convert to fraction of equity for backtesting.py (0 < size < 1)
                    size_frac = min(0.95, (size_units * price) / self.equity)
                    if size_frac > 0:
                        print(f"🚀 LONG ENTRY @ {price:.2f} | SL:{sl:.2f} TP:{tp:.2f} Size:{size_frac:.4f} | Stoch K:{k_now:.1f} D:{d_now:.1f} RSI:{rsi_now:.1f} 🌙")
                        self.buy(size=size_frac, sl=sl, tp=tp)

        # Short entry
        elif stoch_bear_cross and rsi_bear_cross and price_breakout_short:
            sl = self.resistance[-1]
            risk = sl - price
            if risk > 0:
                tp = price - risk * self.rr_ratio
                size_units = int(round((self.equity * self.risk_pct) / risk))
                if size_units > 0:
                    size_frac = min(0.95, (size_units * price) / self.equity)
                    if size_frac > 0:
                        print(f"🚀 SHORT ENTRY @ {price:.2f} | SL:{sl:.2f} TP:{tp:.2f} Size:{size_frac:.4f} | Stoch K:{k_now:.1f} D:{d_now:.1f} RSI:{rsi_now:.1f} 🌙")
                        self.sell(size=size_frac, sl=sl, tp=tp)


bt = Backtest(data, KineticOscillator, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)