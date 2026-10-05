import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# 🌙 Moon Dev's VolatilityContango Backtest 🌙
# Since we only have BTC-USD data, we proxy VIX term structure using
# realized volatility (short vs long window) as a contango proxy.
# Basis = (ShortVol - LongVol) / LongVol

class VolatilityContango(Strategy):
    # Strategy parameters
    short_vol_window = 10    # proxy for VIX cash (fast realized vol)
    long_vol_window = 60     # proxy for VIX futures (slow realized vol)
    basis_threshold = 0.10   # 10% contango trigger
    basis_exit = 0.05        # 5% compression exit
    atr_period = 14
    stop_mult = 2.0          # 2x premium stop rule
    tp_mult = 1.5            # +50% profit target
    risk_pct = 0.02          # 2% equity risk per trade

    def init(self):
        close = pd.Series(self.data.Close)
        # 🌙 Realized volatility proxies for VIX cash vs futures
        log_ret = np.log(close / close.shift(1))

        def realized_vol(series, window):
            return series.rolling(window).std() * np.sqrt(252)

        self.short_vol = self.I(
            lambda s, w: realized_vol(pd.Series(s), w).values,
            log_ret.values, self.short_vol_window, name="ShortVol"
        )
        self.long_vol = self.I(
            lambda s, w: realized_vol(pd.Series(s), w).values,
            log_ret.values, self.long_vol_window, name="LongVol"
        )

        # 🌙 Basis = (ShortVol - LongVol) / LongVol  (contango proxy)
        self.basis = self.I(
            lambda s, l: (s - l) / np.where(l == 0, np.nan, l),
            self.short_vol, self.long_vol, name="Basis"
        )

        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close,
                          timeperiod=self.atr_period, name="ATR")

        # 🌙 Track entry state (Position object has no entry_price attribute)
        self.entry_price = None
        self.entry_atr = None

        print("🌙✨ VolatilityContango indicators initialized! 🚀")

    def next(self):
        price = self.data.Close[-1]
        basis = self.basis[-1]
        atr = self.atr[-1]

        if np.isnan(basis) or np.isnan(atr):
            return

        # 🌙 Entry: deep contango signal
        if not self.position:
            if basis > self.basis_threshold:
                # Position size: fixed fractional risk
                risk_amount = self.equity * self.risk_pct
                # Use ATR as proxy for option premium risk unit
                risk_per_unit = atr * self.stop_mult
                if risk_per_unit <= 0:
                    return
                size = int(round(risk_amount / risk_per_unit))
                if size < 1:
                    size = 1
                print(f"🌙🚀 ENTRY LONG (put proxy) | Basis={basis:.4f} > {self.basis_threshold} | "
                      f"Price={price:.2f} | Size={size}")
                self.buy(size=size)
                self.entry_price = price
                self.entry_atr = atr

        # 🌙 Exit logic
        else:
            entry = self.entry_price
            if entry is None or self.entry_atr is None:
                return
            # Profit target: +50% (put proxy long)
            if price >= entry + self.tp_mult * self.entry_atr:
                print(f"🌙✨ TAKE PROFIT | Price={price:.2f} >= {entry + self.tp_mult*self.entry_atr:.2f}")
                self.position.close()
                self.entry_price = None
                self.entry_atr = None

            # Stop loss: 2x premium rule
            elif price <= entry - self.stop_mult * self.entry_atr:
                print(f"🌙🛑 STOP LOSS (2x premium) | Price={price:.2f} <= {entry - self.stop_mult*self.entry_atr:.2f}")
                self.position.close()
                self.entry_price = None
                self.entry_atr = None

            # Signal exit: basis compression below 5%
            elif basis < self.basis_exit:
                print(f"🌙📉 SIGNAL EXIT | Basis={basis:.4f} < {self.basis_exit}")
                self.position.close()
                self.entry_price = None
                self.entry_atr = None


# 🌙 Data loading
data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Map to backtesting.py required format
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

data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

print(f"🌙 Data loaded: {len(data)} rows ✨")
print(f"🚀 Date range: {data.index[0]} → {data.index[-1]}")

# 🌙 Run backtest
bt = Backtest(
    data,
    VolatilityContango,
    cash=1_000_000,
    commission=0.002
)

stats = bt.run()
print(stats)
print(stats._strategy)