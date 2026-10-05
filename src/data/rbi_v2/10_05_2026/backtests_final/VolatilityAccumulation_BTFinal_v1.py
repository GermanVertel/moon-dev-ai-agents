import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

print("🌙✨ Moon Dev VolatilityAccumulation Backtest Starting! 🚀")

# Load data
data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean column names
data.columns = data.columns.str.strip().str.lower()

# Drop unnamed columns
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper column mapping
data.columns = [col.capitalize() for col in data.columns]

# Ensure datetime index
if 'Datetime' in data.columns:
    data['Datetime'] = pd.to_datetime(data['Datetime'])
    data = data.set_index('Datetime')

data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

print(f"🌙 Data loaded: {len(data)} bars from {data.index[0]} to {data.index[-1]} ✨")


class VolatilityAccumulation(Strategy):
    # Parameters
    vol_window = 30
    drop_window = 3
    std_multiplier = 2.0
    obv_lookback = 5
    exit_std = 1.0
    max_bars_hold = 50
    stop_std = 3.0
    risk_pct = 0.02

    def init(self):
        print("🌙 Initializing indicators... ✨")

        close = pd.Series(self.data.Close)
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        # Daily returns
        returns = close.pct_change()
        self.returns = self.I(lambda x: x, returns.fillna(0).values, name="Returns")

        # Rolling std of returns
        self.vol = self.I(
            lambda x: pd.Series(x).rolling(self.vol_window).std().values,
            returns.fillna(0).values,
            name="Volatility"
        )

        # Rolling mean of returns
        self.mean_ret = self.I(
            lambda x: pd.Series(x).rolling(self.vol_window).mean().values,
            returns.fillna(0).values,
            name="MeanRet"
        )

        # 3-day cumulative % change
        self.pct3 = self.I(
            lambda x: pd.Series(x).pct_change(self.drop_window).values,
            close.values,
            name="Pct3"
        )

        # OBV
        self.obv = self.I(talib.OBV, close.values, volume, name="OBV")

        # ATR for reference
        self.atr = self.I(talib.ATR, high, low, close.values, timeperiod=14, name="ATR")

        # Track entry info
        self.entry_bar = None
        self.entry_price = None
        self.stop_price = None

        print("🌙 Indicators ready! 🚀")

    def next(self):
        i = len(self.data) - 1

        # Need enough historical data
        if i < self.vol_window + self.drop_window + 2:
            return

        vol = self.vol[i]
        mean_ret = self.mean_ret[i]
        pct3 = self.pct3[i]
        obv = self.obv[i]

        if np.isnan(vol) or np.isnan(mean_ret) or np.isnan(pct3) or np.isnan(obv):
            return

        # Threshold: 2 std below mean cumulative return over 3 days
        # Approximate 3-day cumulative mean and std
        cum_mean = mean_ret * self.drop_window
        cum_std = vol * np.sqrt(self.drop_window)
        threshold = cum_mean - self.std_multiplier * cum_std

        # OBV accumulation check: OBV rising or flat over lookback
        if i >= self.obv_lookback:
            obv_prev = self.obv[i - self.obv_lookback]
            obv_rising = obv >= obv_prev
        else:
            obv_rising = False

        # ---- EXIT LOGIC ----
        if self.position:
            bars_held = i - self.entry_bar if self.entry_bar else 0

            # Exit 1: volatility normalizes (price within 1 std of mean)
            normalized = abs(self.returns[i]) < self.exit_std * vol

            # Exit 2: OBV begins declining
            obv_declining = False
            if i >= 3:
                obv_declining = self.obv[i] < self.obv[i - 3]

            # Exit 3: stop loss hit
            stop_hit = self.data.Low[i] <= self.stop_price if self.stop_price else False

            # Exit 4: time-based
            time_exit = bars_held >= self.max_bars_hold

            if stop_hit:
                print(f"🛑 Stop loss hit at {self.data.Close[i]:.2f} 🌙")
                self.position.close()
                self.entry_bar = None
            elif normalized and bars_held > 2:
                print(f"✅ Volatility normalized exit at {self.data.Close[i]:.2f} ✨")
                self.position.close()
                self.entry_bar = None
            elif obv_declining and bars_held > 2:
                print(f"📉 OBV declining exit at {self.data.Close[i]:.2f} 🌙")
                self.position.close()
                self.entry_bar = None
            elif time_exit:
                print(f"⏰ Time-based exit at {self.data.Close[i]:.2f} 🚀")
                self.position.close()
                self.entry_bar = None

        # ---- ENTRY LOGIC ----
        if not self.position:
            extreme_drop = pct3 < threshold
            if extreme_drop and obv_rising:
                price = self.data.Close[i]
                atr = self.atr[i]

                # Position sizing: 1,000,000 units base, scaled by volatility
                base_size = 1_000_000
                size = base_size

                # Stop loss at 3 std below entry
                stop_distance = self.stop_std * vol * price
                stop_price = price - stop_distance

                # Risk-based cap
                risk_per_unit = price - stop_price
                if risk_per_unit > 0:
                    risk_amount = price * size * self.risk_pct
                    risk_size = risk_amount / risk_per_unit
                    size = min(size, risk_size)

                size = int(round(size))
                if size < 1:
                    size = 1

                self.buy(size=size)
                self.entry_bar = i
                self.entry_price = price
                self.stop_price = stop_price

                print(f"🚀 LONG ENTRY | Price: {price:.2f} | Size: {size} | "
                      f"Pct3: {pct3*100:.2f}% | Threshold: {threshold*100:.2f}% | "
                      f"Stop: {stop_price:.2f} | OBV rising: {obv_rising} 🌙✨")


# Run backtest
bt = Backtest(data, VolatilityAccumulation, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)