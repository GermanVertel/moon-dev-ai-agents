import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

print("🌙✨ CompressionTrailbreak Strategy Loading... 🚀")

data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean column names
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

print(f"🌙 Data loaded: {len(data)} bars 📊")


class CompressionTrailbreak(Strategy):
    bb_period = 20
    bb_std = 2.0
    hv_lookback = 20
    atr_period = 14
    atr_multiple = 2.0
    risk_pct = 0.01
    max_hold_bars = 100

    def init(self):
        # Bollinger Bands
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS, self.data.Close,
            timeperiod=self.bb_period, nbdevup=self.bb_std, nbdevdn=self.bb_std
        )

        # Bollinger Band Width
        def bb_width(upper, middle, lower):
            upper = np.asarray(upper, dtype=float)
            middle = np.asarray(middle, dtype=float)
            lower = np.asarray(lower, dtype=float)
            with np.errstate(divide='ignore', invalid='ignore'):
                out = (upper - lower) / middle
            out[~np.isfinite(out)] = np.nan
            return out
        self.bbw = self.I(bb_width, self.bb_upper, self.bb_middle, self.bb_lower)

        # Historical Volatility (rolling std of log returns)
        def hist_vol(c, lookback):
            c = np.asarray(c, dtype=float)
            log_ret = np.zeros_like(c)
            with np.errstate(divide='ignore', invalid='ignore'):
                log_ret[1:] = np.log(c[1:] / c[:-1])
            log_ret[~np.isfinite(log_ret)] = 0.0
            s = pd.Series(log_ret)
            return s.rolling(lookback).std().values
        self.hv = self.I(hist_vol, self.data.Close, self.hv_lookback)

        # ATR
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close,
                          timeperiod=self.atr_period)

        # SMA slope
        self.sma20 = self.I(talib.SMA, self.data.Close, timeperiod=20)

        # RSI
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=14)

        self.highest_since_entry = None
        self.entry_bar = None

        print("🌙✨ Indicators initialized 🚀")

    def next(self):
        price = self.data.Close[-1]

        # Manage open position
        if self.position:
            # Update trailing stop
            if self.highest_since_entry is None or self.data.High[-1] > self.highest_since_entry:
                self.highest_since_entry = self.data.High[-1]

            trail_stop = self.highest_since_entry - self.atr_multiple * self.atr[-1]

            # Time-based exit
            bars_held = len(self.data) - self.entry_bar
            if bars_held >= self.max_hold_bars:
                print(f"🌙⏰ Time exit after {bars_held} bars at {price:.2f}")
                self.position.close()
                self.highest_since_entry = None
                return

            # Close below middle band exit
            if price < self.bb_middle[-1]:
                print(f"🌙📉 Middle band exit at {price:.2f}")
                self.position.close()
                self.highest_since_entry = None
                return

            # Trailing stop hit
            if price <= trail_stop:
                print(f"🌙🛑 Trail stop hit at {price:.2f} (stop={trail_stop:.2f})")
                self.position.close()
                self.highest_since_entry = None
                return

            return

        # Entry logic - need enough bars
        if len(self.data) < max(self.bb_period, self.hv_lookback, self.atr_period) + 2:
            return

        # Squeeze condition evaluated on PREVIOUS bar
        prev_bbw = self.bbw[-2]
        prev_hv = self.hv[-2]

        if np.isnan(prev_bbw) or np.isnan(prev_hv):
            return

        squeeze_active = prev_bbw < prev_hv

        # Breakout trigger: close above upper band on current bar
        breakout = self.data.Close[-1] > self.bb_upper[-1]

        # Confirmation filters
        sma_rising = self.sma20[-1] > self.sma20[-2]
        rsi_ok = self.rsi[-1] > 50

        if squeeze_active and breakout and sma_rising and rsi_ok:
            entry_price = self.data.Close[-1]
            initial_stop = entry_price - self.atr_multiple * self.atr[-1]
            risk_per_unit = entry_price - initial_stop

            if risk_per_unit <= 0:
                return

            # Fraction-based sizing (0 < size < 1)
            risk_pct_size = (self.risk_pct * entry_price) / risk_per_unit
            if risk_pct_size <= 0:
                return
            if risk_pct_size >= 1:
                risk_pct_size = 0.99

            print(f"🌙🚀 BREAKOUT! Entry={entry_price:.2f} Size(fraction)={risk_pct_size:.4f} "
                  f"ATR={self.atr[-1]:.2f} Stop={initial_stop:.2f}")

            self.buy(size=risk_pct_size)
            self.highest_since_entry = self.data.High[-1]
            self.entry_bar = len(self.data)


bt = Backtest(data, CompressionTrailbreak, cash=1_000_000, commission=0.001)

stats = bt.run()
print(stats)
print(stats._strategy)