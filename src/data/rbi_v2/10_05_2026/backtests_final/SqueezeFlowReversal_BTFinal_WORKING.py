import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Load data
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

# Set datetime index
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')

print("🌙 Moon Dev SqueezeFlow Reversal Backtest Initializing... ✨")
print(f"📊 Data loaded: {len(data)} bars")
print(f"🚀 Starting backtest...")


class SqueezeFlowReversal(Strategy):
    # Parameters
    bb_period = 20
    bb_std = 2.0
    cmf_period = 20
    bw_lookback = 20
    bw_sma_exit = 10
    trend_sma = 200
    trailing_stop_pct = 0.05
    risk_pct = 0.02
    time_stop_bars = 25
    use_trend_filter = True

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        # Bollinger Bands
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS, close, timeperiod=self.bb_period,
            nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0
        )

        # Bollinger Bandwidth
        def bandwidth(upper, middle, lower):
            upper = np.asarray(upper, dtype=float)
            middle = np.asarray(middle, dtype=float)
            lower = np.asarray(lower, dtype=float)
            with np.errstate(divide='ignore', invalid='ignore'):
                bw = (upper - lower) / middle
            bw = np.where(np.isfinite(bw), bw, 0.0)
            return bw

        self.bw = self.I(bandwidth, self.bb_upper, self.bb_middle, self.bb_lower)

        # 20-day lowest bandwidth
        self.bw_min = self.I(talib.MIN, self.bw, timeperiod=self.bw_lookback)

        # 10-day SMA of bandwidth for exit
        self.bw_sma = self.I(talib.SMA, self.bw, timeperiod=self.bw_sma_exit)

        # Chaikin Money Flow
        def cmf(high, low, close, volume, period):
            high = np.asarray(high, dtype=float)
            low = np.asarray(low, dtype=float)
            close = np.asarray(close, dtype=float)
            volume = np.asarray(volume, dtype=float)

            with np.errstate(divide='ignore', invalid='ignore'):
                denom = high - low
                mfm = ((close - low) - (high - close)) / denom
            mfm = np.where(np.isfinite(mfm), mfm, 0.0)
            mfv = mfm * volume

            mfv_series = pd.Series(mfv)
            vol_series = pd.Series(volume)
            result = mfv_series.rolling(period).sum() / vol_series.rolling(period).sum()
            result = result.replace([np.inf, -np.inf], 0).fillna(0)
            return result.values

        self.cmf = self.I(cmf, high, low, close, volume, self.cmf_period)

        # 200-period SMA trend filter
        self.sma200 = self.I(talib.SMA, close, timeperiod=self.trend_sma)

        # State
        self.entry_price = None
        self.peak_price = None
        self.bars_in_trade = 0

        print("🌙 Indicators initialized: BB, Bandwidth, CMF, SMA200 ✨")

    def next(self):
        price = self.data.Close[-1]

        # Manage open position
        if self.position:
            self.bars_in_trade += 1
            self.peak_price = max(self.peak_price, self.data.High[-1])

            # Trailing stop (5% below peak)
            trailing_stop = self.peak_price * (1 - self.trailing_stop_pct)
            if price <= trailing_stop:
                print(f"🛑 Trailing stop hit at {price:.2f} (peak {self.peak_price:.2f}) 🌙")
                self.position.close()
                self._reset()
                return

            # Volatility expansion exit
            if not np.isnan(self.bw_sma[-1]) and self.bw[-1] > self.bw_sma[-1]:
                print(f"📈 Bandwidth expansion exit at {price:.2f} (BW {self.bw[-1]:.4f} > SMA {self.bw_sma[-1]:.4f}) ✨")
                self.position.close()
                self._reset()
                return

            # Time stop
            if self.bars_in_trade >= self.time_stop_bars:
                print(f"⏰ Time stop exit at {price:.2f} after {self.bars_in_trade} bars 🚀")
                self.position.close()
                self._reset()
                return

            return

        # Entry logic
        if len(self.data) < max(self.bw_lookback, self.cmf_period, self.trend_sma) + 5:
            return

        # Check for NaN
        if np.isnan(self.bw[-1]) or np.isnan(self.bw_min[-1]) or np.isnan(self.cmf[-1]) or np.isnan(self.cmf[-2]):
            return

        # Condition 1: Bandwidth at 20-day low (squeeze)
        squeeze = self.bw[-1] <= self.bw_min[-1] * 1.001  # small tolerance

        # Condition 2: CMF crosses above zero (manual crossover - no backtesting.lib)
        cmf_cross = self.cmf[-2] <= 0 and self.cmf[-1] > 0

        # Condition 3: Trend filter
        trend_ok = True
        if self.use_trend_filter:
            if np.isnan(self.sma200[-1]):
                trend_ok = False
            else:
                trend_ok = price > self.sma200[-1]

        if squeeze and cmf_cross and trend_ok:
            # Position sizing based on risk
            stop_price = price * (1 - self.trailing_stop_pct)
            risk_per_unit = price - stop_price
            risk_amount = self.equity * self.risk_pct
            position_size = int(round(risk_amount / risk_per_unit)) if risk_per_unit > 0 else 1
            position_size = max(1, min(position_size, int(self.equity / price)))

            print(f"🌙✨ SQUEEZE + CMF CROSS detected! Entering LONG at {price:.2f}")
            print(f"   BW: {self.bw[-1]:.4f} | BW 20d low: {self.bw_min[-1]:.4f}")
            print(f"   CMF: {self.cmf[-2]:.4f} → {self.cmf[-1]:.4f}")
            print(f"   Size: {position_size} | Risk: {self.risk_pct*100:.1f}% 🚀")

            self.buy(size=position_size)
            self.entry_price = price
            self.peak_price = price
            self.bars_in_trade = 0

    def _reset(self):
        self.entry_price = None
        self.peak_price = None
        self.bars_in_trade = 0


bt = Backtest(
    data,
    SqueezeFlowReversal,
    cash=1_000_000,
    commission=0.001,
    exclusive_orders=True
)

stats = bt.run()
print(stats)
print(stats._strategy)