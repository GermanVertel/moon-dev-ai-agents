import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

print("🌙✨ CorrelatedSpreadArb Backtest Initializing... ✨🌙")

# Load data
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')

# Clean columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper case mapping
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

# Set datetime index
if 'datetime' in data.columns:
    data['datetime'] = pd.to_datetime(data['datetime'])
    data.set_index('datetime', inplace=True)

print(f"🚀 Data loaded: {len(data)} bars")
print(f"🌙 Columns: {list(data.columns)}")


class CorrelatedSpreadArb(Strategy):
    """
    Correlated Spread Arbitrage Strategy
    Uses synthetic spread proxy from High/Low range as bid-ask proxy
    since true bid/ask data is not available in OHLCV.
    """
    # Parameters
    zscore_lookback = 80
    corr_lookback = 90
    z_entry = 2.0
    z_exit = 0.5
    z_stop = 3.5
    corr_threshold = 0.7
    vol_cap = 0.03  # realized vol cap
    max_hold_bars = 15
    risk_pct = 0.02

    def init(self):
        print("🌙 Initializing CorrelatedSpreadArb indicators...")

        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)

        # Synthetic bid-ask spread proxy (High - Low) normalized by close
        spread_proxy = (high - low) / close

        # Leg A = current bar spread, Leg B = smoothed/rolling mean spread (correlated "pair")
        self.spread_a = self.I(lambda x: pd.Series(x).values, spread_proxy.values,
                               name='SpreadA')
        self.spread_b = self.I(talib.SMA, spread_proxy.values,
                               timeperiod=20, name='SpreadB')

        # Spread differential
        spread_diff = spread_proxy.values - self.spread_b

        # Rolling correlation between close and volume (proxy for pair correlation)
        def rolling_corr(c, v, window):
            cs = pd.Series(c)
            vs = pd.Series(v)
            return cs.rolling(window).corr(vs).fillna(0).values

        self.corr = self.I(rolling_corr, self.data.Close, self.data.Volume,
                           self.corr_lookback, name='RollingCorr')

        # Z-score of spread differential
        def zscore(x, window):
            s = pd.Series(x)
            mean = s.rolling(window).mean()
            std = s.rolling(window).std()
            z = (s - mean) / std.replace(0, np.nan)
            return z.fillna(0).values

        self.zscore = self.I(zscore, spread_diff, self.zscore_lookback, name='ZScore')

        # Realized volatility (ATR proxy)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low,
                          self.data.Close, timeperiod=14, name='ATR')
        self.atr_pct = self.I(lambda a, c: (a / c), self.atr, self.data.Close,
                              name='ATR_Pct')

        # Volatility filter
        self.vol_ok = self.I(lambda x: (x < self.vol_cap).astype(float),
                             self.atr_pct, name='VolOK')

        # Track entry bar for time stop
        self.entry_bar = None

        print("✨ Indicators ready! 🌙")

    def next(self):
        price = self.data.Close[-1]
        z = self.zscore[-1]
        corr = self.corr[-1]
        vol_ok = self.vol_ok[-1]
        atr = self.atr[-1]

        # Skip warmup
        if len(self.data) < max(self.zscore_lookback, self.corr_lookback) + 5:
            return

        # ===== EXIT LOGIC =====
        if self.position:
            bars_held = len(self.data) - self.entry_bar if self.entry_bar else 0

            # Primary exit: z-score reverted
            if abs(z) < self.z_exit:
                print(f"🌙✨ EXIT: Z-score reverted to {z:.3f} | Price: {price:.2f}")
                self.position.close()
                self.entry_bar = None
                return

            # Stop loss: z-score worsened
            if z > self.z_stop:
                print(f"🛑 STOP LOSS: Z-score blew out to {z:.3f} | Price: {price:.2f}")
                self.position.close()
                self.entry_bar = None
                return

            # Correlation breakdown
            if corr < self.corr_threshold:
                print(f"⚠️ CORRELATION BREAK: r={corr:.3f} | Price: {price:.2f}")
                self.position.close()
                self.entry_bar = None
                return

            # Time stop
            if bars_held >= self.max_hold_bars:
                print(f"⏰ TIME STOP: {bars_held} bars | Price: {price:.2f}")
                self.position.close()
                self.entry_bar = None
                return

            return

        # ===== ENTRY LOGIC =====
        # Long entry: z-score > +2.0 (Leg A spread abnormally wide)
        if z > self.z_entry and corr > self.corr_threshold and vol_ok > 0:
            # Risk-based position sizing
            risk_amount = self.equity * self.risk_pct
            stop_distance = atr * 2 if atr > 0 else price * 0.01
            position_size = int(round(risk_amount / stop_distance)) if stop_distance > 0 else 0

            # Cap position size
            max_size = int(self.equity * 0.95 / price)
            position_size = min(position_size, max_size)

            if position_size > 0:
                print(f"🚀🌙 LONG ENTRY: Z={z:.3f} | Corr={corr:.3f} | "
                      f"Size={position_size} | Price={price:.2f}")
                self.buy(size=position_size)
                self.entry_bar = len(self.data)

        # Short entry: z-score < -2.0 (Leg A spread abnormally tight)
        elif z < -self.z_entry and corr > self.corr_threshold and vol_ok > 0:
            risk_amount = self.equity * self.risk_pct
            stop_distance = atr * 2 if atr > 0 else price * 0.01
            position_size = int(round(risk_amount / stop_distance)) if stop_distance > 0 else 0

            max_size = int(self.equity * 0.95 / price)
            position_size = min(position_size, max_size)

            if position_size > 0:
                print(f"🔻🌙 SHORT ENTRY: Z={z:.3f} | Corr={corr:.3f} | "
                      f"Size={position_size} | Price={price:.2f}")
                self.sell(size=position_size)
                self.entry_bar = len(self.data)


print("🌙🚀 Running CorrelatedSpreadArb Backtest...")
bt = Backtest(data, CorrelatedSpreadArb, cash=1_000_000, commission=0.0002)
stats = bt.run()
print(stats)
print(stats._strategy)
print("✨🌙 Backtest complete! Moon Dev out! 🚀🌙")