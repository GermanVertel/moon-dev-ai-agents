import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# 🌙 Moon Dev's LiquidSpreadArb Backtest 🚀

data_path = '/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv'

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

print("🌙✨ Data loaded successfully! Shape:", data.shape)
print("🚀 First few rows:\n", data.head())


class LiquidSpreadArb(Strategy):
    # Strategy parameters
    spread_lookback = 20
    spread_k = 1.5          # std multiplier for spread dislocation
    vol_imbalance_thresh = 0.3
    atr_period = 14
    atr_max_mult = 3.0      # volatility filter
    risk_pct = 0.005        # 0.5% of capital per trade
    stop_mult = 2.0         # stop loss at 2x entry spread
    max_hold_bars = 20      # time stop
    size = 1_000_000        # requested position size

    def init(self):
        # 🌙 Approximate bid/ask spread proxy from OHLC (since we lack L1 book)
        # spread_proxy = High - Low (intrabar range) — captures transient dislocation
        high = self.data.High
        low = self.data.Low
        close = self.data.Close
        volume = self.data.Volume

        self.spread = self.I(lambda h, l: h - l, high, low, name='SpreadProxy')
        self.spread_ma = self.I(talib.SMA, self.spread, timeperiod=self.spread_lookback, name='SpreadMA')
        self.spread_std = self.I(talib.STDDEV, self.spread, timeperiod=self.spread_lookback, name='SpreadSTD')

        # Volume imbalance proxy: use signed volume via close-open direction
        def vol_imbalance(c, o, v):
            direction = np.sign(c - o)
            signed = direction * v
            # rolling window sum
            s = pd.Series(signed).rolling(4).sum().values
            tot = pd.Series(v).rolling(4).sum().values
            with np.errstate(divide='ignore', invalid='ignore'):
                imb = np.where(tot > 0, s / tot, 0.0)
            return imb

        self.vol_imb = self.I(vol_imbalance, close, self.data.Open, volume, name='VolImbalance')

        # ATR for volatility filter
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close,
                          timeperiod=self.atr_period, name='ATR')
        # ATR baseline (SMA of ATR)
        self.atr_ma = self.I(talib.SMA, self.atr, timeperiod=self.spread_lookback, name='ATR_MA')

        # Track elevated spread counter for 2-consecutive confirmation
        self.elevated_count = 0
        self.entry_spread = None
        self.entry_bar = None

        print("🌙✨ LiquidSpreadArb indicators initialized! 🚀")

    def next(self):
        price = self.data.Close[-1]
        spread_now = self.spread[-1]
        spread_mean = self.spread_ma[-1]
        spread_sd = self.spread_std[-1]
        imb = self.vol_imb[-1]
        atr_now = self.atr[-1]
        atr_avg = self.atr_ma[-1]

        # Guard against NaNs
        if (np.isnan(spread_now) or np.isnan(spread_mean) or np.isnan(spread_sd)
                or np.isnan(imb) or np.isnan(atr_now) or np.isnan(atr_avg)):
            return

        # 🌙 Volatility filter — avoid extreme volatility regimes
        vol_ok = atr_now < self.atr_max_mult * atr_avg

        # Detect elevated spread
        threshold = spread_mean + self.spread_k * spread_sd
        spread_elevated = spread_now > threshold

        if spread_elevated and vol_ok:
            self.elevated_count += 1
        else:
            self.elevated_count = 0

        # 🌙 Manage open positions first
        if self.position:
            bars_held = len(self.data) - self.entry_bar if self.entry_bar else 0

            # Profit target: spread reverted to baseline
            if spread_now <= spread_mean:
                print(f"🌙✨ [EXIT-PROFIT] Spread reverted! spread={spread_now:.2f} <= baseline={spread_mean:.2f} @ {price:.2f}")
                self.position.close()
                self.entry_spread = None
                self.entry_bar = None
                return

            # Stop loss: spread widened beyond 2x entry spread
            if self.entry_spread and spread_now > self.stop_mult * self.entry_spread:
                print(f"🚨 [EXIT-STOP] Spread widened! now={spread_now:.2f} > 2x entry={self.entry_spread:.2f} @ {price:.2f}")
                self.position.close()
                self.entry_spread = None
                self.entry_bar = None
                return

            # Time stop
            if bars_held >= self.max_hold_bars:
                print(f"⏰ [EXIT-TIME] Max hold reached ({bars_held} bars) @ {price:.2f}")
                self.position.close()
                self.entry_spread = None
                self.entry_bar = None
                return

            return  # don't stack new positions while managing one

        # 🌙 Entry logic — require 2 consecutive elevated spread bars
        if self.elevated_count >= 2:
            # Long entry: sell-side imbalance (oversupply → expect reversion up)
            if imb < -self.vol_imbalance_thresh:
                size = int(round(self.size))
                print(f"🌙🚀 [LONG ENTRY] Spread={spread_now:.2f} > thr={threshold:.2f}, "
                      f"VolImb={imb:.3f} @ {price:.2f} | size={size}")
                self.buy(size=size)
                self.entry_spread = spread_now
                self.entry_bar = len(self.data)

            # Short entry: buy-side imbalance (demand spike → expect reversion down)
            elif imb > self.vol_imbalance_thresh:
                size = int(round(self.size))
                print(f"🌙🚀 [SHORT ENTRY] Spread={spread_now:.2f} > thr={threshold:.2f}, "
                      f"VolImb={imb:.3f} @ {price:.2f} | size={size}")
                self.sell(size=size)
                self.entry_spread = spread_now
                self.entry_bar = len(self.data)


# 🌙 Run the backtest
bt = Backtest(data, LiquidSpreadArb, cash=1_000_000, commission=0.0002, exclusive_orders=True)
stats = bt.run()
print(stats)
print(stats._strategy)