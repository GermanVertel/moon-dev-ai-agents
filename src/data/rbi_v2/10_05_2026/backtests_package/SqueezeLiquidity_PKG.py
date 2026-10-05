import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Load data
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
data = data.dropna()

print("🌙✨ Moon Dev Backtest AI - SqueezeLiquidity Strategy ✨🌙")
print(f"📊 Data loaded: {len(data)} bars")
print(f"📅 Range: {data.index[0]} to {data.index[-1]}")


class SqueezeLiquidity(Strategy):
    bb_period = 20
    bb_std = 2.0
    bbw_lookback = 100
    bbw_threshold = 10  # percentile
    atr_period = 14
    risk_pct = 0.01
    time_stop_bars = 60
    trail_atr_mult = 2.0

    def init(self):
        close = pd.Series(self.data.Close, index=self.data.index)
        high = pd.Series(self.data.High, index=self.data.index)
        low = pd.Series(self.data.Low, index=self.data.index)

        # Bollinger Bands - use talib directly with wrapped lambda to split outputs
        def bb_upper(x):
            u, m, l = talib.BBANDS(x, timeperiod=self.bb_period,
                                   nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0)
            return u

        def bb_middle(x):
            u, m, l = talib.BBANDS(x, timeperiod=self.bb_period,
                                   nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0)
            return m

        def bb_lower(x):
            u, m, l = talib.BBANDS(x, timeperiod=self.bb_period,
                                   nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0)
            return l

        self.bb_upper = self.I(bb_upper, close)
        self.bb_middle = self.I(bb_middle, close)
        self.bb_lower = self.I(bb_lower, close)

        # Bollinger Bandwidth
        def bbw_fn(upper, middle, lower):
            return (upper - lower) / middle

        self.bbw = self.I(bbw_fn, self.bb_upper, self.bb_middle, self.bb_lower)

        # Rolling percentile rank of BBW
        def pct_rank(arr):
            out = np.full(len(arr), np.nan)
            for i in range(self.bbw_lookback, len(arr)):
                window = arr[i - self.bbw_lookback:i + 1]
                if not np.isnan(window).any():
                    out[i] = (window < arr[i]).sum() / len(window) * 100
            return out

        self.bbw_pct = self.I(pct_rank, self.bbw)

        # ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        # Volume SMA for confirmation
        self.vol_sma = self.I(talib.SMA, pd.Series(self.data.Volume, index=self.data.index), timeperiod=20)

        self.entry_bar = 0
        self.entry_price = 0
        self.stop_price = 0
        self.trail_stop = 0
        self.direction = 0

        print("🌙 Indicators initialized: BB, BBW percentile, ATR, Volume SMA")

    def next(self):
        i = len(self.data) - 1
        if i < self.bbw_lookback + 5:
            return

        price = self.data.Close[-1]
        bbw_pct = self.bbw_pct[-1]
        upper = self.bb_upper[-1]
        lower = self.bb_lower[-1]
        middle = self.bb_middle[-1]
        atr = self.atr[-1]

        if np.isnan(bbw_pct) or np.isnan(atr) or atr <= 0:
            return

        # ===== Manage open position =====
        if self.position:
            bars_held = i - self.entry_bar

            if self.direction == 1:
                # Mean reversion exit
                if price >= middle:
                    print(f"🌙✨ LONG mean-reversion exit @ {price:.2f} (middle={middle:.2f})")
                    self.position.close()
                    return

                # Trailing stop
                new_trail = price - self.trail_atr_mult * atr
                if new_trail > self.trail_stop:
                    self.trail_stop = new_trail
                if price <= self.trail_stop:
                    print(f"🚀 LONG trailing stop hit @ {price:.2f}")
                    self.position.close()
                    return

                if price <= self.stop_price:
                    print(f"❌ LONG hard stop @ {price:.2f}")
                    self.position.close()
                    return

            elif self.direction == -1:
                if price <= middle:
                    print(f"🌙✨ SHORT mean-reversion exit @ {price:.2f} (middle={middle:.2f})")
                    self.position.close()
                    return

                new_trail = price + self.trail_atr_mult * atr
                if new_trail < self.trail_stop:
                    self.trail_stop = new_trail
                if price >= self.trail_stop:
                    print(f"🚀 SHORT trailing stop hit @ {price:.2f}")
                    self.position.close()
                    return

                if price >= self.stop_price:
                    print(f"❌ SHORT hard stop @ {price:.2f}")
                    self.position.close()
                    return

            # Time stop
            if bars_held >= self.time_stop_bars:
                print(f"⏰ Time stop after {bars_held} bars @ {price:.2f}")
                self.position.close()
                return

            return

        # ===== Entry logic =====
        # Regime kill switch
        if bbw_pct >= 50:
            return

        # Squeeze gate
        if bbw_pct >= self.bbw_threshold:
            return

        # Volume confirmation
        volume_rising = self.data.Volume[-1] > self.vol_sma[-1]

        # Long entry
        if price > upper and volume_rising:
            # Stop beyond opposite band or 2*ATR, whichever is wider
            stop = min(lower, price - 2 * atr)
            risk = price - stop
            if risk <= 0:
                return
            size = int(round((self.equity * self.risk_pct) / risk))
            if size <= 0:
                return
            print(f"🌙🚀 LONG ENTRY @ {price:.2f} | BBW_pct={bbw_pct:.1f} | upper={upper:.2f} | stop={stop:.2f} | size={size}")
            self.buy(size=size)
            self.entry_bar = i
            self.entry_price = price
            self.stop_price = stop
            self.trail_stop = stop
            self.direction = 1

        # Short entry
        elif price < lower and volume_rising:
            stop = max(upper, price + 2 * atr)
            risk = stop - price
            if risk <= 0:
                return
            size = int(round((self.equity * self.risk_pct) / risk))
            if size <= 0:
                return
            print(f"🌙🚀 SHORT ENTRY @ {price:.2f} | BBW_pct={bbw_pct:.1f} | lower={lower:.2f} | stop={stop:.2f} | size={size}")
            self.sell(size=size)
            self.entry_bar = i
            self.entry_price = price
            self.stop_price = stop
            self.trail_stop = stop
            self.direction = -1


print("🌙 Setting up backtest...")
bt = Backtest(data, SqueezeLiquidity, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)