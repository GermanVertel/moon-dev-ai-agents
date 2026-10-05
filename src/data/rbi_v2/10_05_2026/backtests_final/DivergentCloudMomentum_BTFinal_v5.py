import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Load data
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper column mapping
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

print("🌙✨ Moon Dev Data Loaded! ✨🌙")
print(f"📊 Rows: {len(data)}")
print(f"🚀 Columns: {list(data.columns)}")


class DivergentCloudMomentum(Strategy):
    bb_period = 20
    bb_std = 2.0
    tenkan_period = 9
    kijun_period = 26
    senkou_b_period = 52
    atr_period = 14
    rsi_period = 14
    risk_pct = 0.02
    time_stop_bars = 15

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low

        # Bollinger Bands
        self.bb_mid = self.I(talib.SMA, close, timeperiod=self.bb_period, name='BB_Mid')
        self.bb_upper, self.bb_mid2, self.bb_lower = self.I(
            talib.BBANDS, close, timeperiod=self.bb_period,
            nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0,
            name='BBANDS'
        )

        # %B and BandWidth
        def pct_b(c, lower, upper):
            denom = upper - lower
            denom = np.where(denom == 0, np.nan, denom)
            return (c - lower) / denom

        self.pct_b = self.I(pct_b, close, self.bb_lower, self.bb_upper, name='%B')

        def band_width(upper, lower, mid):
            mid_safe = np.where(mid == 0, np.nan, mid)
            return (upper - lower) / mid_safe

        self.bandwidth = self.I(band_width, self.bb_upper, self.bb_lower, self.bb_mid, name='BandWidth')

        # Ichimoku
        self.tenkan = self.I(talib.SMA, (high + low) / 2, timeperiod=self.tenkan_period, name='Tenkan')
        self.kijun = self.I(talib.SMA, (high + low) / 2, timeperiod=self.kijun_period, name='Kijun')
        self.senkou_a = self.I(lambda t, k: (t + k) / 2, self.tenkan, self.kijun, name='SenkouA')
        self.senkou_b = self.I(talib.SMA, (high + low) / 2, timeperiod=self.senkou_b_period, name='SenkouB')

        # ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period, name='ATR')

        # RSI
        self.rsi = self.I(talib.RSI, close, timeperiod=self.rsi_period, name='RSI')

        # Chikou (lagging) - close shifted forward 26
        def chikou_fn(c):
            s = pd.Series(c).shift(self.kijun_period)
            return s.values
        self.chikou = self.I(chikou_fn, close, name='Chikou')

        self.entry_bar = None
        self.entry_price = None
        self.stop_price = None

    def next(self):
        price = self.data.Close[-1]

        # Need enough history
        if len(self.data) < self.senkou_b_period + self.kijun_period + 5:
            return

        # Cloud boundaries (current displaced values)
        cloud_top = max(self.senkou_a[-1], self.senkou_b[-1])
        cloud_bot = min(self.senkou_a[-1], self.senkou_b[-1])

        above_cloud = price > cloud_top
        inside_cloud = cloud_bot <= price <= cloud_top
        below_cloud = price < cloud_bot

        pb = self.pct_b[-1]
        bw = self.bandwidth[-1]
        bw_series = np.array(self.bandwidth)
        bw_valid = bw_series[~np.isnan(bw_series)]
        bw_expanding = False
        bw_ok = True
        if len(bw_valid) > 5:
            bw_expanding = bw_valid[-1] > bw_valid[-2]
        if len(bw_valid) >= 100:
            recent = bw_valid[-100:]
            low10 = np.nanpercentile(recent, 10)
            if bw_valid[-1] < low10:
                bw_ok = False

        rsi = self.rsi[-1]
        tenkan = self.tenkan[-1]
        kijun = self.kijun[-1]
        tenkan_prev = self.tenkan[-2]
        kijun_prev = self.kijun[-2]

        # Chikou above price 26 ago
        chikou_val = self.chikou[-1]
        price_26_ago = self.data.Close[-self.kijun_period - 1] if len(self.data) > self.kijun_period else np.nan
        chikou_above = (not np.isnan(chikou_val)) and (not np.isnan(price_26_ago)) and (chikou_val > price_26_ago)

        # === EXIT LOGIC (for short positions) ===
        if self.position:
            # Primary exit: %B > 1 (bullish momentum)
            if pb > 1:
                print(f"🌙✨ EXIT SIGNAL: %B > 1 (bullish reversal) at {price:.2f} 🚀")
                self.position.close()
                self.entry_bar = None
                return

            # Secondary: Tenkan crosses above Kijun
            if tenkan_prev <= kijun_prev and tenkan > kijun:
                print(f"🌙✨ EXIT SIGNAL: Tenkan crossed above Kijun at {price:.2f} 🚀")
                self.position.close()
                self.entry_bar = None
                return

            # Profit target: mid BB or Kijun (for shorts, target is BELOW price)
            target = max(self.bb_mid[-1], kijun)
            if price <= target:
                print(f"🌙✨ EXIT SIGNAL: Profit target hit at {price:.2f} 🚀")
                self.position.close()
                self.entry_bar = None
                return

            # Time stop
            if self.entry_bar is not None and (len(self.data) - self.entry_bar) >= self.time_stop_bars:
                print(f"🌙⏰ EXIT SIGNAL: Time stop after {self.time_stop_bars} bars at {price:.2f}")
                self.position.close()
                self.entry_bar = None
                return

            # Trailing: once price re-enters bands from below, trail at lower band
            if pb > 0 and self.stop_price is not None and self.bb_lower[-1] > self.stop_price:
                self.stop_price = self.bb_lower[-1]
                print(f"🌙📈 Trailing stop updated to {self.stop_price:.2f}")

            # Stop loss check
            if self.stop_price is not None and price >= self.stop_price:
                print(f"🌙🛑 STOP LOSS hit at {price:.2f} (stop: {self.stop_price:.2f})")
                self.position.close()
                self.entry_bar = None
                return

        # === ENTRY LOGIC (short entries on bearish momentum) ===
        if not self.position:
            if not bw_ok:
                return

            bearish_momentum = pb < 0

            # Trigger A: price below lower band AND above cloud
            trigger_a = bearish_momentum and above_cloud

            # Trigger B: price below lower band AND Tenkan crosses below Kijun but both above cloud
            tenkan_cross_down = tenkan_prev >= kijun_prev and tenkan < kijun
            trigger_b = bearish_momentum and tenkan_cross_down and (tenkan > cloud_top) and (kijun > cloud_top)

            if trigger_a or trigger_b:
                # Confirmation filters (need 2 of 3)
                confirmations = 0
                if bw_expanding:
                    confirmations += 1
                if chikou_above:
                    confirmations += 1
                if rsi < 35:
                    confirmations += 1

                if confirmations >= 2:
                    # Risk management
                    atr = self.atr[-1]
                    stop = self.bb_lower[-1] + 1.5 * atr
                    risk_per_unit = stop - price

                    if risk_per_unit <= 0:
                        return

                    equity = self.equity
                    risk_amount = equity * self.risk_pct
                    size = risk_amount / risk_per_unit
                    size = int(round(size))

                    if size > 0:
                        # Cap size
                        size = min(size, 1000000)
                        self.sell(size=size)
                        self.entry_bar = len(self.data)
                        self.entry_price = price
                        self.stop_price = stop

                        print(f"🌙🚀 SHORT ENTRY! Price: {price:.2f} | Size: {size} | Stop: {stop:.2f}")
                        print(f"   📊 %B: {pb:.3f} | RSI: {rsi:.2f} | BW Expanding: {bw_expanding}")
                        print(f"   ☁️ Cloud Top: {cloud_top:.2f} | Confirmations: {confirmations}/3")


bt = Backtest(data, DivergentCloudMomentum, cash=1000000, commission=0.002)
stats = bt.run()
print(stats)
print(stats._strategy)