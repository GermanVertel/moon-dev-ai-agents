import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# 🌙 Moon Dev's LiquidityVolatility Strategy ✨
print("🌙 Moon Dev initializing LiquidityVolatility backtest... ✨")

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

# Ensure datetime index
if 'datetime' in data.columns:
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')

data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

# 🌙 Ensure all numeric columns are float64 (talib requires double)
data['Open'] = data['Open'].astype(np.float64)
data['High'] = data['High'].astype(np.float64)
data['Low'] = data['Low'].astype(np.float64)
data['Close'] = data['Close'].astype(np.float64)
data['Volume'] = data['Volume'].astype(np.float64)

print(f"🌙 Data loaded: {len(data)} candles ✨")
print(f"🚀 Date range: {data.index[0]} -> {data.index[-1]}")


class LiquidityVolatility(Strategy):
    """
    🌙 LiquidityVolatility Strategy ✨
    """

    cluster_lookback = 20
    proximity_pct = 0.005

    atr_period = 14
    atr_baseline_period = 50
    vol_spike_mult = 2.0

    vol_ma_period = 20
    vol_mult = 1.5

    risk_pct = 0.005
    sl_pct = 0.007
    tp1_pct = 0.015
    tp2_pct = 0.030
    time_stop_bars = 20
    cooldown_bars = 4

    def init(self):
        print("🌙 Initializing indicators... ✨")

        # 🌙 Cast data arrays to float64 for talib
        high = np.asarray(self.data.High, dtype=np.float64)
        low = np.asarray(self.data.Low, dtype=np.float64)
        close = np.asarray(self.data.Close, dtype=np.float64)
        volume = np.asarray(self.data.Volume, dtype=np.float64)

        self.atr = self.I(talib.ATR, high, low, close,
                          timeperiod=self.atr_period, name="ATR")

        self.atr_baseline = self.I(talib.SMA, self.atr, timeperiod=self.atr_baseline_period,
                                   name="ATR_baseline")

        # 🌙 Wrap volume in float64 for SMA
        self.vol_ma = self.I(talib.SMA, pd.Series(volume, dtype=np.float64),
                             timeperiod=self.vol_ma_period, name="Vol_MA")

        self.swing_high = self.I(talib.MAX, high, timeperiod=self.cluster_lookback,
                                 name="SwingHigh")
        self.swing_low = self.I(talib.MIN, low, timeperiod=self.cluster_lookback,
                                name="SwingLow")

        self.entry_price = None
        self.tp1_hit = False
        self.entry_bar = None
        self.last_loss_bar = -9999

        print("🚀 Indicators ready!")

    def next(self):
        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]

        if np.isnan(self.atr[-1]) or np.isnan(self.atr_baseline[-1]) or np.isnan(self.vol_ma[-1]):
            return
        if self.atr_baseline[-1] <= 0 or self.vol_ma[-1] <= 0:
            return

        vol_spike = self.atr[-1] > (self.vol_spike_mult * self.atr_baseline[-1])
        vol_decay = self.atr[-1] < (1.2 * self.atr_baseline[-1])

        volume_conf = self.data.Volume[-1] > (self.vol_mult * self.vol_ma[-1])

        cluster_high = self.swing_high[-2] if len(self.swing_high) > 1 else np.nan
        cluster_low = self.swing_low[-2] if len(self.swing_low) > 1 else np.nan

        if np.isnan(cluster_high) or np.isnan(cluster_low):
            return

        in_cooldown = (len(self.data) - self.last_loss_bar) < self.cooldown_bars

        if self.position:
            bars_held = len(self.data) - self.entry_bar if self.entry_bar else 0

            # 🌙 Use last trade entry price (Position has no entry_price attr)
            try:
                entry_px = self.trades[-1].entry_price
            except Exception:
                entry_px = self.entry_price

            if entry_px is None:
                return

            if self.position.is_long:
                if vol_decay:
                    print(f"🌙 Volatility decay exit (LONG) @ {price:.2f} ✨")
                    self.position.close()
                    self._reset_trade()
                    return

                if bars_held >= self.time_stop_bars:
                    print(f"⏰ Time stop exit (LONG) @ {price:.2f}")
                    self.position.close()
                    self._reset_trade()
                    return

                if not self.tp1_hit and price >= entry_px * (1 + self.tp1_pct):
                    print(f"🎯 TP1 hit (LONG) @ {price:.2f} - closing 50%")
                    self.position.close()
                    self.tp1_hit = True
                    return

                if self.tp1_hit and price >= entry_px * (1 + self.tp2_pct):
                    print(f"🎯 TP2 hit (LONG) @ {price:.2f} - closing rest 🚀")
                    self.position.close()
                    self._reset_trade()
                    return

                if price <= entry_px * (1 - self.sl_pct):
                    print(f"🛑 Hard SL hit (LONG) @ {price:.2f}")
                    self.position.close()
                    self.last_loss_bar = len(self.data)
                    self._reset_trade()
                    return

            elif self.position.is_short:
                if vol_decay:
                    print(f"🌙 Volatility decay exit (SHORT) @ {price:.2f} ✨")
                    self.position.close()
                    self._reset_trade()
                    return

                if bars_held >= self.time_stop_bars:
                    print(f"⏰ Time stop exit (SHORT) @ {price:.2f}")
                    self.position.close()
                    self._reset_trade()
                    return

                if not self.tp1_hit and price <= entry_px * (1 - self.tp1_pct):
                    print(f"🎯 TP1 hit (SHORT) @ {price:.2f} - closing 50%")
                    self.position.close()
                    self.tp1_hit = True
                    return

                if self.tp1_hit and price <= entry_px * (1 - self.tp2_pct):
                    print(f"🎯 TP2 hit (SHORT) @ {price:.2f} - closing rest 🚀")
                    self.position.close()
                    self._reset_trade()
                    return

                if price >= entry_px * (1 + self.sl_pct):
                    print(f"🛑 Hard SL hit (SHORT) @ {price:.2f}")
                    self.position.close()
                    self.last_loss_bar = len(self.data)
                    self._reset_trade()
                    return

            return

        if in_cooldown:
            return

        if not vol_spike or not volume_conf:
            return

        if high > cluster_high and price > cluster_high:
            prev_close = self.data.Close[-2]
            if abs(prev_close - cluster_high) / cluster_high <= self.proximity_pct * 3:
                sl_price = price * (1 - self.sl_pct)
                risk_per_unit = price - sl_price

                if risk_per_unit > 0:
                    risk_amount = self.equity * self.risk_pct
                    size = int(round(risk_amount / risk_per_unit))
                    size = max(1, min(size, 1000000))

                    print(f"🚀 LONG entry @ {price:.2f} | cluster_high={cluster_high:.2f} | "
                          f"ATR_spike={self.atr[-1]/self.atr_baseline[-1]:.2f}x | size={size} 🌙")
                    self.buy(size=size, sl=sl_price)
                    self.entry_price = price
                    self.entry_bar = len(self.data)
                    self.tp1_hit = False

        elif low < cluster_low and price < cluster_low:
            prev_close = self.data.Close[-2]
            if abs(prev_close - cluster_low) / cluster_low <= self.proximity_pct * 3:
                sl_price = price * (1 + self.sl_pct)
                risk_per_unit = sl_price - price

                if risk_per_unit > 0:
                    risk_amount = self.equity * self.risk_pct
                    size = int(round(risk_amount / risk_per_unit))
                    size = max(1, min(size, 1000000))

                    print(f"🔻 SHORT entry @ {price:.2f} | cluster_low={cluster_low:.2f} | "
                          f"ATR_spike={self.atr[-1]/self.atr_baseline[-1]:.2f}x | size={size} 🌙")
                    self.sell(size=size, sl=sl_price)
                    self.entry_price = price
                    self.entry_bar = len(self.data)
                    self.tp1_hit = False

    def _reset_trade(self):
        self.entry_price = None
        self.entry_bar = None
        self.tp1_hit = False


print("🌙 Running backtest... ✨🚀")
bt = Backtest(data, LiquidityVolatility, cash=1_000_000, commission=0.001)

stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙✨ Backtest complete! 🚀")