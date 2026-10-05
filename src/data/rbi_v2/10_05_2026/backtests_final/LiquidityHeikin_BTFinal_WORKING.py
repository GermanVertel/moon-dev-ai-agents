import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

print("🌙✨🚀 Moon Dev's LiquidityHeikin Backtest Initializing... ✨🌙🚀")

# Load data
data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper column mapping
data.columns = [col.capitalize() for col in data.columns]

# Ensure datetime index
if 'Datetime' in data.columns:
    data['Datetime'] = pd.to_datetime(data['Datetime'])
    data = data.set_index('Datetime')

print(f"🌙 Data loaded: {len(data)} rows, columns: {list(data.columns)}")

# Resample to 4H for HA and ATR
data_4h = data.resample('4H').agg({
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'Close': 'last',
    'Volume': 'sum'
}).dropna()

print(f"🌙 4H data: {len(data_4h)} rows")


class LiquidityHeikin(Strategy):
    """
    Moon Dev's LiquidityHeikin Strategy 🌙
    - 4H Heikin-Ashi flip detection
    - ATR volatility filter (current ATR < 20-period avg ATR)
    - Swing high/low pivots for stops
    - Liquidation cluster proxy: recent swing extremes
    """

    # Parameters
    atr_period = 14
    atr_avg_period = 20
    swing_window = 5
    risk_pct = 0.01
    rr_target = 2.0
    time_stop_bars = 5

    def init(self):
        print("🌙 Initializing indicators...")

        # Heikin-Ashi calculation on 4H data
        close_4h = pd.Series(data_4h['Close'].values, index=data_4h.index)
        open_4h = pd.Series(data_4h['Open'].values, index=data_4h.index)
        high_4h = pd.Series(data_4h['High'].values, index=data_4h.index)
        low_4h = pd.Series(data_4h['Low'].values, index=data_4h.index)

        ha_close = (open_4h + high_4h + low_4h + close_4h) / 4.0
        ha_open = ha_close.copy()
        for i in range(1, len(ha_open)):
            ha_open.iloc[i] = (ha_open.iloc[i - 1] + ha_close.iloc[i - 1]) / 2.0
        ha_high = pd.concat([high_4h, ha_open, ha_close], axis=1).max(axis=1)
        ha_low = pd.concat([low_4h, ha_open, ha_close], axis=1).min(axis=1)

        # Reindex to original data index
        ha_open_full = ha_open.reindex(data.index, method='ffill')
        ha_close_full = ha_close.reindex(data.index, method='ffill')
        ha_high_full = ha_high.reindex(data.index, method='ffill')
        ha_low_full = ha_low.reindex(data.index, method='ffill')

        # HA color: 1 = bullish (close > open), -1 = bearish
        ha_color = np.where(ha_close_full > ha_open_full, 1, -1)
        self.ha_color = self.I(lambda: ha_color, name='HA_Color')
        self.ha_close = self.I(lambda: ha_close_full.values, name='HA_Close')
        self.ha_open = self.I(lambda: ha_open_full.values, name='HA_Open')
        self.ha_high = self.I(lambda: ha_high_full.values, name='HA_High')
        self.ha_low = self.I(lambda: ha_low_full.values, name='HA_Low')

        # ATR on 4H data
        atr_4h = talib.ATR(high_4h.values, low_4h.values, close_4h.values, timeperiod=self.atr_period)
        atr_avg_4h = talib.SMA(atr_4h, timeperiod=self.atr_avg_period)
        atr_full = pd.Series(atr_4h, index=data_4h.index).reindex(data.index, method='ffill')
        atr_avg_full = pd.Series(atr_avg_4h, index=data_4h.index).reindex(data.index, method='ffill')

        self.atr = self.I(lambda: atr_full.values, name='ATR')
        self.atr_avg = self.I(lambda: atr_avg_full.values, name='ATR_Avg')

        # Swing highs/lows on original data
        swing_high = talib.MAX(self.data.High, timeperiod=self.swing_window)
        swing_low = talib.MIN(self.data.Low, timeperiod=self.swing_window)
        self.swing_high = self.I(lambda: swing_high, name='Swing_High')
        self.swing_low = self.I(lambda: swing_low, name='Swing_Low')

        print("🌙✨ Indicators ready! 🚀")

    def next(self):
        # Need enough bars
        if len(self.data) < self.swing_window + 5:
            return

        # Skip if already in position
        if self.position:
            self._manage_position()
            return

        price = self.data.Close[-1]
        ha_color = self.ha_color[-1]
        ha_color_prev = self.ha_color[-2]
        atr = self.atr[-1]
        atr_avg = self.atr_avg[-1]

        if np.isnan(atr) or np.isnan(atr_avg) or atr <= 0:
            return

        # Volatility filter: current ATR < average ATR
        low_vol = atr < atr_avg

        # Swing levels
        swing_low = self.swing_low[-1]
        swing_high = self.swing_high[-1]

        # Liquidation cluster proxy: price near recent swing extremes
        near_long_cluster = (price - swing_low) < 0.5 * atr
        near_short_cluster = (swing_high - price) < 0.5 * atr

        # HA flip detection (no backtesting.lib crossover)
        ha_flip_bull = (ha_color == 1) and (ha_color_prev == -1)
        ha_flip_bear = (ha_color == -1) and (ha_color_prev == 1)

        # === LONG ENTRY ===
        if ha_flip_bull and low_vol and near_long_cluster:
            sl = swing_low - 0.5 * atr
            risk = price - sl
            if risk > 0:
                tp = price + self.rr_target * risk
                size = int(round(1_000_000 / price))
                if size > 0:
                    print(f"🌙🚀 LONG ENTRY! Price={price:.2f} SL={sl:.2f} TP={tp:.2f} Size={size} HA_FLIP_BULL 📈")
                    self.buy(size=size, sl=sl, tp=tp)
                    self._entry_bar = len(self.data)

        # === SHORT ENTRY ===
        elif ha_flip_bear and low_vol and near_short_cluster:
            sl = swing_high + 0.5 * atr
            risk = sl - price
            if risk > 0:
                tp = price - self.rr_target * risk
                size = int(round(1_000_000 / price))
                if size > 0:
                    print(f"🌙🚀 SHORT ENTRY! Price={price:.2f} SL={sl:.2f} TP={tp:.2f} Size={size} HA_FLIP_BEAR 📉")
                    self.sell(size=size, sl=sl, tp=tp)
                    self._entry_bar = len(self.data)

    def _manage_position(self):
        # Time stop: exit if no progress in N bars
        if hasattr(self, '_entry_bar'):
            bars_in_trade = len(self.data) - self._entry_bar
            if bars_in_trade >= self.time_stop_bars:
                ha_color = self.ha_color[-1]
                # Exit on HA flip back against position
                if self.position.is_long and ha_color == -1:
                    print(f"🌙⏰ TIME/HA EXIT LONG at {self.data.Close[-1]:.2f}")
                    self.position.close()
                elif self.position.is_short and ha_color == 1:
                    print(f"🌙⏰ TIME/HA EXIT SHORT at {self.data.Close[-1]:.2f}")
                    self.position.close()


print("🌙✨🚀 Running Backtest... ✨🌙🚀")
bt = Backtest(
    data,
    LiquidityHeikin,
    cash=1_000_000,
    commission=0.001,
    exclusive_orders=True
)

stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙✨ Backtest Complete! 🚀🌙")