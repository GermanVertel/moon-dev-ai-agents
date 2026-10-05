import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# ============================================================
# 🌙 MOON DEV BACKTEST AI - VolatilityIgnition Strategy 🚀
# ============================================================

print("🌙 Moon Dev Backtest AI initializing... ✨")
print("🚀 Loading VolatilityIgnition strategy... 🎯")

# Load data
data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
print(f"📂 Loading data from: {data_path}")

data = pd.read_csv(data_path)

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper column mapping
data = data.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
data = data.set_index(pd.to_datetime(data['datetime']))

# Ensure all OHLCV columns are float64 (talib requires double arrays) 🌙
for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    data[col] = data[col].astype(np.float64)

print(f"✅ Data loaded: {len(data)} bars")
print(f"📊 Columns: {list(data.columns)}")
print(f"🌙 Date range: {data.index[0]} to {data.index[-1]}")
print("🚀 Moon Dev Backtest AI ready to launch! ✨")


class VolatilityIgnition(Strategy):
    """
    🌙 VolatilityIgnition Strategy 🌙

    Trades breakouts only when volatility AND volume confirm expansion.
    - Donchian breakout (20-bar)
    - ATR(20) > SMA(ATR(20), 90) - volatility regime filter
    - Volume > SMA(Volume, 50) - participation filter
    - Stop: 1x ATR, Target: 2x ATR (2:1 R:R)
    - Time stop: 15 bars
    """

    # Tunable parameters
    donchian_period = 20
    atr_period = 20
    atr_sma_period = 90
    volume_sma_period = 50
    atr_stop_mult = 1.0
    reward_risk = 2.0
    time_stop_bars = 15
    risk_pct = 0.01  # 1% risk per trade

    def init(self):
        print("🌙 Initializing indicators... ✨")

        # Donchian channels (previous bar's high/low to avoid lookahead)
        self.donchian_high = self.I(talib.MAX, self.data.High, timeperiod=self.donchian_period)
        self.donchian_low = self.I(talib.MIN, self.data.Low, timeperiod=self.donchian_period)

        # ATR for volatility regime
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period)
        self.atr_sma = self.I(talib.SMA, self.atr, timeperiod=self.atr_sma_period)

        # Volume SMA - cast to float64 for talib 🌙
        volume_arr = np.asarray(self.data.Volume, dtype=np.float64)
        self.volume_sma = self.I(talib.SMA, volume_arr, timeperiod=self.volume_sma_period)

        # Track entry bar for time stop
        self.entry_bar = None
        self.entry_price = None
        self.stop_price = None
        self.tp_price = None

        print("🚀 Indicators ready! Moon Dev power engaged! 🌙")

    def next(self):
        # Need enough history
        if len(self.data) < self.atr_sma_period + 5:
            return

        # Check for NaN
        if (np.isnan(self.atr[-1]) or np.isnan(self.atr_sma[-1]) or
            np.isnan(self.volume_sma[-1]) or np.isnan(self.donchian_high[-2])):
            return

        price = self.data.Close[-1]
        high_prev = self.donchian_high[-2]  # previous bar's donchian high
        low_prev = self.donchian_low[-2]

        # Volatility & volume filters
        vol_regime_ok = self.atr[-1] > self.atr_sma[-1]
        volume_ok = self.data.Volume[-1] > self.volume_sma[-1]

        # ============ EXIT LOGIC ============
        if self.position:
            bars_in_trade = len(self.data) - self.entry_bar

            if self.position.is_long:
                # Stop loss hit
                if self.data.Low[-1] <= self.stop_price:
                    print(f"🛑 LONG STOP hit @ {self.stop_price:.2f} | Moon Dev protecting capital! 🌙")
                    self.position.close()
                    return
                # Take profit hit
                if self.data.High[-1] >= self.tp_price:
                    print(f"🎯 LONG TP hit @ {self.tp_price:.2f} | Moon Dev profits secured! 💰")
                    self.position.close()
                    return
                # Time stop
                if bars_in_trade >= self.time_stop_bars:
                    print(f"⏰ LONG TIME STOP @ {price:.2f} | Dead trade exited! 🌙")
                    self.position.close()
                    return

            elif self.position.is_short:
                if self.data.High[-1] >= self.stop_price:
                    print(f"🛑 SHORT STOP hit @ {self.stop_price:.2f} | Moon Dev protecting capital! 🌙")
                    self.position.close()
                    return
                if self.data.Low[-1] <= self.tp_price:
                    print(f"🎯 SHORT TP hit @ {self.tp_price:.2f} | Moon Dev profits secured! 💰")
                    self.position.close()
                    return
                if bars_in_trade >= self.time_stop_bars:
                    print(f"⏰ SHORT TIME STOP @ {price:.2f} | Dead trade exited! 🌙")
                    self.position.close()
                    return
            return

        # ============ ENTRY LOGIC ============
        # Long entry: breakout above donchian high + vol regime + volume
        long_breakout = price > high_prev
        # Short entry: breakout below donchian low + vol regime + volume
        short_breakout = price < low_prev

        if long_breakout and vol_regime_ok and volume_ok:
            stop_dist = self.atr[-1] * self.atr_stop_mult
            if stop_dist <= 0:
                return
            stop_price = price - stop_dist
            tp_price = price + stop_dist * self.reward_risk

            # Position sizing: 1,000,000 units as specified
            size = 1_000_000

            print(f"🚀 LONG ENTRY @ {price:.2f} | ATR={self.atr[-1]:.2f} | "
                  f"Stop={stop_price:.2f} | TP={tp_price:.2f} | Size={size} 🌙")
            self.buy(size=size)
            self.entry_bar = len(self.data)
            self.entry_price = price
            self.stop_price = stop_price
            self.tp_price = tp_price

        elif short_breakout and vol_regime_ok and volume_ok:
            stop_dist = self.atr[-1] * self.atr_stop_mult
            if stop_dist <= 0:
                return
            stop_price = price + stop_dist
            tp_price = price - stop_dist * self.reward_risk

            size = 1_000_000

            print(f"🔻 SHORT ENTRY @ {price:.2f} | ATR={self.atr[-1]:.2f} | "
                  f"Stop={stop_price:.2f} | TP={tp_price:.2f} | Size={size} 🌙")
            self.sell(size=size)
            self.entry_bar = len(self.data)
            self.entry_price = price
            self.stop_price = stop_price
            self.tp_price = tp_price


print("🌙 Running VolatilityIgnition backtest... 🚀")
bt = Backtest(data, VolatilityIgnition, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)
print("✨ Moon Dev Backtest complete! 🌙🚀")