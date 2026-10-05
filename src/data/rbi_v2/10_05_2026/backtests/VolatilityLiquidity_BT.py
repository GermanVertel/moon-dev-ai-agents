import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

print("🌙✨ Moon Dev Backtest AI - VolatilityLiquidity Strategy ✨🌙")

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
if 'datetime' in data.columns:
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')

print(f"🚀 Loaded {len(data)} bars of data")
print(f"🌙 Data columns: {list(data.columns)}")


class VolatilityLiquidity(Strategy):
    # Parameters
    bb_period = 20
    bb_std = 2.0
    bbw_lookback = 100
    bbw_percentile = 20  # 20th percentile
    atr_period = 14
    vol_ma_period = 20
    vol_spike_mult = 1.5
    rsi_period = 14
    risk_pct = 0.02  # 2% risk per trade
    rr_target = 2.5  # 2.5R target
    time_stop_bars = 15
    trail_atr_mult = 1.5

    def init(self):
        print("🌙 Initializing VolatilityLiquidity indicators...")
        
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        # Bollinger Bands
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS, close, timeperiod=self.bb_period,
            nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0
        )

        # Bollinger Band Width
        self.bbw = self.I(
            lambda u, m, l: (u - l) / np.where(m > 0, m, 1),
            self.bb_upper, self.bb_middle, self.bb_lower
        )

        # BBW rolling percentile threshold
        self.bbw_threshold = self.I(
            lambda x: pd.Series(x).rolling(self.bbw_lookback).quantile(self.bbw_percentile / 100.0).values,
            self.bbw
        )

        # ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        # Volume MA
        self.vol_ma = self.I(talib.SMA, volume, timeperiod=self.vol_ma_period)

        # RSI
        self.rsi = self.I(talib.RSI, close, timeperiod=self.rsi_period)

        # MACD
        self.macd, self.macd_signal, self.macd_hist = self.I(
            talib.MACD, close, fastperiod=12, slowperiod=26, signalperiod=9
        )

        print("✨ Indicators ready! 🚀")

    def next(self):
        price = self.data.Close[-1]

        # Ensure we have enough data
        if len(self.data) < self.bbw_lookback + 5:
            return

        # Skip if indicators not ready
        if np.isnan(self.bbw_threshold[-1]) or np.isnan(self.atr[-1]) or np.isnan(self.bbw[-1]):
            return

        # --- State checks ---
        bbw_contraction = self.bbw[-1] < self.bbw_threshold[-1]
        vol_spike = self.data.Volume[-1] > (self.vol_spike_mult * self.vol_ma[-1])

        # --- Manage open position ---
        if self.position:
            entry_price = self.trades[-1].entry_price
            bars_held = len(self.data) - self.trades[-1].entry_bar

            # Time stop
            if bars_held >= self.time_stop_bars:
                print(f"⏰ Time stop hit after {bars_held} bars — exiting 🌙")
                self.position.close()
                return

            if self.position.is_long:
                # Trailing stop using middle BB once 1R in profit
                risk = entry_price - self.trades[-1].sl if self.trades[-1].sl else self.atr[-1]
                if price >= entry_price + risk:
                    new_sl = max(self.trades[-1].sl or 0, self.bb_middle[-1])
                    if new_sl > (self.trades[-1].sl or 0):
                        self.trades[-1].sl = new_sl
                        print(f"📈 Trailing long SL to {new_sl:.2f} 🌙")

            elif self.position.is_short:
                risk = self.trades[-1].sl - entry_price if self.trades[-1].sl else self.atr[-1]
                if price <= entry_price - risk:
                    new_sl = min(self.trades[-1].sl or float('inf'), self.bb_middle[-1])
                    if new_sl < (self.trades[-1].sl or float('inf')):
                        self.trades[-1].sl = new_sl
                        print(f"📉 Trailing short SL to {new_sl:.2f} 🌙")
            return

        # --- Entry logic ---
        if not bbw_contraction:
            return

        # Confirmation: volume spike OR MACD crossover OR RSI alignment
        macd_bull = self.macd[-1] > self.macd_signal[-1] and self.macd[-2] <= self.macd_signal[-2]
        macd_bear = self.macd[-1] < self.macd_signal[-1] and self.macd[-2] >= self.macd_signal[-2]
        rsi_bull = self.rsi[-1] > 50
        rsi_bear = self.rsi[-1] < 50

        # Breakout confirmation: close beyond bands
        close_above_upper = self.data.Close[-1] > self.bb_upper[-1] and self.data.Close[-2] <= self.bb_upper[-2]
        close_below_lower = self.data.Close[-1] < self.bb_lower[-1] and self.data.Close[-2] >= self.bb_lower[-2]

        # Liquidation zone proxy: swing extremes over lookback
        recent_high = talib.MAX(self.data.High, timeperiod=20)[-1]
        recent_low = talib.MIN(self.data.Low, timeperiod=20)[-1]
        near_upper_zone = abs(self.data.High[-1] - recent_high) < 0.5 * self.atr[-1]
        near_lower_zone = abs(self.data.Low[-1] - recent_low) < 0.5 * self.atr[-1]

        # Position sizing: 1,000,000 units base, scaled by risk
        contraction_depth = (self.bbw_threshold[-1] - self.bbw[-1]) / self.bbw_threshold[-1] if self.bbw_threshold[-1] > 0 else 0
        size_scale = 1.0 - min(0.5, contraction_depth)  # tighter contraction = smaller size
        base_size = 1_000_000
        position_size = int(round(base_size * size_scale))

        # --- Long entry ---
        if (close_above_upper or (vol_spike and rsi_bull and self.data.Close[-1] > self.bb_middle[-1])) \
                and near_upper_zone and (vol_spike or macd_bull):
            sl = self.bb_lower[-1] - 0.5 * self.atr[-1]
            risk = price - sl
            if risk > 0:
                tp = price + self.rr_target * risk
                print(f"🚀🌙 LONG SIGNAL! Price={price:.2f} BBW={self.bbw[-1]:.4f} < Thresh={self.bbw_threshold[-1]:.4f}")
                print(f"   SL={sl:.2f} TP={tp:.2f} Size={position_size}")
                self.buy(size=position_size, sl=sl, tp=tp)

        # --- Short entry ---
        elif (close_below_lower or (vol_spike and rsi_bear and self.data.Close[-1] < self.bb_middle[-1])) \
                and near_lower_zone and (vol_spike or macd_bear):
            sl = self.bb_upper[-1] + 0.5 * self.atr[-1]
            risk = sl - price
            if risk > 0:
                tp = price - self.rr_target * risk
                print(f"🔻🌙 SHORT SIGNAL! Price={price:.2f} BBW={self.bbw[-1]:.4f} < Thresh={self.bbw_threshold[-1]:.4f}")
                print(f"   SL={sl:.2f} TP={tp:.2f} Size={position_size}")
                self.sell(size=position_size, sl=sl, tp=tp)


print("🌙 Setting up backtest...")
bt = Backtest(
    data,
    VolatilityLiquidity,
    cash=1_000_000,
    commission=0.001,
    exclusive_orders=True
)

print("🚀 Running backtest...")
stats = bt.run()
print(stats)
print(stats._strategy)