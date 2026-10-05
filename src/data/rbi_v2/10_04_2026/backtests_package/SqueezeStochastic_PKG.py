import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# ============================================================
# 🌙 MOON DEV - SQUEEZE STOCHASTIC STRATEGY 🚀
# ============================================================

data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"

print("🌙 Moon Dev loading data from:", data_path)
data = pd.read_csv(data_path)

# Clean columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Map to proper case
data.columns = [c.capitalize() for c in data.columns]

# Ensure datetime
if 'Datetime' in data.columns:
    data['Datetime'] = pd.to_datetime(data['Datetime'])
    data = data.set_index('Datetime')

# Keep required columns
data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

print("✨ Moon Dev data ready! Rows:", len(data))
print(data.head())


class SqueezeStochastic(Strategy):
    # --- Parameters ---
    bb_period = 20
    bb_std = 2.0
    squeeze_lookback = 100
    squeeze_threshold = 0.30  # lowest 30% of bandwidth range
    stoch_k = 14
    stoch_d = 3
    stoch_smooth = 3
    vol_ma_period = 20
    vol_mult = 1.5
    atr_period = 14
    atr_mult = 1.5
    risk_pct = 0.02  # 2% risk per trade
    rr_ratio = 2.0   # 2:1 reward-to-risk

    def init(self):
        print("🌙 Moon Dev initializing SqueezeStochastic indicators...")

        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        # Bollinger Bands (talib.BBANDS returns upper, middle, lower)
        bb_upper, bb_middle, bb_lower = talib.BBANDS(
            close, timeperiod=self.bb_period,
            nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0
        )
        self.bb_upper = self.I(lambda: bb_upper, name="BB_Upper")
        self.bb_middle = self.I(lambda: bb_middle, name="BB_Middle")
        self.bb_lower = self.I(lambda: bb_lower, name="BB_Lower")

        # Bandwidth
        self.bandwidth = self.I(
            lambda u, m, l: (u - l) / m,
            self.bb_upper, self.bb_middle, self.bb_lower,
            name="Bandwidth"
        )

        # Rolling min/max of bandwidth for squeeze detection
        self.bw_min = self.I(talib.MIN, self.bandwidth, timeperiod=self.squeeze_lookback, name="BW_Min")
        self.bw_max = self.I(talib.MAX, self.bandwidth, timeperiod=self.squeeze_lookback, name="BW_Max")

        # Stochastic (talib.STOCH returns slowk, slowd)
        stoch_k, stoch_d = talib.STOCH(
            high, low, close,
            fastk_period=self.stoch_k,
            slowk_period=self.stoch_smooth,
            slowk_matype=0,
            slowd_period=self.stoch_d,
            slowd_matype=0
        )
        self.stoch_k = self.I(lambda: stoch_k, name="Stoch_K")
        self.stoch_d = self.I(lambda: stoch_d, name="Stoch_D")

        # Volume MA
        self.vol_ma = self.I(talib.SMA, volume, timeperiod=self.vol_ma_period, name="Vol_MA")

        # ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period, name="ATR")

        # EMA for trailing
        self.ema20 = self.I(talib.EMA, close, timeperiod=20, name="EMA20")

        print("🚀 Moon Dev indicators ready!")

    def is_squeeze(self):
        bw = self.bandwidth[-1]
        bw_min = self.bw_min[-1]
        bw_max = self.bw_max[-1]
        if np.isnan(bw) or np.isnan(bw_min) or np.isnan(bw_max) or bw_max == bw_min:
            return False
        # position in range
        pos = (bw - bw_min) / (bw_max - bw_min)
        return pos <= self.squeeze_threshold

    def next(self):
        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        volume = self.data.Volume[-1]

        if len(self.data) < self.squeeze_lookback + 5:
            return

        # Skip if indicators not ready
        if (np.isnan(self.bb_upper[-1]) or np.isnan(self.stoch_k[-1]) or
                np.isnan(self.stoch_d[-1]) or np.isnan(self.vol_ma[-1]) or
                np.isnan(self.atr[-1]) or np.isnan(self.ema20[-1])):
            return

        squeeze = self.is_squeeze()
        vol_confirm = volume > self.vol_mult * self.vol_ma[-1]

        # --- Manage existing position ---
        if self.position:
            if self.position.is_long:
                # Take profit at upper band
                if high >= self.bb_upper[-1]:
                    print(f"🌙✨ Moon Dev LONG TP hit at {price:.2f}")
                    self.position.close()
                    return
                # Stoch overbought cross down
                if (self.stoch_k[-1] > 80 and self.stoch_d[-1] > 80 and
                        self.stoch_k[-2] < self.stoch_d[-2] and
                        self.stoch_k[-1] > self.stoch_d[-1]):
                    print(f"🌙✨ Moon Dev LONG Stoch reversal exit at {price:.2f}")
                    self.position.close()
                    return
                # Trailing stop: close below EMA20
                if price < self.ema20[-1]:
                    print(f"🌙✨ Moon Dev LONG trailing stop (EMA) at {price:.2f}")
                    self.position.close()
                    return

            elif self.position.is_short:
                # Take profit at lower band
                if low <= self.bb_lower[-1]:
                    print(f"🌙✨ Moon Dev SHORT TP hit at {price:.2f}")
                    self.position.close()
                    return
                # Stoch oversold cross up
                if (self.stoch_k[-1] < 20 and self.stoch_d[-1] < 20 and
                        self.stoch_k[-2] > self.stoch_d[-2] and
                        self.stoch_k[-1] < self.stoch_d[-1]):
                    print(f"🌙✨ Moon Dev SHORT Stoch reversal exit at {price:.2f}")
                    self.position.close()
                    return
                # Trailing stop: close above EMA20
                if price > self.ema20[-1]:
                    print(f"🌙✨ Moon Dev SHORT trailing stop (EMA) at {price:.2f}")
                    self.position.close()
                    return
            return

        # --- Entry logic ---
        if not squeeze:
            return
        if not vol_confirm:
            return

        # Long: price pierced lower band recently, stoch oversold cross up, close back above lower band
        touched_lower = low <= self.bb_lower[-1] * 1.001
        close_above_lower = price > self.bb_lower[-1]
        stoch_oversold = self.stoch_k[-1] < 20 and self.stoch_d[-1] < 20
        stoch_cross_up = (self.stoch_k[-2] < self.stoch_d[-2] and
                          self.stoch_k[-1] > self.stoch_d[-1])

        if touched_lower and close_above_lower and stoch_oversold and stoch_cross_up:
            stop_price = low - self.atr_mult * self.atr[-1]
            risk = price - stop_price
            if risk <= 0:
                return
            tp_price = price + self.rr_ratio * risk
            # Cap TP at upper band if closer
            if self.bb_upper[-1] < tp_price:
                tp_price = self.bb_upper[-1]
            # Require at least 2:1
            if (tp_price - price) / risk < self.rr_ratio:
                print("🌙 ⚠️ Skipping LONG - insufficient R:R")
                return

            size = int(round((self.equity * self.risk_pct) / risk))
            if size <= 0:
                return
            print(f"🌙🚀 Moon Dev LONG ENTRY @ {price:.2f} | SL {stop_price:.2f} | TP {tp_price:.2f} | size {size}")
            self.buy(size=size, sl=stop_price, tp=tp_price)

        # Short: price pierced upper band, stoch overbought cross down, close back below upper band
        touched_upper = high >= self.bb_upper[-1] * 0.999
        close_below_upper = price < self.bb_upper[-1]
        stoch_overbought = self.stoch_k[-1] > 80 and self.stoch_d[-1] > 80
        stoch_cross_down = (self.stoch_k[-2] > self.stoch_d[-2] and
                            self.stoch_k[-1] < self.stoch_d[-1])

        if touched_upper and close_below_upper and stoch_overbought and stoch_cross_down:
            stop_price = high + self.atr_mult * self.atr[-1]
            risk = stop_price - price
            if risk <= 0:
                return
            tp_price = price - self.rr_ratio * risk
            if self.bb_lower[-1] > tp_price:
                tp_price = self.bb_lower[-1]
            if (price - tp_price) / risk < self.rr_ratio:
                print("🌙 ⚠️ Skipping SHORT - insufficient R:R")
                return

            size = int(round((self.equity * self.risk_pct) / risk))
            if size <= 0:
                return
            print(f"🌙🚀 Moon Dev SHORT ENTRY @ {price:.2f} | SL {stop_price:.2f} | TP {tp_price:.2f} | size {size}")
            self.sell(size=size, sl=stop_price, tp=tp_price)


print("🌙 Moon Dev launching backtest...")
bt = Backtest(data, SqueezeStochastic, cash=1_000_000, commission=0.001, exclusive_orders=True)
stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙✨ Moon Dev backtest complete! 🚀")