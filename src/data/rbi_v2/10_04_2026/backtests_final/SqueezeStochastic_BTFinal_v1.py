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

# Rename to proper case
data = data.rename(columns={
    'open': 'Open', 'high': 'High', 'low': 'Low',
    'close': 'Close', 'volume': 'Volume'
})

# Ensure datetime index
if 'datetime' in data.columns:
    data = data.set_index(pd.to_datetime(data['datetime']))
elif 'Datetime' in data.columns:
    data = data.set_index(pd.to_datetime(data['Datetime']))

# Keep required columns
data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

print("✨ Moon Dev data ready! Rows:", len(data))
print(data.head())


class SqueezeStochastic(Strategy):
    # --- Parameters ---
    bb_period = 20
    bb_std = 2.0
    squeeze_lookback = 100
    squeeze_threshold = 0.30
    stoch_k = 14
    stoch_d = 3
    stoch_smooth = 3
    vol_ma_period = 20
    vol_mult = 1.5
    atr_period = 14
    atr_mult = 1.5
    risk_pct = 0.02
    rr_ratio = 2.0

    def init(self):
        print("🌙 Moon Dev initializing SqueezeStochastic indicators...")

        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        # Bollinger Bands
        def bb_upper_fn():
            u, m, l = talib.BBANDS(close, timeperiod=self.bb_period,
                                   nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0)
            return u

        def bb_middle_fn():
            u, m, l = talib.BBANDS(close, timeperiod=self.bb_period,
                                   nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0)
            return m

        def bb_lower_fn():
            u, m, l = talib.BBANDS(close, timeperiod=self.bb_period,
                                   nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0)
            return l

        self.bb_upper = self.I(bb_upper_fn, name="BB_Upper")
        self.bb_middle = self.I(bb_middle_fn, name="BB_Middle")
        self.bb_lower = self.I(bb_lower_fn, name="BB_Lower")

        # Bandwidth
        self.bandwidth = self.I(
            lambda u, m, l: (u - l) / m,
            self.bb_upper, self.bb_middle, self.bb_lower,
            name="Bandwidth"
        )

        # Rolling min/max of bandwidth
        self.bw_min = self.I(talib.MIN, self.bandwidth, timeperiod=self.squeeze_lookback, name="BW_Min")
        self.bw_max = self.I(talib.MAX, self.bandwidth, timeperiod=self.squeeze_lookback, name="BW_Max")

        # Stochastic
        def stoch_k_fn():
            k, d = talib.STOCH(high, low, close,
                               fastk_period=self.stoch_k,
                               slowk_period=self.stoch_smooth,
                               slowk_matype=0,
                               slowd_period=self.stoch_d,
                               slowd_matype=0)
            return k

        def stoch_d_fn():
            k, d = talib.STOCH(high, low, close,
                               fastk_period=self.stoch_k,
                               slowk_period=self.stoch_smooth,
                               slowk_matype=0,
                               slowd_period=self.stoch_d,
                               slowd_matype=0)
            return d

        self.stoch_k = self.I(stoch_k_fn, name="Stoch_K")
        self.stoch_d = self.I(stoch_d_fn, name="Stoch_D")

        # Volume MA
        self.vol_ma = self.I(talib.SMA, volume, timeperiod=self.vol_ma_period, name="Vol_MA")

        # ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period, name="ATR")

        # EMA
        self.ema20 = self.I(talib.EMA, close, timeperiod=20, name="EMA20")

        print("🚀 Moon Dev indicators ready!")

    def is_squeeze(self):
        bw = self.bandwidth[-1]
        bw_min = self.bw_min[-1]
        bw_max = self.bw_max[-1]
        if np.isnan(bw) or np.isnan(bw_min) or np.isnan(bw_max) or bw_max == bw_min:
            return False
        pos = (bw - bw_min) / (bw_max - bw_min)
        return pos <= self.squeeze_threshold

    def next(self):
        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        volume = self.data.Volume[-1]

        if len(self.data) < self.squeeze_lookback + 5:
            return

        if (np.isnan(self.bb_upper[-1]) or np.isnan(self.stoch_k[-1]) or
                np.isnan(self.stoch_d[-1]) or np.isnan(self.vol_ma[-1]) or
                np.isnan(self.atr[-1]) or np.isnan(self.ema20[-1])):
            return

        squeeze = self.is_squeeze()
        vol_confirm = volume > self.vol_mult * self.vol_ma[-1]

        # --- Manage existing position ---
        if self.position:
            if self.position.is_long:
                if high >= self.bb_upper[-1]:
                    print(f"🌙✨ Moon Dev LONG TP hit at {price:.2f}")
                    self.position.close()
                    return
                if (self.stoch_k[-1] > 80 and self.stoch_d[-1] > 80 and
                        self.stoch_k[-2] < self.stoch_d[-2] and
                        self.stoch_k[-1] > self.stoch_d[-1]):
                    print(f"🌙✨ Moon Dev LONG Stoch reversal exit at {price:.2f}")
                    self.position.close()
                    return
                if price < self.ema20[-1]:
                    print(f"🌙✨ Moon Dev LONG trailing stop (EMA) at {price:.2f}")
                    self.position.close()
                    return

            elif self.position.is_short:
                if low <= self.bb_lower[-1]:
                    print(f"🌙✨ Moon Dev SHORT TP hit at {price:.2f}")
                    self.position.close()
                    return
                if (self.stoch_k[-1] < 20 and self.stoch_d[-1] < 20 and
                        self.stoch_k[-2] > self.stoch_d[-2] and
                        self.stoch_k[-1] < self.stoch_d[-1]):
                    print(f"🌙✨ Moon Dev SHORT Stoch reversal exit at {price:.2f}")
                    self.position.close()
                    return
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

        # Long
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
            if self.bb_upper[-1] < tp_price:
                tp_price = self.bb_upper[-1]
            if (tp_price - price) / risk < self.rr_ratio:
                print("🌙 ⚠️ Skipping LONG - insufficient R:R")
                return

            # Use fraction-based sizing (risk_pct of equity)
            size = self.risk_pct
            print(f"🌙🚀 Moon Dev LONG ENTRY @ {price:.2f} | SL {stop_price:.2f} | TP {tp_price:.2f} | size {size}")
            self.buy(size=size, sl=stop_price, tp=tp_price)

        # Short
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

            size = self.risk_pct
            print(f"🌙🚀 Moon Dev SHORT ENTRY @ {price:.2f} | SL {stop_price:.2f} | TP {tp_price:.2f} | size {size}")
            self.sell(size=size, sl=stop_price, tp=tp_price)


print("🌙 Moon Dev launching backtest...")
bt = Backtest(data, SqueezeStochastic, cash=1_000_000, commission=0.001, exclusive_orders=True)
stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙✨ Moon Dev backtest complete! 🚀")