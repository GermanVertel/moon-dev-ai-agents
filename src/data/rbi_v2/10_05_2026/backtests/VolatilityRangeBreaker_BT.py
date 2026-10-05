import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# ==============================================================
# 🌙 MOON DEV'S VOLATILITY RANGE BREAKER 🌙
# ==============================================================

print("🌙✨ Moon Dev's VolatilityRangeBreaker is warming up... 🚀")

data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"

data = pd.read_csv(data_path)

# 🧹 Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# 🗺️ Map to backtesting.py required columns
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

# 🕐 Datetime index
if 'datetime' in data.columns:
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')

print(f"🌙✨ Data loaded: {len(data)} bars from {data.index[0]} to {data.index[-1]} 🚀")


class VolatilityRangeBreaker(Strategy):
    # ⚙️ Strategy parameters
    range_lookback = 20       # bars for range detection
    atr_period = 14           # ATR period
    atr_mult = 0.35           # breakout buffer multiplier (k)
    ema_period = 50           # trend bias EMA
    bb_period = 20            # Bollinger Band period
    bb_std = 2.0              # Bollinger Band std
    bb_lookback = 100         # lookback for BB width percentile
    bb_squeeze_pct = 20       # lowest 20th percentile
    atr_ma_period = 20        # ATR moving average for expansion filter
    tp_atr_mult = 1.75        # take profit ATR multiplier
    sl_atr_mult = 1.5         # stop loss ATR multiplier
    risk_pct = 0.01           # 1% risk per trade

    def init(self):
        print("🌙✨ Initializing VolatilityRangeBreaker indicators... 🚀")
        close = self.data.Close
        high = self.data.High
        low = self.data.Low

        # 📊 ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)
        # 📈 ATR moving average (expansion filter)
        self.atr_ma = self.I(talib.SMA, self.atr, timeperiod=self.atr_ma_period)

        # 📏 Range boundaries
        self.range_high = self.I(talib.MAX, high, timeperiod=self.range_lookback)
        self.range_low = self.I(talib.MIN, low, timeperiod=self.range_lookback)

        # 🎯 Trend bias EMA
        self.ema = self.I(talib.EMA, close, timeperiod=self.ema_period)

        # 🎈 Bollinger Bands for squeeze detection
        self.bb_upper, self.bb_mid, self.bb_lower = self.I(
            talib.BBANDS, close, timeperiod=self.bb_period,
            nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0
        )

        # 📉 BB width
        self.bb_width = self.I(
            lambda u, l, m: (u - l) / np.where(m == 0, np.nan, m),
            self.bb_upper, self.bb_lower, self.bb_mid
        )

        # 🔻 BB width percentile threshold (rolling)
        self.bb_width_thresh = self.I(
            lambda w: pd.Series(w).rolling(self.bb_lookback).quantile(
                self.bb_squeeze_pct / 100.0).values,
            self.bb_width
        )

        print("🌙✨ Indicators ready. Let's break some ranges! 🚀")

    def next(self):
        # Need enough bars
        if len(self.data) < max(self.range_lookback, self.ema_period,
                                self.bb_lookback, self.atr_ma_period) + 5:
            return

        price = self.data.Close[-1]
        atr = self.atr[-1]
        atr_ma = self.atr_ma[-1]
        rh = self.range_high[-1]
        rl = self.range_low[-1]
        ema = self.ema[-1]
        bbw = self.bb_width[-1]
        bbw_thresh = self.bb_width_thresh[-1]

        if np.isnan(atr) or np.isnan(atr_ma) or np.isnan(rh) or np.isnan(rl) \
           or np.isnan(ema) or np.isnan(bbw) or np.isnan(bbw_thresh):
            return

        # 🎯 Dynamic breakout triggers
        upper_trigger = rh + self.atr_mult * atr
        lower_trigger = rl - self.atr_mult * atr

        # 🌊 Volatility filter: ATR expanding OR BB width squeeze
        atr_expanding = atr > atr_ma
        bb_squeeze = bbw <= bbw_thresh
        vol_filter = atr_expanding or bb_squeeze

        # 🧭 Trend bias
        long_bias = price > ema
        short_bias = price < ema

        # 🚪 Entry logic
        if not self.position:
            # Long breakout
            if price > upper_trigger and vol_filter and long_bias:
                sl = price - self.sl_atr_mult * atr
                tp = price + self.tp_atr_mult * atr
                risk = price - sl
                if risk > 0:
                    size = int(round((self.equity * self.risk_pct) / risk))
                    if size > 0:
                        print(f"🌙🚀 LONG BREAKOUT! Price={price:.2f} > Trigger={upper_trigger:.2f} | "
                              f"ATR={atr:.2f} | SL={sl:.2f} TP={tp:.2f} | Size={size}")
                        self.buy(size=size, sl=sl, tp=tp)

            # Short breakout
            elif price < lower_trigger and vol_filter and short_bias:
                sl = price + self.sl_atr_mult * atr
                tp = price - self.tp_atr_mult * atr
                risk = sl - price
                if risk > 0:
                    size = int(round((self.equity * self.risk_pct) / risk))
                    if size > 0:
                        print(f"🌙🔻 SHORT BREAKOUT! Price={price:.2f} < Trigger={lower_trigger:.2f} | "
                              f"ATR={atr:.2f} | SL={sl:.2f} TP={tp:.2f} | Size={size}")
                        self.sell(size=size, sl=sl, tp=tp)

        # 🚪 Failed breakout / re-entry exit
        else:
            # If price re-enters range (back inside boundaries), exit
            if self.position.is_long and price < rh:
                print(f"🌙⚠️ Failed long breakout — price re-entered range at {price:.2f}. Exiting.")
                self.position.close()
            elif self.position.is_short and price > rl:
                print(f"🌙⚠️ Failed short breakout — price re-entered range at {price:.2f}. Exiting.")
                self.position.close()


print("🌙✨ Launching backtest with size = 1,000,000 🚀")

bt = Backtest(
    data,
    VolatilityRangeBreaker,
    cash=1_000_000,
    commission=0.0002,
    exclusive_orders=True
)

stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙✨ VolatilityRangeBreaker backtest complete! 🌙")