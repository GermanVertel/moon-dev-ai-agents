import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# =============================================================================
# 🌙 MOON DEV BACKTEST AI - LiquidityStochastic Strategy 🚀
# =============================================================================

print("🌙 Moon Dev is warming up the engines... ✨")

data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"

data = pd.read_csv(data_path)

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper case mapping
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
data = data.dropna()

print(f"🌙 Moon Dev loaded {len(data)} candles of data ✨")
print(f"🚀 Data range: {data.index[0]} -> {data.index[-1]}")


class LiquidityStochastic(Strategy):
    # 🎯 Strategy parameters
    k_period = 14
    d_period = 3
    smooth_k = 3
    cmf_period = 20
    atr_period = 14
    ema_fast = 50
    ema_slow = 200
    risk_pct = 0.01        # 1% risk per trade
    atr_mult = 1.5         # stop = 1.5 * ATR
    reward_mult = 2.0      # 2R take profit
    cmf_neutral = 0.05     # avoid trades when |CMF| < 0.05

    def init(self):
        print("🌙 Moon Dev initializing indicators... ✨")

        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        # 🎯 Stochastic Oscillator (14, 3, 3)
        self.k, self.d = self.I(
            talib.STOCH,
            high, low, close,
            fastk_period=self.k_period,
            slowk_period=self.smooth_k,
            slowk_matype=0,
            slowd_period=self.d_period,
            slowd_matype=0,
        )

        # 🎯 Chaikin Money Flow (20)
        def cmf_func(high, low, close, volume, period):
            mfm = ((close - low) - (high - close)) / (high - low)
            mfm = np.where((high - low) == 0, 0, mfm)
            mfv = mfm * volume
            cmf = pd.Series(mfv).rolling(period).sum() / pd.Series(volume).rolling(period).sum()
            return cmf.values

        self.cmf = self.I(cmf_func, high, low, close, volume, self.cmf_period)

        # 🎯 ATR for stops
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        # 🎯 Trend EMAs
        self.ema_fast_line = self.I(talib.EMA, close, timeperiod=self.ema_fast)
        self.ema_slow_line = self.I(talib.EMA, close, timeperiod=self.ema_slow)

        # 🎯 Swing highs/lows for reference
        self.swing_low = self.I(talib.MIN, low, timeperiod=20)
        self.swing_high = self.I(talib.MAX, high, timeperiod=20)

        print("🌙 Moon Dev indicators ready! Let's find some liquidity! 🚀")

    def next(self):
        price = self.data.Close[-1]

        # Need enough bars
        if len(self.data) < max(self.ema_slow, self.cmf_period, self.k_period) + 5:
            return

        # Current indicator values
        k_now = self.k[-1]
        d_now = self.d[-1]
        k_prev = self.k[-2]
        d_prev = self.d[-2]

        cmf_now = self.cmf[-1]
        cmf_prev = self.cmf[-2]
        cmf_prev2 = self.cmf[-3]

        atr_now = self.atr[-1]
        ema_f = self.ema_fast_line[-1]
        ema_s = self.ema_slow_line[-1]

        if np.isnan(k_now) or np.isnan(d_now) or np.isnan(cmf_now) or np.isnan(atr_now):
            return

        # 🌙 Bullish / Bearish crosses (no backtesting.lib!)
        bull_cross = k_prev <= d_prev and k_now > d_now
        bear_cross = k_prev >= d_prev and k_now < d_now

        # 💰 CMF rising / falling
        cmf_rising = cmf_now > cmf_prev and cmf_prev >= cmf_prev2
        cmf_falling = cmf_now < cmf_prev and cmf_prev <= cmf_prev2

        # 📈 Trend context
        uptrend = ema_f > ema_s
        downtrend = ema_f < ema_s

        # ================== ENTRY LOGIC ==================
        if not self.position:
            # 🟢 LONG Entry
            long_oversold = bull_cross and k_now < 20 and d_now < 20
            long_uptrend = bull_cross and 20 <= k_now < 50 and uptrend

            cmf_ok_long = cmf_now > self.cmf_neutral and (cmf_rising or cmf_now > 0)

            if (long_oversold or long_uptrend) and cmf_ok_long:
                stop = price - self.atr_mult * atr_now
                risk = price - stop
                if risk <= 0:
                    return
                position_size = int(round((self.equity * self.risk_pct) / risk))
                if position_size <= 0:
                    return
                tp = price + self.reward_mult * risk
                self.buy(size=position_size, sl=stop, tp=tp)
                print(f"🌙✨ LONG ENTRY! K={k_now:.2f} D={d_now:.2f} CMF={cmf_now:.3f} "
                      f"Price={price:.2f} SL={stop:.2f} TP={tp:.2f} Size={position_size} 🚀")

            # 🔴 SHORT Entry
            short_overbought = bear_cross and k_now > 80 and d_now > 80
            short_downtrend = bear_cross and 50 < k_now <= 80 and downtrend

            cmf_ok_short = cmf_now < -self.cmf_neutral and (cmf_falling or cmf_now < 0)

            if (short_overbought or short_downtrend) and cmf_ok_short:
                stop = price + self.atr_mult * atr_now
                risk = stop - price
                if risk <= 0:
                    return
                position_size = int(round((self.equity * self.risk_pct) / risk))
                if position_size <= 0:
                    return
                tp = price - self.reward_mult * risk
                self.sell(size=position_size, sl=stop, tp=tp)
                print(f"🌙✨ SHORT ENTRY! K={k_now:.2f} D={d_now:.2f} CMF={cmf_now:.3f} "
                      f"Price={price:.2f} SL={stop:.2f} TP={tp:.2f} Size={position_size} 🚀")

        # ================== EXIT LOGIC ==================
        else:
            if self.position.is_long:
                # Exit on bearish cross in overbought OR CMF flips negative
                exit_signal = (bear_cross and k_now > 80) or (cmf_now < 0 and cmf_prev < 0)
                if exit_signal:
                    self.position.close()
                    print(f"🌙💫 LONG EXIT! K={k_now:.2f} D={d_now:.2f} CMF={cmf_now:.3f} Price={price:.2f}")

            elif self.position.is_short:
                # Exit on bullish cross in oversold OR CMF flips positive
                exit_signal = (bull_cross and k_now < 20) or (cmf_now > 0 and cmf_prev > 0)
                if exit_signal:
                    self.position.close()
                    print(f"🌙💫 SHORT EXIT! K={k_now:.2f} D={d_now:.2f} CMF={cmf_now:.3f} Price={price:.2f}")


print("🌙 Moon Dev launching the backtest... 🚀✨")

bt = Backtest(
    data,
    LiquidityStochastic,
    cash=1_000_000,
    commission=0.001,
    exclusive_orders=True,
)

stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙 Moon Dev backtest complete! ✨🚀")