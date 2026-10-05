import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# ============================================================
# 🌙 MOON DEV'S VOLATILITY STOCHASTIC BACKTEST 🚀
# ============================================================

print("🌙 Moon Dev is loading the cosmic data... ✨")

data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper column mapping
data.columns = [c.capitalize() for c in data.columns]
if 'Datetime' in data.columns:
    data = data.set_index('Datetime')
    data.index = pd.to_datetime(data.index)

print(f"🌙 Data loaded: {len(data)} rows of lunar price action 🚀")
print(f"✨ Columns: {list(data.columns)}")


class VolatilityStochastic(Strategy):
    # ---- Strategy Parameters ----
    atr_period = 150           # Long-term ATR for volatility regime
    vol_lookback = 200         # Lookback for percentile ranking
    vol_low_pct = 20           # Bottom percentile threshold
    vol_high_pct = 80          # Top percentile threshold
    stoch_k = 14
    stoch_d = 3
    stoch_smooth = 3
    sma_period = 200
    risk_pct = 0.01            # 1% risk per trade
    atr_stop_mult = 1.0        # 1x ATR stop
    rr_ratio = 2.0             # 2R take profit
    time_stop = 20             # Exit after N bars

    def init(self):
        print("🌙 Moon Dev is initializing the VolatilityStochastic indicators... ✨")

        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)

        # Long-term ATR (volatility regime)
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period, name='ATR_LT')

        # ATR percentile ranking over long lookback
        def atr_pct_rank(arr):
            s = pd.Series(arr)
            return s.rolling(self.vol_lookback).apply(
                lambda x: (x.iloc[-1] >= x).mean() * 100 if len(x) > 0 else np.nan,
                raw=False
            ).values
        self.atr_rank = self.I(atr_pct_rank, self.atr, name='ATR_Rank')

        # Stochastic oscillator
        self.stoch_k, self.stoch_d = self.I(
            talib.STOCH, high, low, close,
            fastk_period=self.stoch_k,
            slowk_period=self.stoch_smooth,
            slowk_matype=0,
            slowd_period=self.stoch_d,
            slowd_matype=0,
            name='Stoch'
        )

        # 200 SMA trend filter
        self.sma200 = self.I(talib.SMA, close, timeperiod=self.sma_period, name='SMA200')

        # Short-term ATR for stops
        self.atr_st = self.I(talib.ATR, high, low, close, timeperiod=14, name='ATR_ST')

        print("✨ Moon Dev indicators ready for liftoff! 🚀")

    def next(self):
        # Need enough history
        if len(self.data) < max(self.vol_lookback, self.atr_period, self.sma_period) + 5:
            return

        price = self.data.Close[-1]
        atr_rank = self.atr_rank[-1]
        atr_val = self.atr_st[-1]

        if np.isnan(atr_rank) or np.isnan(atr_val) or atr_val <= 0:
            return

        # ---- Manage open position ----
        if self.position:
            # Time stop
            if len(self.data) - self.trades[-1].entry_bar >= self.time_stop:
                print(f"⏰ Moon Dev time stop hit at {price:.2f} 🌙")
                self.position.close()
                return

            # Trailing stochastic exit
            if self.position.is_long:
                if self.stoch_k[-1] < self.stoch_d[-1] and self.stoch_k[-2] >= self.stoch_d[-2] and self.stoch_k[-1] > 70:
                    print(f"🌙 Moon Dev long exit on stoch reversal at {price:.2f} ✨")
                    self.position.close()
                    return
            elif self.position.is_short:
                if self.stoch_k[-1] > self.stoch_d[-1] and self.stoch_k[-2] <= self.stoch_d[-2] and self.stoch_k[-1] < 30:
                    print(f"🌙 Moon Dev short exit on stoch reversal at {price:.2f} ✨")
                    self.position.close()
                    return
            return

        # ---- Volatility regime check ----
        vol_extreme = (atr_rank <= self.vol_low_pct) or (atr_rank >= self.vol_high_pct)
        if not vol_extreme:
            return

        # ---- Stochastic cross detection ----
        k_prev, k_now = self.stoch_k[-2], self.stoch_k[-1]
        d_prev, d_now = self.stoch_d[-2], self.stoch_d[-1]

        # Long signal: %K crosses above %D from oversold
        long_signal = (k_prev < d_prev) and (k_now > d_now) and (k_prev < 20)
        # Short signal: %K crosses below %D from overbought
        short_signal = (k_prev > d_prev) and (k_now < d_now) and (k_prev > 80)

        # ---- Risk-based position sizing (volatility parity) ----
        risk_amount = self.equity * self.risk_pct
        stop_dist = atr_val * self.atr_stop_mult
        if stop_dist <= 0:
            return
        position_size = int(round(risk_amount / stop_dist))
        if position_size < 1:
            position_size = 1

        # ---- Long Entry ----
        if long_signal and price > self.sma200[-1]:
            sl = price - stop_dist
            tp = price + stop_dist * self.rr_ratio
            print(f"🚀🌙 Moon Dev LONG signal! Price={price:.2f} ATRrank={atr_rank:.1f} SL={sl:.2f} TP={tp:.2f} Size={position_size}")
            self.buy(size=position_size, sl=sl, tp=tp)

        # ---- Short Entry ----
        elif short_signal and price < self.sma200[-1]:
            sl = price + stop_dist
            tp = price - stop_dist * self.rr_ratio
            print(f"🚀🌙 Moon Dev SHORT signal! Price={price:.2f} ATRrank={atr_rank:.1f} SL={sl:.2f} TP={tp:.2f} Size={position_size}")
            self.sell(size=position_size, sl=sl, tp=tp)


print("🌙 Moon Dev is launching the backtest... 🚀✨")
bt = Backtest(data, VolatilityStochastic, cash=1_000_000, commission=0.001, exclusive_orders=True)
stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙 Moon Dev backtest complete! ✨🚀")