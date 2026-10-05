import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Load and prepare data
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
data = data.dropna()

# Ensure numeric dtypes (talib requires double arrays)
for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    data[col] = pd.to_numeric(data[col], errors='coerce').astype('float64')
data = data.dropna()

print("🌙 Moon Dev: Data loaded successfully! ✨")
print(f"🚀 Shape: {data.shape}")
print(f"📊 Date range: {data.index[0]} to {data.index[-1]}")


class VolatilityLiquidation(Strategy):
    # Strategy parameters
    atr_period = 14
    atr_ma_period = 50
    atr_mult = 1.5
    bb_period = 20
    bb_std = 2
    ema_fast = 50
    ema_slow = 200
    rsi_period = 14
    rsi_oversold = 30
    rsi_overbought = 70
    risk_pct = 0.01
    atr_stop_mult = 1.0
    atr_target_mult = 2.0
    time_stop_bars = 5
    vol_mult = 1.2  # volume spike threshold vs 20-avg

    def init(self):
        print("🌙 Moon Dev: Initializing VolatilityLiquidation strategy... ✨")
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        # ATR & its moving average
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)
        self.atr_ma = self.I(talib.SMA, self.atr, timeperiod=self.atr_ma_period)

        # Bollinger Bands - use a wrapper to return single bands
        def bb_upper_func(c):
            return talib.BBANDS(c, timeperiod=self.bb_period,
                                nbdevup=self.bb_std, nbdevdn=self.bb_std)[0]

        def bb_lower_func(c):
            return talib.BBANDS(c, timeperiod=self.bb_period,
                                nbdevup=self.bb_std, nbdevdn=self.bb_std)[2]

        self.bb_upper = self.I(bb_upper_func, close)
        self.bb_lower = self.I(bb_lower_func, close)
        self.bb_width = self.I(lambda u, l, c: (u - l) / c,
                               self.bb_upper, self.bb_lower, close)

        # EMAs
        self.ema50 = self.I(talib.EMA, close, timeperiod=self.ema_fast)
        self.ema200 = self.I(talib.EMA, close, timeperiod=self.ema_slow)

        # RSI
        self.rsi = self.I(talib.RSI, close, timeperiod=self.rsi_period)

        # Volume MA - cast to float64 to satisfy talib double requirement
        vol_arr = np.asarray(volume, dtype=np.float64)
        self.vol_ma = self.I(talib.SMA, vol_arr, timeperiod=20)

        # Liquidation proxy: large wick + volume spike.
        body = self.I(lambda o, c: np.abs(np.asarray(c, dtype=np.float64) - np.asarray(o, dtype=np.float64)),
                      self.data.Open, close)
        upper_wick = self.I(lambda h, o, c: np.asarray(h, dtype=np.float64) - np.maximum(np.asarray(o, dtype=np.float64), np.asarray(c, dtype=np.float64)),
                            high, self.data.Open, close)
        lower_wick = self.I(lambda l, o, c: np.minimum(np.asarray(o, dtype=np.float64), np.asarray(c, dtype=np.float64)) - np.asarray(l, dtype=np.float64),
                            low, self.data.Open, close)
        self.body = body
        self.upper_wick = upper_wick
        self.lower_wick = lower_wick

        # Track trade entry bar for time stop
        self.entry_bar = None
        self.entry_price = None
        self.stop_price = None
        self.target_price = None

        print("🌙 Moon Dev: Indicators ready! 🚀")

    def next(self):
        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        open_ = self.data.Open[-1]

        atr = self.atr[-1]
        atr_ma = self.atr_ma[-1]
        ema50 = self.ema50[-1]
        ema200 = self.ema200[-1]
        rsi = self.rsi[-1]
        vol = self.data.Volume[-1]
        vol_ma = self.vol_ma[-1]

        if np.isnan(atr) or np.isnan(atr_ma) or np.isnan(ema50) or np.isnan(ema200) or np.isnan(rsi):
            return

        # ---------- Manage open position ----------
        if self.position:
            bars_held = len(self.data) - self.entry_bar
            # Time stop
            if bars_held >= self.time_stop_bars:
                print(f"⏰ Moon Dev: Time stop hit @ {price:.2f} 🌙")
                self.position.close()
                return

            # Trailing stop after 1x ATR in favor
            if self.position.is_long:
                if high >= self.entry_price + atr:
                    new_stop = price - atr
                    if new_stop > self.stop_price:
                        self.stop_price = new_stop
                        print(f"🎯 Moon Dev: Trailing long stop to {new_stop:.2f} ✨")
                if low <= self.stop_price:
                    print(f"🛑 Moon Dev: Long stop hit @ {price:.2f} 🌙")
                    self.position.close()
                    return
                if high >= self.target_price:
                    print(f"💰 Moon Dev: Long target hit @ {price:.2f} 🚀")
                    self.position.close()
                    return

            elif self.position.is_short:
                if low <= self.entry_price - atr:
                    new_stop = price + atr
                    if new_stop < self.stop_price:
                        self.stop_price = new_stop
                        print(f"🎯 Moon Dev: Trailing short stop to {new_stop:.2f} ✨")
                if high >= self.stop_price:
                    print(f"🛑 Moon Dev: Short stop hit @ {price:.2f} 🌙")
                    self.position.close()
                    return
                if low <= self.target_price:
                    print(f"💰 Moon Dev: Short target hit @ {price:.2f} 🚀")
                    self.position.close()
                    return
            return

        # ---------- Volatility regime filter ----------
        vol_high = atr > (self.atr_mult * atr_ma)
        vol_spike = vol > (self.vol_mult * vol_ma)
        if not vol_high:
            return

        # ---------- Trend context ----------
        uptrend = price > ema200 and ema50 > self.ema50[-5] if len(self.ema50) > 5 else False
        downtrend = price < ema200 and ema50 < self.ema50[-5] if len(self.ema50) > 5 else False

        # ---------- Liquidation cascade proxy ----------
        long_cascade = (low < self.data.Low[-2]) and (self.body[-1] > atr * 0.5) and (self.lower_wick[-1] > self.body[-1] * 0.5) and vol_spike
        short_cascade = (high > self.data.High[-2]) and (self.body[-1] > atr * 0.5) and (self.upper_wick[-1] > self.body[-1] * 0.5) and vol_spike

        # ---------- Exhaustion confirmation ----------
        long_exhaustion = rsi < self.rsi_oversold
        short_exhaustion = rsi > self.rsi_overbought

        # ---------- Entry logic ----------
        if uptrend and long_cascade and long_exhaustion:
            stop = low - self.atr_stop_mult * atr
            target = price + self.atr_target_mult * atr
            risk = price - stop
            if risk <= 0:
                return
            reward = target - price
            if reward / risk < 2.0:
                print(f"⚠️ Moon Dev: RR < 2, skipping long 🌙 ({reward/risk:.2f})")
                return
            size_factor = 0.5 if atr > 2 * atr_ma else 1.0
            risk_amount = self.equity * self.risk_pct * size_factor
            size = int(round(risk_amount / risk))
            if size < 1:
                size = 1
            print(f"🌙 Moon Dev: 🟢 LONG entry @ {price:.2f} | stop {stop:.2f} | target {target:.2f} | size {size} 🚀")
            self.buy(size=size)
            self.entry_bar = len(self.data)
            self.entry_price = price
            self.stop_price = stop
            self.target_price = target

        elif downtrend and short_cascade and short_exhaustion:
            stop = high + self.atr_stop_mult * atr
            target = price - self.atr_target_mult * atr
            risk = stop - price
            if risk <= 0:
                return
            reward = price - target
            if reward / risk < 2.0:
                print(f"⚠️ Moon Dev: RR < 2, skipping short 🌙 ({reward/risk:.2f})")
                return
            size_factor = 0.5 if atr > 2 * atr_ma else 1.0
            risk_amount = self.equity * self.risk_pct * size_factor
            size = int(round(risk_amount / risk))
            if size < 1:
                size = 1
            print(f"🌙 Moon Dev: 🔴 SHORT entry @ {price:.2f} | stop {stop:.2f} | target {target:.2f} | size {size} 🚀")
            self.sell(size=size)
            self.entry_bar = len(self.data)
            self.entry_price = price
            self.stop_price = stop
            self.target_price = target


print("🌙 Moon Dev: Starting backtest... ✨")
bt = Backtest(data, VolatilityLiquidation, cash=1_000_000, commission=0.001, exclusive_orders=True)
stats = bt.run()
print(stats)
print(stats._strategy)