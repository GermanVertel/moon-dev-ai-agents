import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# ============================================================
# 🌙 Moon Dev's VolumeBandDivergence Backtest 🌙
# ============================================================

data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"

print("🌙 Moon Dev loading data from the cosmos... ✨")
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
    'volume': 'Volume',
})

# Set datetime index
if 'datetime' in data.columns:
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')

print(f"🚀 Moon Dev data ready! Shape: {data.shape}, Range: {data.index[0]} -> {data.index[-1]} 🌙")


class VolumeBandDivergence(Strategy):
    # Strategy parameters
    bb_period = 20
    bb_std = 2.0
    swing_lookback = 7
    atr_period = 14
    atr_mult = 1.5
    risk_pct = 0.02
    rsi_period = 14
    rsi_long_max = 45
    rsi_short_min = 55
    time_stop_bars = 40
    ema_period = 200

    def init(self):
        print("🌙 Moon Dev initializing indicators... ✨")
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        # Bollinger Bands
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS, close, timeperiod=self.bb_period,
            nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0
        )

        # OBV
        self.obv = self.I(talib.OBV, close, volume)

        # ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        # RSI
        self.rsi = self.I(talib.RSI, close, timeperiod=self.rsi_period)

        # EMA 200 trend filter
        self.ema200 = self.I(talib.EMA, close, timeperiod=self.ema_period)

        # Swing highs/lows
        self.swing_high = self.I(talib.MAX, high, timeperiod=self.swing_lookback)
        self.swing_low = self.I(talib.MIN, low, timeperiod=self.swing_lookback)

        # Track entry bar for time stop
        self.entry_bar = None

        print("🚀 Moon Dev indicators online! 🌙")

    def next(self):
        i = len(self.data) - 1
        if i < max(self.bb_period, self.swing_lookback, self.atr_period, self.ema_period) + 2:
            return

        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]

        upper = self.bb_upper[-1]
        middle = self.bb_middle[-1]
        lower = self.bb_lower[-1]
        atr = self.atr[-1]
        rsi = self.rsi[-1]
        ema200 = self.ema200[-1]

        if np.isnan(upper) or np.isnan(lower) or np.isnan(atr):
            return

        # ============================================================
        # Manage open positions
        # ============================================================
        if self.position:
            # Time stop
            if self.entry_bar is not None and (i - self.entry_bar) >= self.time_stop_bars:
                print(f"⏰ Moon Dev time stop hit! Closing position at {price:.2f} 🌙")
                self.position.close()
                self.entry_bar = None
                return

            if self.position.is_long:
                # Take profit at middle band
                if price >= middle:
                    print(f"✨ Moon Dev LONG TP hit at middle band {middle:.2f}! Closing at {price:.2f} 🌙")
                    self.position.close()
                    self.entry_bar = None
                    return
            elif self.position.is_short:
                # Take profit at middle band
                if price <= middle:
                    print(f"✨ Moon Dev SHORT TP hit at middle band {middle:.2f}! Closing at {price:.2f} 🌙")
                    self.position.close()
                    self.entry_bar = None
                    return
            return

        # ============================================================
        # LONG ENTRY: lower band touch + positive OBV divergence
        # ============================================================
        if low <= lower:
            # Check OBV divergence over lookback: price lower low, OBV higher low
            lookback = self.swing_lookback
            if i >= lookback:
                price_now = self.data.Low[-1]
                price_prev = self.data.Low[-1 - lookback]
                obv_now = self.obv[-1]
                obv_prev = self.obv[-1 - lookback]

                price_lower_low = price_now <= price_prev
                obv_higher_low = obv_now > obv_prev

                # RSI filter
                rsi_ok = rsi < self.rsi_long_max

                if price_lower_low and obv_higher_low and rsi_ok:
                    stop_price = low - (self.atr_mult * atr)
                    risk_per_unit = price - stop_price
                    if risk_per_unit <= 0:
                        return

                    risk_amount = self.equity * self.risk_pct
                    size = int(round(risk_amount / risk_per_unit))
                    if size < 1:
                        size = 1

                    print(f"🌙✨ Moon Dev LONG SIGNAL! Price={price:.2f}, Lower={lower:.2f}, "
                          f"OBV div confirmed, RSI={rsi:.1f}, Size={size} 🚀")
                    self.buy(size=size, sl=stop_price)
                    self.entry_bar = i
                    return

        # ============================================================
        # SHORT ENTRY: upper band touch + negative OBV divergence
        # ============================================================
        if high >= upper:
            lookback = self.swing_lookback
            if i >= lookback:
                price_now = self.data.High[-1]
                price_prev = self.data.High[-1 - lookback]
                obv_now = self.obv[-1]
                obv_prev = self.obv[-1 - lookback]

                price_higher_high = price_now >= price_prev
                obv_lower_high = obv_now < obv_prev

                rsi_ok = rsi > self.rsi_short_min

                if price_higher_high and obv_lower_high and rsi_ok:
                    stop_price = high + (self.atr_mult * atr)
                    risk_per_unit = stop_price - price
                    if risk_per_unit <= 0:
                        return

                    risk_amount = self.equity * self.risk_pct
                    size = int(round(risk_amount / risk_per_unit))
                    if size < 1:
                        size = 1

                    print(f"🌙✨ Moon Dev SHORT SIGNAL! Price={price:.2f}, Upper={upper:.2f}, "
                          f"OBV div confirmed, RSI={rsi:.1f}, Size={size} 🚀")
                    self.sell(size=size, sl=stop_price)
                    self.entry_bar = i
                    return


print("🌙 Moon Dev launching backtest... 🚀")
bt = Backtest(
    data,
    VolumeBandDivergence,
    cash=1_000_000,
    commission=0.001,
    exclusive_orders=True
)

stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙 Moon Dev backtest complete! ✨🚀")