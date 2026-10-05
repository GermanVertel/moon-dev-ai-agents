import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# ============================================================
# 🌙 MOON DEV ADAPTIVE FADEOUT STRATEGY 🌙
# ============================================================

print("🌙✨ Moon Dev Backtest AI initializing AdaptiveFadeout... 🚀")

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

# Ensure datetime index
if 'datetime' in data.columns:
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')

print(f"🌙 Data loaded: {len(data)} bars ✨")
print(f"🚀 Columns: {list(data.columns)}")


class AdaptiveFadeout(Strategy):
    # Strategy parameters
    donchian_period = 20
    atr_period = 14
    atr_avg_period = 20
    bb_period = 20
    bb_std = 2.0
    rsi_period = 14
    atr_expansion_threshold = 1.2
    risk_pct = 0.01
    sl_atr_mult = 0.5
    tp_atr_mult = 1.5
    time_stop_bars = 5
    trail_atr_mult = 1.0

    def init(self):
        print("🌙 Initializing indicators... ✨")
        close = self.data.Close
        high = self.data.High
        low = self.data.Low

        # Donchian Channels (use prior bar to avoid lookahead)
        self.dc_upper = self.I(talib.MAX, high, timeperiod=self.donchian_period)
        self.dc_lower = self.I(talib.MIN, low, timeperiod=self.donchian_period)

        # ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)
        self.atr_avg = self.I(talib.SMA, self.atr, timeperiod=self.atr_avg_period)

        # Bollinger Bands - use custom function to extract individual bands
        def bb_upper_func(c):
            upper, middle, lower = talib.BBANDS(c, timeperiod=self.bb_period,
                                                nbdevup=self.bb_std, nbdevdn=self.bb_std)
            return upper

        def bb_middle_func(c):
            upper, middle, lower = talib.BBANDS(c, timeperiod=self.bb_period,
                                                nbdevup=self.bb_std, nbdevdn=self.bb_std)
            return middle

        def bb_lower_func(c):
            upper, middle, lower = talib.BBANDS(c, timeperiod=self.bb_period,
                                                nbdevup=self.bb_std, nbdevdn=self.bb_std)
            return lower

        self.bb_upper = self.I(bb_upper_func, close)
        self.bb_middle = self.I(bb_middle_func, close)
        self.bb_lower = self.I(bb_lower_func, close)

        # BBW = (upper - lower) / middle
        self.bbw = self.I(lambda u, l, m: (u - l) / np.where(m == 0, np.nan, m),
                          self.bb_upper, self.bb_lower, self.bb_middle)

        # RSI
        self.rsi = self.I(talib.RSI, close, timeperiod=self.rsi_period)

        # EMA for trailing stop
        self.ema20 = self.I(talib.EMA, close, timeperiod=20)

        print("🌙 Indicators ready! 🚀")

    def next(self):
        # Need enough bars
        if len(self.data) < max(self.donchian_period, self.atr_avg_period, self.bb_period) + 5:
            return

        price = self.data.Close[-1]
        prev_price = self.data.Close[-2]

        # Current indicator values
        dc_upper = self.dc_upper[-1]
        dc_lower = self.dc_lower[-1]
        dc_upper_prev = self.dc_upper[-2]
        dc_lower_prev = self.dc_lower[-2]
        atr = self.atr[-1]
        atr_avg = self.atr_avg[-1]
        bbw = self.bbw[-1]
        bbw_prev = self.bbw[-2]
        rsi = self.rsi[-1]

        if (np.isnan(atr) or np.isnan(atr_avg) or np.isnan(bbw)
                or np.isnan(bbw_prev) or np.isnan(rsi)
                or np.isnan(dc_upper) or np.isnan(dc_lower)
                or np.isnan(dc_upper_prev) or np.isnan(dc_lower_prev)):
            return

        # Volatility confirmation check
        atr_expanding = atr > self.atr_expansion_threshold * atr_avg
        bbw_expanding = bbw > bbw_prev
        volatility_confirms = atr_expanding and bbw_expanding

        # ============================================================
        # EXIT LOGIC (manage open trades)
        # ============================================================
        if self.position:
            # Get entry price from last trade (position has no .entry_price)
            if len(self.trades) > 0:
                entry_price = self.trades[-1].entry_price
                entry_bar = self.trades[-1].entry_bar
            else:
                entry_price = price
                entry_bar = len(self.data) - 1

            bars_held = len(self.data) - 1 - entry_bar

            if self.position.is_short:
                # Stop loss: 0.5 ATR above entry
                sl = entry_price + self.sl_atr_mult * atr
                tp = entry_price - self.tp_atr_mult * atr

                # Time stop
                if bars_held >= self.time_stop_bars:
                    print(f"🌙⏰ Time stop hit on SHORT! Closing at {price:.2f} ✨")
                    self.position.close()
                    return

                # Trailing stop: once 1 ATR in favor, trail with EMA20
                if price < entry_price - self.trail_atr_mult * atr:
                    if price > self.ema20[-1]:
                        print(f"🌙📉 Trailing stop triggered on SHORT at {price:.2f} 🚀")
                        self.position.close()
                        return

                if price >= sl:
                    print(f"🌙🛑 SHORT stop loss hit at {price:.2f} ✨")
                    self.position.close()
                    return

                if price <= tp:
                    print(f"🌙🎯 SHORT take profit hit at {price:.2f} 🚀")
                    self.position.close()
                    return

            elif self.position.is_long:
                sl = entry_price - self.sl_atr_mult * atr
                tp = entry_price + self.tp_atr_mult * atr

                if bars_held >= self.time_stop_bars:
                    print(f"🌙⏰ Time stop hit on LONG! Closing at {price:.2f} ✨")
                    self.position.close()
                    return

                if price > entry_price + self.trail_atr_mult * atr:
                    if price < self.ema20[-1]:
                        print(f"🌙📈 Trailing stop triggered on LONG at {price:.2f} 🚀")
                        self.position.close()
                        return

                if price <= sl:
                    print(f"🌙🛑 LONG stop loss hit at {price:.2f} ✨")
                    self.position.close()
                    return

                if price >= tp:
                    print(f"🌙🎯 LONG take profit hit at {price:.2f} 🚀")
                    self.position.close()
                    return

        # ============================================================
        # ENTRY LOGIC
        # ============================================================
        if not self.position:
            # ---- SHORT: False Upside Breakout ----
            broke_upper = prev_price <= dc_upper_prev and price > dc_upper
            if broke_upper and not volatility_confirms and rsi > 60:
                sl_price = price + self.sl_atr_mult * atr
                risk_per_unit = sl_price - price
                if risk_per_unit > 0:
                    risk_amount = self.equity * self.risk_pct
                    size = risk_amount / risk_per_unit
                    size = max(1, int(round(size)))
                    # Cap size to available cash (fraction-based sizing)
                    max_size = (self.equity * 0.95) / price
                    size = min(size, int(max_size))
                    if size >= 1:
                        print(f"🌙🔻 SHORT signal! False upside breakout at {price:.2f} | RSI={rsi:.1f} | ATR={atr:.2f} | Size={size} 🚀")
                        self.sell(size=size)

            # ---- LONG: False Downside Breakout ----
            broke_lower = prev_price >= dc_lower_prev and price < dc_lower
            if broke_lower and not volatility_confirms and rsi < 40:
                sl_price = price - self.sl_atr_mult * atr
                risk_per_unit = price - sl_price
                if risk_per_unit > 0:
                    risk_amount = self.equity * self.risk_pct
                    size = risk_amount / risk_per_unit
                    size = max(1, int(round(size)))
                    # Cap size to available cash (fraction-based sizing)
                    max_size = (self.equity * 0.95) / price
                    size = min(size, int(max_size))
                    if size >= 1:
                        print(f"🌙🔺 LONG signal! False downside breakout at {price:.2f} | RSI={rsi:.1f} | ATR={atr:.2f} | Size={size} 🚀")
                        self.buy(size=size)


# ============================================================
# 🌙 RUN BACKTEST 🚀
# ============================================================
print("🌙✨ Starting Moon Dev AdaptiveFadeout backtest... 🚀")
bt = Backtest(data, AdaptiveFadeout, cash=1_000_000, commission=0.002)
stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙✨ Backtest complete! 🚀")