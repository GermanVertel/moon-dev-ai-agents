import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

print("🌙 Moon Dev Backtest AI initializing... ✨")

# Load data
data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Map columns to proper case
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

print(f"🚀 Data loaded: {len(data)} rows 🌙")
print(f"📊 Columns: {list(data.columns)}")


class ChikouVolatility(Strategy):
    # Strategy parameters
    bb_length = 20
    bb_std = 2.0
    chikou_lag = 26
    atr_period = 14
    ema_period = 200
    squeeze_lookback = 50
    risk_pct = 0.02

    def init(self):
        print("🌙 Initializing ChikouVolatility indicators... ✨")

        close = self.data.Close
        high = self.data.High
        low = self.data.Low

        # Bollinger Bands
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS, close, timeperiod=self.bb_length,
            nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0,
            name=['BB_Upper', 'BB_Middle', 'BB_Lower']
        )

        # Bollinger Bandwidth
        def bandwidth(upper, lower, middle):
            return (upper - lower) / middle
        self.bbw = self.I(bandwidth, self.bb_upper, self.bb_lower, self.bb_middle, name='BBW')

        # ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period, name='ATR')

        # 200 EMA for trend bias
        self.ema200 = self.I(talib.EMA, close, timeperiod=self.ema_period, name='EMA200')

        # Chikou Span (close shifted back 26) — for signal logic we compare
        # current close vs close[26 bars ago]
        self.close_arr = close

        # Squeeze detection: rolling min of bandwidth
        def rolling_min_bbw(bbw, window):
            s = pd.Series(bbw)
            return s.rolling(window).min().values
        self.bbw_min = self.I(rolling_min_bbw, self.bbw, self.squeeze_lookback, name='BBW_Min')

        print("✅ Indicators ready! 🌙")

    def next(self):
        i = len(self.data) - 1
        lag = self.chikou_lag

        # Need enough history
        if i < lag + self.squeeze_lookback + 5:
            return

        price = self.data.Close[-1]
        prev_price = self.data.Close[-2]

        # Current values
        bb_u = self.bb_upper[-1]
        bb_m = self.bb_middle[-1]
        bb_l = self.bb_lower[-1]
        bbw_now = self.bbw[-1]
        bbw_prev = self.bbw[-2]
        bbw_min_recent = self.bbw_min[-1]
        atr = self.atr[-1]
        ema200 = self.ema200[-1]

        # Chikou Span: current close vs close 26 bars ago
        chikou_now = price
        hist_price_now = self.data.Close[-lag]

        chikou_prev = prev_price
        hist_price_prev = self.data.Close[-lag - 1]

        # Chikou cross signals
        chikou_bull_cross = (chikou_prev <= hist_price_prev) and (chikou_now > hist_price_now)
        chikou_bear_cross = (chikou_prev >= hist_price_prev) and (chikou_now < hist_price_now)

        # Chikou position (for exit)
        chikou_above = chikou_now > hist_price_now
        chikou_below = chikou_now < hist_price_now

        # Squeeze detection: bandwidth was recently at a low
        was_squeezed = bbw_min_recent < bbw_now * 0.85
        # Expansion: bandwidth rising
        expanding = bbw_now > bbw_prev

        # Trend bias
        bull_bias = price > ema200
        bear_bias = price < ema200

        # Chikou displacement filter (avoid over-extended)
        displacement = abs(chikou_now - hist_price_now)
        not_overextended = displacement < 2 * atr

        # ---- ENTRY LOGIC ----
        if not self.position:
            # Long entry
            long_signal = (
                chikou_bull_cross and
                price > bb_m and
                expanding and was_squeezed and
                price > bb_u and
                bull_bias and
                not_overextended
            )
            if long_signal:
                # Risk-based sizing
                stop_dist = max(price - bb_l, 0.5 * atr)
                if stop_dist <= 0:
                    stop_dist = atr
                risk_amount = self.equity * self.risk_pct
                size = int(round(risk_amount / stop_dist))
                if size > 0:
                    sl = price - stop_dist
                    tp = price + 2 * atr
                    self.buy(size=size, sl=sl, tp=tp)
                    print(f"🌙✨ LONG ENTRY @ {price:.2f} | SL: {sl:.2f} | TP: {tp:.2f} | Size: {size} 🚀")

            # Short entry
            short_signal = (
                chikou_bear_cross and
                price < bb_m and
                expanding and was_squeezed and
                price < bb_l and
                bear_bias and
                not_overextended
            )
            if short_signal:
                stop_dist = max(bb_u - price, 0.5 * atr)
                if stop_dist <= 0:
                    stop_dist = atr
                risk_amount = self.equity * self.risk_pct
                size = int(round(risk_amount / stop_dist))
                if size > 0:
                    sl = price + stop_dist
                    tp = price - 2 * atr
                    self.sell(size=size, sl=sl, tp=tp)
                    print(f"🌙✨ SHORT ENTRY @ {price:.2f} | SL: {sl:.2f} | TP: {tp:.2f} | Size: {size} 🚀")

        # ---- EXIT LOGIC ----
        else:
            if self.position.is_long:
                exit_long = (
                    price < bb_m or
                    chikou_below or
                    price <= bb_l
                )
                if exit_long:
                    self.position.close()
                    print(f"🌙 EXIT LONG @ {price:.2f} ✨")

            elif self.position.is_short:
                exit_short = (
                    price > bb_m or
                    chikou_above or
                    price >= bb_u
                )
                if exit_short:
                    self.position.close()
                    print(f"🌙 EXIT SHORT @ {price:.2f} ✨")


# Run backtest
print("🚀 Launching backtest... 🌙")
bt = Backtest(
    data,
    ChikouVolatility,
    cash=1_000_000,
    commission=0.002,
    exclusive=False
)

stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙 Moon Dev Backtest Complete! ✨🚀")