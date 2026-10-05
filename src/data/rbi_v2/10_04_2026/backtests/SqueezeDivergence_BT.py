import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Load and clean data
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'datetime': 'Date',
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})
data['Date'] = pd.to_datetime(data['Date'])
data = data.set_index('Date')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

print("🌙 Moon Dev SqueezeDivergence backtest initializing... ✨")
print(f"📊 Data loaded: {len(data)} bars from {data.index[0]} to {data.index[-1]} 🚀")


class SqueezeDivergence(Strategy):
    # BB parameters
    bb_period = 20
    bb_dev = 2.0
    bb_width_avg_period = 20
    # MACD parameters
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    # ATR
    atr_period = 14
    atr_mult = 1.5
    # Risk
    risk_pct = 0.02
    # Swing lookback
    swing_lookback = 10

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low

        # Bollinger Bands
        self.bb_upper, self.bb_mid, self.bb_lower = self.I(
            talib.BBANDS, close, timeperiod=self.bb_period,
            nbdevup=self.bb_dev, nbdevdn=self.bb_dev, matype=0
        )

        # BB Width
        def bb_width(upper, lower, mid):
            return (upper - lower) / mid

        self.bb_width = self.I(bb_width, self.bb_upper, self.bb_lower, self.bb_mid)
        self.bb_width_avg = self.I(talib.SMA, self.bb_width, timeperiod=self.bb_width_avg_period)

        # BB midpoint slope (current vs N bars ago)
        def mid_slope(mid, n):
            return mid - np.roll(mid, n)

        self.bb_mid_slope = self.I(mid_slope, self.bb_mid, 5)

        # MACD
        self.macd, self.macd_signal_line, self.macd_hist = self.I(
            talib.MACD, close,
            fastperiod=self.macd_fast,
            slowperiod=self.macd_slow,
            signalperiod=self.macd_signal
        )

        # ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        # Swing high/low
        self.swing_high = self.I(talib.MAX, high, timeperiod=self.swing_lookback)
        self.swing_low = self.I(talib.MIN, low, timeperiod=self.swing_lookback)

        print("🌙 Indicators initialized: BB, BB Width, MACD, ATR, Swing H/L ✨")

    def next(self):
        if len(self.data) < 60:
            return

        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]

        upper = self.bb_upper[-1]
        mid = self.bb_mid[-1]
        lower = self.bb_lower[-1]

        width = self.bb_width[-1]
        width_prev = self.bb_width[-2]
        width_avg = self.bb_width_avg[-1]

        mid_slope = self.bb_mid_slope[-1]

        macd = self.macd[-1]
        macd_prev = self.macd[-2]
        sig = self.macd_signal_line[-1]
        sig_prev = self.macd_signal_line[-2]

        atr = self.atr[-1]
        swing_low = self.swing_low[-1]
        swing_high = self.swing_high[-1]

        if np.isnan(width) or np.isnan(width_avg) or np.isnan(macd) or np.isnan(atr):
            return

        # ===== EXIT LOGIC =====
        if self.position:
            if self.position.is_long:
                # Exit long: MACD crosses below signal OR price closes below BB mid
                macd_cross_down = macd_prev >= sig_prev and macd < sig
                close_below_mid = price < mid
                if macd_cross_down or close_below_mid:
                    self.position.close()
                    print(f"🌙 EXIT LONG @ {price:.2f} | MACDxDown={macd_cross_down} BelowMid={close_below_mid} ✨")
            elif self.position.is_short:
                # Exit short: MACD crosses above signal OR price closes above BB mid
                macd_cross_up = macd_prev <= sig_prev and macd > sig
                close_above_mid = price > mid
                if macd_cross_up or close_above_mid:
                    self.position.close()
                    print(f"🌙 EXIT SHORT @ {price:.2f} | MACDxUp={macd_cross_up} AboveMid={close_above_mid} ✨")

        # ===== ENTRY LOGIC =====
        if self.position:
            return

        # LONG ENTRY (Uptrend Continuation)
        width_expanding = width > width_avg and width > width_prev
        mid_rising = mid_slope > 0
        macd_above_zero = macd > 0
        macd_above_signal = macd > sig
        price_above_mid = price > mid
        # Pullback near BB midpoint: price within 1 ATR above mid
        near_mid = price > mid and (price - mid) < atr

        if (width_expanding and mid_rising and macd_above_zero
                and macd_above_signal and price_above_mid and near_mid):
            stop = min(swing_low, mid - self.atr_mult * atr)
            risk = price - stop
            if risk > 0:
                # 1:2 RR target
                target = price + 2 * risk
                # Size: 1,000,000 units base, scaled by risk
                size = 1000000
                self.buy(size=size, sl=stop, tp=target)
                print(f"🚀 LONG ENTRY @ {price:.2f} | SL={stop:.2f} TP={target:.2f} | "
                      f"Width={width:.4f} Avg={width_avg:.4f} MACD={macd:.2f} ✨")

        # SHORT ENTRY (Downtrend Continuation)
        width_tightening = width < width_avg and width < width_prev
        mid_falling = mid_slope < 0
        macd_below_zero = macd < 0
        macd_below_signal = macd < sig
        price_below_mid = price < mid
        near_mid_short = price < mid and (mid - price) < atr

        if (width_tightening and mid_falling and macd_below_zero
                and macd_below_signal and price_below_mid and near_mid_short):
            stop = max(swing_high, mid + self.atr_mult * atr)
            risk = stop - price
            if risk > 0:
                target = price - 2 * risk
                size = 1000000
                self.sell(size=size, sl=stop, tp=target)
                print(f"🔻 SHORT ENTRY @ {price:.2f} | SL={stop:.2f} TP={target:.2f} | "
                      f"Width={width:.4f} Avg={width_avg:.4f} MACD={macd:.2f} ✨")


bt = Backtest(data, SqueezeDivergence, cash=1000000, commission=0.0002, exclusive=False)
stats = bt.run()
print(stats)
print(stats._strategy)