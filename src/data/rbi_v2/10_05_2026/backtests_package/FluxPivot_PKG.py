import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Load data
data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper column mapping
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

print("🌙 Moon Dev FluxPivot Backtest Starting... ✨")
print(f"🚀 Data loaded: {len(data)} bars from {data.index[0]} to {data.index[-1]}")


class FluxPivot(Strategy):
    # MACD params
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    # ATR params
    atr_period = 14
    atr_mult = 1.5
    # Order flow lookback
    delta_lookback = 5
    # Risk
    risk_pct = 0.02
    rr_ratio = 2.0
    # Time stop
    time_stop_bars = 15

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        # MACD
        macd, macd_sig, macd_hist = talib.MACD(
            close, fastperiod=self.macd_fast, slowperiod=self.macd_slow, signalperiod=self.macd_signal
        )
        self.macd = self.I(lambda: macd, name='MACD')
        self.macd_sig = self.I(lambda: macd_sig, name='MACD_Signal')
        self.macd_hist = self.I(lambda: macd_hist, name='MACD_Hist')

        # ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period, name='ATR')

        # Cumulative Delta proxy: use close position within bar range * volume
        # (buying pressure when close near high, selling when close near low)
        rng = (high - low).replace(0, np.nan)
        delta_bar = ((close - low) / rng - 0.5) * 2 * volume
        delta_bar = delta_bar.fillna(0)
        cum_delta = delta_bar.cumsum()
        self.cum_delta = self.I(lambda: cum_delta.values, name='CumDelta')

        # Delta slope over lookback
        self.delta_slope = self.I(
            lambda: pd.Series(cum_delta.values).diff(self.delta_lookback).values,
            name='Delta_Slope'
        )

        # Swing high/low for stops
        self.swing_low = self.I(talib.MIN, low, timeperiod=10, name='SwingLow')
        self.swing_high = self.I(talib.MAX, high, timeperiod=10, name='SwingHigh')

        print("🌙✨ Indicators initialized: MACD, ATR, CumDelta, Swing levels")

    def next(self):
        price = self.data.Close[-1]
        macd_now = self.macd[-1]
        macd_prev = self.macd[-2]
        slope = self.delta_slope[-1]
        atr = self.atr[-1]

        if np.isnan(macd_now) or np.isnan(macd_prev) or np.isnan(atr) or atr <= 0:
            return

        # Entry logic
        if not self.position:
            # Bullish: MACD crosses above zero AND delta rising
            if macd_prev < 0 and macd_now > 0 and slope > 0:
                sl = self.swing_low[-1] - 0.5 * atr
                risk = price - sl
                if risk <= 0:
                    return
                tp = price + self.rr_ratio * risk
                size = int(round((self.equity * self.risk_pct) / risk))
                if size <= 0:
                    return
                print(f"🌙🚀 LONG FluxPivot signal! Price={price:.2f} SL={sl:.2f} TP={tp:.2f} Size={size} ✨")
                self.buy(size=size, sl=sl, tp=tp)

            # Bearish: MACD crosses below zero AND delta falling
            elif macd_prev > 0 and macd_now < 0 and slope < 0:
                sl = self.swing_high[-1] + 0.5 * atr
                risk = sl - price
                if risk <= 0:
                    return
                tp = price - self.rr_ratio * risk
                size = int(round((self.equity * self.risk_pct) / risk))
                if size <= 0:
                    return
                print(f"🌙🔻 SHORT FluxPivot signal! Price={price:.2f} SL={sl:.2f} TP={tp:.2f} Size={size} ✨")
                self.sell(size=size, sl=sl, tp=tp)

        else:
            # Exit signal: MACD crosses back against position
            bars_held = len(self.data) - self.trades[-1].entry_bar if self.trades else 0

            if self.position.is_long:
                if macd_prev > 0 and macd_now < 0:
                    print(f"🌙 Exit LONG: MACD crossed below zero at {price:.2f}")
                    self.position.close()
                elif bars_held >= self.time_stop_bars:
                    print(f"🌙⏰ Time-stop exit LONG at {price:.2f}")
                    self.position.close()

            elif self.position.is_short:
                if macd_prev < 0 and macd_now > 0:
                    print(f"🌙 Exit SHORT: MACD crossed above zero at {price:.2f}")
                    self.position.close()
                elif bars_held >= self.time_stop_bars:
                    print(f"🌙⏰ Time-stop exit SHORT at {price:.2f}")
                    self.position.close()


bt = Backtest(
    data,
    FluxPivot,
    cash=1_000_000,
    commission=0.001,
    exclusive=False,
    trade_on_close=True,
)

stats = bt.run()
print(stats)
print(stats._strategy)