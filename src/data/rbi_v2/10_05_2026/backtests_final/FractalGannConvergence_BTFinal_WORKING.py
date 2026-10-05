import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

print("🌙✨ Moon Dev Backtest AI initializing... FractalGannConvergence loading! 🚀")

# Load data
data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper column mapping
data.columns = [col.capitalize() for col in data.columns]
if 'Datetime' in data.columns:
    data = data.set_index('Datetime')
    data.index = pd.to_datetime(data.index)

print(f"🌙 Data loaded: {len(data)} bars | Columns: {list(data.columns)} ✨")


class FractalGannConvergence(Strategy):
    # Parameters
    atr_period = 14
    rr_ratio = 2.0
    risk_pct = 0.01
    swing_lookback = 5
    tolerance_pct = 0.002  # 0.2% price tolerance for convergence

    def init(self):
        print("🌙 Initializing indicators... ✨")
        # ATR for volatility / stop sizing
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close,
                          timeperiod=self.atr_period)

        # 5-bar fractal detection using rolling MAX/MIN
        self.roll_min_low = self.I(talib.MIN, self.data.Low, timeperiod=5)
        self.roll_max_high = self.I(talib.MAX, self.data.High, timeperiod=5)

        # Gann swing structure: prior swing high / low over lookback
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=self.swing_lookback)
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=self.swing_lookback)

        # SMA of ATR for volatility filter
        self.atr_avg = self.I(talib.SMA, self.atr, timeperiod=50)

        print("🌙 Indicators ready! 🚀")

    def next(self):
        i = len(self.data) - 1
        if i < 10:
            return

        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        atr = self.atr[-1]

        if np.isnan(atr) or atr <= 0:
            return

        # ---- Volatility filter ----
        atr_avg = self.atr_avg[-1]
        if not np.isnan(atr_avg) and atr_avg > 0:
            if atr > 2.0 * atr_avg:
                if self.position:
                    pass
                return  # skip extreme volatility

        # ---- Fractal detection (5-bar) ----
        # Bullish fractal: bar at i-2 is the middle, lowest low of the 5-bar window
        # Window bars: i-4, i-3, i-2, i-1, i
        if i >= 5:
            mid_low = self.data.Low[-3]   # middle bar of last 5
            window_lows = [self.data.Low[-5], self.data.Low[-4], self.data.Low[-3],
                           self.data.Low[-2], self.data.Low[-1]]
            mid_high = self.data.High[-3]
            window_highs = [self.data.High[-5], self.data.High[-4], self.data.High[-3],
                            self.data.High[-2], self.data.High[-1]]

            bullish_fractal = (mid_low == min(window_lows)) and \
                              (self.data.Low[-4] > mid_low) and \
                              (self.data.Low[-2] > mid_low)

            bearish_fractal = (mid_high == max(window_highs)) and \
                              (self.data.High[-4] < mid_high) and \
                              (self.data.High[-2] < mid_high)
        else:
            bullish_fractal = False
            bearish_fractal = False

        # ---- Gann signal ----
        # Bullish Gann: price breaks above prior swing high with higher low structure
        prior_swing_high = self.swing_high[-2]
        prior_swing_low = self.swing_low[-2]

        bullish_gann = (not np.isnan(prior_swing_high)) and (price > prior_swing_high) and \
                       (self.data.Low[-1] > self.data.Low[-2])

        bearish_gann = (not np.isnan(prior_swing_low)) and (price < prior_swing_low) and \
                       (self.data.High[-1] < self.data.High[-2])

        # ---- Convergence: fractal + Gann near same price ----
        long_convergence = False
        short_convergence = False

        if bullish_fractal or bearish_fractal:
            fractal_ref_low = self.data.Low[-3] if bullish_fractal else None
            fractal_ref_high = self.data.High[-3] if bearish_fractal else None

            if bullish_fractal and bullish_gann:
                if abs(price - fractal_ref_low) / price <= self.tolerance_pct or \
                   abs(price - prior_swing_high) / price <= self.tolerance_pct:
                    long_convergence = True

            if bearish_fractal and bearish_gann:
                if abs(price - fractal_ref_high) / price <= self.tolerance_pct or \
                   abs(price - prior_swing_low) / price <= self.tolerance_pct:
                    short_convergence = True

        # ---- Entry logic ----
        if not self.position:
            if long_convergence:
                stop_price = self.data.Low[-3] - atr * 0.5  # fractal low minus buffer
                risk = price - stop_price
                if risk > 0:
                    tp_price = price + risk * self.rr_ratio
                    risk_amount = self.equity * self.risk_pct
                    size = int(round(risk_amount / risk))
                    if size > 0:
                        print(f"🌙🚀 LONG CONVERGENCE! Price={price:.2f} SL={stop_price:.2f} "
                              f"TP={tp_price:.2f} Size={size} ✨")
                        self.buy(size=size, sl=stop_price, tp=tp_price)

            elif short_convergence:
                stop_price = self.data.High[-3] + atr * 0.5
                risk = stop_price - price
                if risk > 0:
                    tp_price = price - risk * self.rr_ratio
                    risk_amount = self.equity * self.risk_pct
                    size = int(round(risk_amount / risk))
                    if size > 0:
                        print(f"🌙🚀 SHORT CONVERGENCE! Price={price:.2f} SL={stop_price:.2f} "
                              f"TP={tp_price:.2f} Size={size} ✨")
                        self.sell(size=size, sl=stop_price, tp=tp_price)

        else:
            # Exit on opposite signal
            if self.position.is_long and short_convergence:
                print(f"🌙🔄 Opposite SHORT signal — closing long ✨")
                self.position.close()
            elif self.position.is_short and long_convergence:
                print(f"🌙🔄 Opposite LONG signal — closing short ✨")
                self.position.close()


print("🌙✨ Running backtest... 🚀")
bt = Backtest(data, FractalGannConvergence, cash=1_000_000, commission=0.001, exclusive_orders=True)
stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙✨ Backtest complete! 🚀")