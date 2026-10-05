import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Load data
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')

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
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

print("🌙✨ Moon Dev data loaded:", data.shape)
print(data.head())


class FractalVolatility(Strategy):
    atr_period = 14
    atr_ma_period = 20
    ema_period = 50
    risk_pct = 0.02
    rr_ratio = 2.0
    time_stop_bars = 18
    fractal_lookback = 2

    def init(self):
        print("🌙🚀 Initializing FractalVolatility strategy...")
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period)
        self.atr_ma = self.I(talib.SMA, self.atr, timeperiod=self.atr_ma_period)
        self.ema = self.I(talib.EMA, self.data.Close, timeperiod=self.ema_period)

        # Fractal detection using rolling max/min
        n = self.fractal_lookback
        high = self.data.High
        low = self.data.Low

        # Up fractal: high[i] > highs on both sides
        up_fractal = np.zeros(len(high), dtype=bool)
        down_fractal = np.zeros(len(low), dtype=bool)

        h = np.array(high)
        l = np.array(low)

        for i in range(n, len(h) - n):
            window_h = h[i - n:i + n + 1]
            if h[i] == window_h.max() and (window_h == h[i]).sum() == 1:
                up_fractal[i] = True
            window_l = l[i - n:i + n + 1]
            if l[i] == window_l.min() and (window_l == l[i]).sum() == 1:
                down_fractal[i] = True

        self.up_fractal = self.I(lambda: up_fractal, name='up_fractal')
        self.down_fractal = self.I(lambda: down_fractal, name='down_fractal')

        # Store fractal levels
        self.last_up_fractal_price = np.full(len(h), np.nan)
        self.last_down_fractal_price = np.full(len(l), np.nan)

        last_up = np.nan
        last_down = np.nan
        for i in range(len(h)):
            if up_fractal[i]:
                last_up = h[i]
            if down_fractal[i]:
                last_down = l[i]
            self.last_up_fractal_price[i] = last_up
            self.last_down_fractal_price[i] = last_down

        self.last_up_fractal_price = self.I(lambda: self.last_up_fractal_price, name='last_up_fractal')
        self.last_down_fractal_price = self.I(lambda: self.last_down_fractal_price, name='last_down_fractal')

        self.entry_bar = None
        self.stop_price = None
        self.target_price = None

        print("🌙✨ Indicators initialized!")

    def next(self):
        price = self.data.Close[-1]
        i = len(self.data) - 1

        if i < self.ema_period + 5:
            return

        atr_val = self.atr[-1]
        atr_ma_val = self.atr_ma[-1]
        ema_val = self.ema[-1]

        if np.isnan(atr_val) or np.isnan(atr_ma_val) or np.isnan(ema_val):
            return

        # Volatility regime: elevated
        vol_elevated = atr_val > atr_ma_val

        # Manage open position
        if self.position:
            bars_held = i - self.entry_bar if self.entry_bar else 0

            # Volatility decay exit
            if atr_val <= atr_ma_val:
                print(f"🌙💨 Volatility decay exit at {price:.2f}")
                self.position.close()
                self.entry_bar = None
                return

            # Time stop
            if bars_held >= self.time_stop_bars:
                print(f"⏰ Time stop exit at {price:.2f} after {bars_held} bars")
                self.position.close()
                self.entry_bar = None
                return

            # Target hit
            if self.position.is_long and price >= self.target_price:
                print(f"🎯 Target hit (long) at {price:.2f}")
                self.position.close()
                self.entry_bar = None
                return
            if self.position.is_short and price <= self.target_price:
                print(f"🎯 Target hit (short) at {price:.2f}")
                self.position.close()
                self.entry_bar = None
                return

            # Stop hit handled by backtesting.py via sl
            return

        # No position - look for entries
        if not vol_elevated:
            return

        last_up = self.last_up_fractal_price[-1]
        last_down = self.last_down_fractal_price[-1]

        if np.isnan(last_up) or np.isnan(last_down):
            return

        # Long entry: down fractal formed, price closes above fractal high (last_up as proxy for pattern break)
        # Confirmation: current close > previous close (bullish bar) and price crossed above recent down fractal high region
        # Use last_up fractal as the confirmation level
        prev_close = self.data.Close[-2]

        # Long: recent down fractal, price now breaking above the last up fractal (confirmation)
        if not np.isnan(last_down) and price > last_up and prev_close <= last_up and price > ema_val * 0.98:
            # Risk: stop below last down fractal
            stop = last_down
            risk = price - stop
            if risk <= 0:
                return
            target = price + risk * self.rr_ratio

            # Position sizing based on risk
            equity = self.equity
            risk_amount = equity * self.risk_pct
            size = int(round(risk_amount / risk))
            if size < 1:
                return

            print(f"🌙🟢 LONG entry at {price:.2f} | stop={stop:.2f} target={target:.2f} size={size}")
            self.buy(size=size, sl=stop, tp=target)
            self.entry_bar = i
            self.stop_price = stop
            self.target_price = target

        # Short: recent up fractal, price breaks below last down fractal
        elif not np.isnan(last_up) and price < last_down and prev_close >= last_down and price < ema_val * 1.02:
            stop = last_up
            risk = stop - price
            if risk <= 0:
                return
            target = price - risk * self.rr_ratio

            equity = self.equity
            risk_amount = equity * self.risk_pct
            size = int(round(risk_amount / risk))
            if size < 1:
                return

            print(f"🌙🔴 SHORT entry at {price:.2f} | stop={stop:.2f} target={target:.2f} size={size}")
            self.sell(size=size, sl=stop, tp=target)
            self.entry_bar = i
            self.stop_price = stop
            self.target_price = target


bt = Backtest(data, FractalVolatility, cash=1_000_000, commission=0.001, exclusive_orders=True)
stats = bt.run()
print(stats)
print(stats._strategy)