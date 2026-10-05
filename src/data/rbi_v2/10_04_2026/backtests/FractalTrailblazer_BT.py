import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Load data
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
print("🌙✨ Data loaded successfully! Rows:", len(data), "🚀")


class FractalTrailblazer(Strategy):
    ema_period = 200
    atr_period = 14
    adx_period = 14
    adx_threshold = 20
    risk_pct = 0.01
    atr_mult = 2.0
    time_stop = 20

    def init(self):
        self.ema = self.I(talib.EMA, self.data.Close, timeperiod=self.ema_period)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period)
        self.adx = self.I(talib.ADX, self.data.High, self.data.Low, self.data.Close, timeperiod=self.adx_period)

        # Fractal detection using talib MAX/MIN on shifted windows
        high = self.data.High
        low = self.data.Low
        self.up_fractal = self.I(talib.MAX, high, timeperiod=5)
        self.dn_fractal = self.I(talib.MIN, low, timeperiod=5)

        self.last_up_fractal = np.nan
        self.last_dn_fractal = np.nan
        self.entry_price = None
        self.stop_price = None
        self.bars_since_fractal = 0
        self.trade_dir = 0

        print("🌙 FractalTrailblazer initialized! EMA:", self.ema_period, "ATR:", self.atr_period, "ADX:", self.adx_period, "🚀")

    def next(self):
        i = len(self.data) - 1
        if i < 5:
            return

        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]

        # Confirmed fractal at index i-2 (needs 2 bars after)
        fi = i - 2
        if fi >= 2:
            fh = self.data.High[fi]
            fl = self.data.Low[fi]
            is_up = (fh >= self.data.High[fi-1] and fh >= self.data.High[fi-2] and
                     fh >= self.data.High[fi+1] and fh >= self.data.High[fi+2])
            is_dn = (fl <= self.data.Low[fi-1] and fl <= self.data.Low[fi-2] and
                     fl <= self.data.Low[fi+1] and fl <= self.data.Low[fi+2])
            if is_up:
                self.last_up_fractal = fh
                self.bars_since_fractal = 0
                print(f"🌙✨ New UP fractal confirmed at {fh:.2f} (bar {fi}) 🚀")
            if is_dn:
                self.last_dn_fractal = fl
                self.bars_since_fractal = 0
                print(f"🌙✨ New DOWN fractal confirmed at {fl:.2f} (bar {fi}) 🚀")

        self.bars_since_fractal += 1

        ema = self.ema[-1]
        ema_prev = self.ema[-2]
        atr = self.atr[-1]
        adx = self.adx[-1]

        if np.isnan(ema) or np.isnan(atr) or np.isnan(adx):
            return

        ema_slope_up = ema > ema_prev
        ema_slope_dn = ema < ema_prev

        # Manage open position
        if self.position:
            if self.position.is_long:
                # Update trailing stop to most recent down fractal if higher than current stop
                if not np.isnan(self.last_dn_fractal):
                    new_stop = self.last_dn_fractal
                    if new_stop > self.stop_price:
                        self.stop_price = new_stop
                        print(f"🌙📈 Long trailing stop updated to fractal: {self.stop_price:.2f}")

                # ATR backstop
                atr_stop = self.entry_price - self.atr_mult * atr
                if atr_stop > self.stop_price:
                    self.stop_price = atr_stop

                # Exit checks
                if low <= self.stop_price:
                    print(f"🌙🛑 LONG STOP HIT at {self.stop_price:.2f} | exit price {self.stop_price:.2f}")
                    self.position.close()
                    self.trade_dir = 0
                elif self.bars_since_fractal >= self.time_stop:
                    print(f"🌙⏰ TIME STOP on long after {self.time_stop} bars")
                    self.position.close()
                    self.trade_dir = 0

            elif self.position.is_short:
                if not np.isnan(self.last_up_fractal):
                    new_stop = self.last_up_fractal
                    if new_stop < self.stop_price:
                        self.stop_price = new_stop
                        print(f"🌙📉 Short trailing stop updated to fractal: {self.stop_price:.2f}")

                atr_stop = self.entry_price + self.atr_mult * atr
                if atr_stop < self.stop_price:
                    self.stop_price = atr_stop

                if high >= self.stop_price:
                    print(f"🌙🛑 SHORT STOP HIT at {self.stop_price:.2f}")
                    self.position.close()
                    self.trade_dir = 0
                elif self.bars_since_fractal >= self.time_stop:
                    print(f"🌙⏰ TIME STOP on short after {self.time_stop} bars")
                    self.position.close()
                    self.trade_dir = 0
            return

        # Entry logic
        if adx < self.adx_threshold:
            return

        # Long entry
        if (not np.isnan(self.last_up_fractal) and price > self.last_up_fractal
                and price > ema and ema_slope_up):
            stop = self.last_dn_fractal if not np.isnan(self.last_dn_fractal) else price - self.atr_mult * atr
            # Enforce min stop distance of 1x ATR
            if price - stop < atr:
                stop = price - atr
            risk_per_unit = price - stop
            if risk_per_unit <= 0:
                return
            risk_amount = self.equity * self.risk_pct
            size = int(round(risk_amount / risk_per_unit))
            if size < 1:
                size = 1
            # Cap by equity
            max_size = int(self.equity / price)
            size = min(size, max_size) if max_size > 0 else size
            if size < 1:
                return
            self.buy(size=size)
            self.entry_price = price
            self.stop_price = stop
            self.trade_dir = 1
            self.bars_since_fractal = 0
            print(f"🌙🚀 LONG ENTRY @ {price:.2f} | stop {stop:.2f} | size {size} | ADX {adx:.1f}")

        # Short entry
        elif (not np.isnan(self.last_dn_fractal) and price < self.last_dn_fractal
              and price < ema and ema_slope_dn):
            stop = self.last_up_fractal if not np.isnan(self.last_up_fractal) else price + self.atr_mult * atr
            if stop - price < atr:
                stop = price + atr
            risk_per_unit = stop - price
            if risk_per_unit <= 0:
                return
            risk_amount = self.equity * self.risk_pct
            size = int(round(risk_amount / risk_per_unit))
            if size < 1:
                size = 1
            max_size = int(self.equity / price)
            size = min(size, max_size) if max_size > 0 else size
            if size < 1:
                return
            self.sell(size=size)
            self.entry_price = price
            self.stop_price = stop
            self.trade_dir = -1
            self.bars_since_fractal = 0
            print(f"🌙🔻 SHORT ENTRY @ {price:.2f} | stop {stop:.2f} | size {size} | ADX {adx:.1f}")


bt = Backtest(data, FractalTrailblazer, cash=1_000_000, commission=0.0002, exclusive_orders=True)
stats = bt.run()
print(stats)
print(stats._strategy)