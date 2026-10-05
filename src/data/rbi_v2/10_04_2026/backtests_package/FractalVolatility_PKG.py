import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Load and clean data
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

print("🌙 Moon Dev FractalVolatility Backtest Initializing... ✨")
print(f"📊 Data loaded: {len(data)} bars from {data.index[0]} to {data.index[-1]}")


class FractalVolatility(Strategy):
    bb_period = 20
    bb_std = 2.0
    atr_period = 14
    atr_ma_period = 20
    bw_ma_period = 20
    risk_pct = 0.01
    sl_atr_mult = 1.5
    tp_atr_mult = 2.5
    max_positions = 2

    def init(self):
        close = pd.Series(self.data.Close, index=self.data.index)
        high = pd.Series(self.data.High, index=self.data.index)
        low = pd.Series(self.data.Low, index=self.data.index)
        volume = pd.Series(self.data.Volume, index=self.data.index)

        # Bollinger Bands - using talib.BBANDS with column selection
        def bb_upper(close):
            u, m, l = talib.BBANDS(np.asarray(close, dtype=float),
                                   timeperiod=self.bb_period,
                                   nbdevup=self.bb_std,
                                   nbdevdn=self.bb_std,
                                   matype=0)
            return u

        def bb_middle(close):
            u, m, l = talib.BBANDS(np.asarray(close, dtype=float),
                                   timeperiod=self.bb_period,
                                   nbdevup=self.bb_std,
                                   nbdevdn=self.bb_std,
                                   matype=0)
            return m

        def bb_lower(close):
            u, m, l = talib.BBANDS(np.asarray(close, dtype=float),
                                   timeperiod=self.bb_period,
                                   nbdevup=self.bb_std,
                                   nbdevdn=self.bb_std,
                                   matype=0)
            return l

        self.bb_upper = self.I(bb_upper, close)
        self.bb_middle = self.I(bb_middle, close)
        self.bb_lower = self.I(bb_lower, close)

        # %B and Bandwidth
        def calc_pctb(c, u, l):
            u = np.asarray(u, dtype=float)
            l = np.asarray(l, dtype=float)
            c = np.asarray(c, dtype=float)
            denom = u - l
            with np.errstate(divide='ignore', invalid='ignore'):
                pctb = (c - l) / denom
            return pctb

        def calc_bw(u, l, m):
            u = np.asarray(u, dtype=float)
            l = np.asarray(l, dtype=float)
            m = np.asarray(m, dtype=float)
            with np.errstate(divide='ignore', invalid='ignore'):
                bw = (u - l) / m
            return bw

        self.pctb = self.I(calc_pctb, close, self.bb_upper, self.bb_lower)
        self.bandwidth = self.I(calc_bw, self.bb_upper, self.bb_lower, self.bb_middle)
        self.bw_ma = self.I(talib.SMA, self.bandwidth, timeperiod=self.bw_ma_period)

        # ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)
        self.atr_ma = self.I(talib.SMA, self.atr, timeperiod=self.atr_ma_period)

        # Volume MA
        self.vol_ma = self.I(talib.SMA, volume, timeperiod=20)

        # Fractals - Williams 5-bar pattern (2 left, center, 2 right)
        h = np.asarray(high, dtype=float)
        l = np.asarray(low, dtype=float)
        n = len(h)

        frac_high = np.full(n, np.nan)
        frac_low = np.full(n, np.nan)

        for i in range(2, n - 2):
            # Fractal high
            if h[i] > h[i-1] and h[i] > h[i-2] and h[i] > h[i+1] and h[i] > h[i+2]:
                frac_high[i] = h[i]
            # Fractal low
            if l[i] < l[i-1] and l[i] < l[i-2] and l[i] < l[i+1] and l[i] < l[i+2]:
                frac_low[i] = l[i]

        self.frac_high = self.I(lambda: frac_high)
        self.frac_low = self.I(lambda: frac_low)

        self.entry_price = None
        self.stop_price = None
        self.tp_price = None
        self.trade_dir = 0

        print("🌙 Indicators initialized: BB, %B, BW, ATR, Fractals ✨")

    def next(self):
        i = len(self.data) - 1
        if i < 30:
            return

        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]

        # Manage existing position
        if self.position:
            self._manage_position(price, high, low)
            return

        # Check max positions
        if len(self.trades) >= self.max_positions:
            return

        # Check ATR chaos filter (skip if ATR 50%+ above MA)
        if not np.isnan(self.atr_ma[-1]) and self.atr_ma[-1] > 0:
            atr_ratio = self.atr[-1] / self.atr_ma[-1]
            if atr_ratio >= 1.5:
                return

        # Bandwidth filter: must be above 20-period average
        if np.isnan(self.bw_ma[-1]) or self.bandwidth[-1] < self.bw_ma[-1]:
            return

        # Volume filter (optional - require above 20-period avg)
        if not np.isnan(self.vol_ma[-1]) and self.data.Volume[-1] < self.vol_ma[-1]:
            return

        # Get last 3 confirmed fractal highs/lows
        frac_highs = self._get_last_fractals(self.frac_high, i, 3)
        frac_lows = self._get_last_fractals(self.frac_low, i, 3)

        # Bullish divergence: compare last two fractal lows
        if len(frac_lows) >= 2:
            f1_idx, f1_val = frac_lows[-2]
            f2_idx, f2_val = frac_lows[-1]

            # Confirm fractal #2 fully formed (2 bars after)
            if i - f2_idx >= 2:
                # Lower low
                if f2_val < f1_val:
                    # Touches/pierces lower BB
                    if self.data.Low[f2_idx] <= self.bb_lower[f2_idx]:
                        # %B higher at f2 than f1 (positive divergence)
                        p1 = self.pctb[f1_idx]
                        p2 = self.pctb[f2_idx]
                        if not np.isnan(p1) and not np.isnan(p2) and p2 > p1:
                            self._enter_long(price)

        # Bearish divergence: compare last two fractal highs
        if len(frac_highs) >= 2:
            f1_idx, f1_val = frac_highs[-2]
            f2_idx, f2_val = frac_highs[-1]

            if i - f2_idx >= 2:
                # Higher high
                if f2_val > f1_val:
                    # Touches/pierces upper BB
                    if self.data.High[f2_idx] >= self.bb_upper[f2_idx]:
                        # %B lower at f2 than f1 (negative divergence)
                        p1 = self.pctb[f1_idx]
                        p2 = self.pctb[f2_idx]
                        if not np.isnan(p1) and not np.isnan(p2) and p2 < p1:
                            self._enter_short(price)

    def _get_last_fractals(self, fractal_arr, current_idx, count):
        result = []
        for j in range(current_idx, max(0, current_idx - 200), -1):
            v = fractal_arr[j]
            if not np.isnan(v):
                result.append((j, v))
                if len(result) >= count:
                    break
        result.reverse()
        return result

    def _enter_long(self, price):
        atr = self.atr[-1]
        if np.isnan(atr) or atr <= 0:
            return
        sl = price - self.sl_atr_mult * atr
        tp = price + self.tp_atr_mult * atr
        risk_amount = self.equity * self.risk_pct
        risk_per_unit = self.sl_atr_mult * atr
        size = risk_amount / risk_per_unit
        size = int(round(size))
        if size < 1:
            size = 1
        print(f"🚀 MOON DEV LONG SIGNAL! Price: {price:.2f} | SL: {sl:.2f} | TP: {tp:.2f} | Size: {size} 🌙")
        self.buy(size=size)
        self.entry_price = price
        self.stop_price = sl
        self.tp_price = tp
        self.trade_dir = 1

    def _enter_short(self, price):
        atr = self.atr[-1]
        if np.isnan(atr) or atr <= 0:
            return
        sl = price + self.sl_atr_mult * atr
        tp = price - self.tp_atr_mult * atr
        risk_amount = self.equity * self.risk_pct
        risk_per_unit = self.sl_atr_mult * atr
        size = risk_amount / risk_per_unit
        size = int(round(size))
        if size < 1:
            size = 1
        print(f"🔻 MOON DEV SHORT SIGNAL! Price: {price:.2f} | SL: {sl:.2f} | TP: {tp:.2f} | Size: {size} 🌙")
        self.sell(size=size)
        self.entry_price = price
        self.stop_price = sl
        self.tp_price = tp
        self.trade_dir = -1

    def _manage_position(self, price, high, low):
        atr = self.atr[-1]
        atr_ma = self.atr_ma[-1]

        # Volatility trailing stop
        trail_mult = self.sl_atr_mult
        if not np.isnan(atr) and not np.isnan(atr_ma) and atr_ma > 0:
            ratio = atr / atr_ma
            if ratio >= 1.4:
                trail_mult = 0.5
            elif ratio >= 1.2:
                trail_mult = 1.0

        if self.trade_dir == 1:
            # Trailing stop
            new_sl = price - trail_mult * atr if not np.isnan(atr) else self.stop_price
            if new_sl > self.stop_price:
                self.stop_price = new_sl

            # Stop loss
            if low <= self.stop_price:
                print(f"🛑 LONG STOP HIT @ {self.stop_price:.2f} 🌙")
                self.position.close()
                return
            # Take profit
            if high >= self.tp_price:
                print(f"🎯 LONG TP HIT @ {self.tp_price:.2f} ✨")
                self.position.close()
                return
            # Band reversion exit: close back inside middle BB against direction
            if price < self.bb_middle[-1]:
                print(f"📉 LONG BAND REVERSION EXIT @ {price:.2f} 🌙")
                self.position.close()
                return

        elif self.trade_dir == -1:
            new_sl = price + trail_mult * atr if not np.isnan(atr) else self.stop_price
            if new_sl < self.stop_price:
                self.stop_price = new_sl

            if high >= self.stop_price:
                print(f"🛑 SHORT STOP HIT @ {self.stop_price:.2f} 🌙")
                self.position.close()
                return
            if low <= self.tp_price:
                print(f"🎯 SHORT TP HIT @ {self.tp_price:.2f} ✨")
                self.position.close()
                return
            if price > self.bb_middle[-1]:
                print(f"📈 SHORT BAND REVERSION EXIT @ {price:.2f} 🌙")
                self.position.close()
                return


print("🌙✨ Starting Moon Dev FractalVolatility Backtest... 🚀")
bt = Backtest(data, FractalVolatility, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙 Moon Dev Backtest Complete! ✨🚀")