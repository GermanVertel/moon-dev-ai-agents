import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

print("🌙✨ Moon Dev's FractalChannel Backtest Initializing... ✨🌙")

data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open', 'high': 'High', 'low': 'Low',
    'close': 'Close', 'volume': 'Volume'
})
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')
print(f"🚀 Data loaded: {len(data)} bars from {data.index[0]} to {data.index[-1]} 🌙")


def fractal_up(high, n=2):
    """Bill Williams up fractal: n bars on each side with lower highs."""
    arr = high.values
    out = np.full(len(arr), np.nan)
    for i in range(n, len(arr) - n):
        center = arr[i]
        left = arr[i - n:i]
        right = arr[i + 1:i + n + 1]
        if center > left.max() and center > right.max():
            out[i] = center
    return out


def fractal_down(low, n=2):
    """Bill Williams down fractal: n bars on each side with higher lows."""
    arr = low.values
    out = np.full(len(arr), np.nan)
    for i in range(n, len(arr) - n):
        center = arr[i]
        left = arr[i - n:i]
        right = arr[i + 1:i + n + 1]
        if center < left.min() and center < right.min():
            out[i] = center
    return out


def forward_fill_fractal(fractal_vals, shift=2):
    """Shift by 2 (confirmation) then forward-fill so only confirmed fractals are visible."""
    s = pd.Series(fractal_vals)
    confirmed = s.shift(shift)
    return confirmed.ffill().values


class FractalChannel(Strategy):
    # Inputs
    atr_period = 14
    ema_period = 50
    ema_smooth_period = 3
    risk_pct = 0.01
    stop_atr_buffer = 0.25
    channel_break_atr = 0.5
    min_channel_atr = 0.5
    cooldown_bars = 3
    tp_channel_mult = 2.0

    def init(self):
        print("🌙 Initializing indicators... ✨")
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close,
                          timeperiod=self.atr_period, name='ATR')
        self.ema = self.I(talib.EMA, self.data.Close, timeperiod=self.ema_period, name='EMA50')

        # Fractals
        fu = fractal_up(self.data.High, 2)
        fd = fractal_down(self.data.Low, 2)
        self.up_fractal = self.I(lambda: forward_fill_fractal(fu, 2), name='UpFrac')
        self.dn_fractal = self.I(lambda: forward_fill_fractal(fd, 2), name='DnFrac')

        # EMA-smooth fractal levels
        self.up_fractal_s = self.I(talib.EMA, pd.Series(self.up_fractal), timeperiod=self.ema_smooth_period,
                                   name='UpFracS')
        self.dn_fractal_s = self.I(talib.EMA, pd.Series(self.dn_fractal), timeperiod=self.ema_smooth_period,
                                   name='DnFracS')

        self.ema_slope = self.I(lambda: np.append([np.nan], np.diff(self.ema) / self.ema[:-1]),
                                name='EMAslope')

        self.last_exit_bar = -10 ** 9
        self.trade_stop = None
        self.trade_target = None
        self.entry_price = None
        print("🚀 Indicators ready! 🌙")

    def next(self):
        i = len(self.data) - 1
        price = self.data.Close[-1]

        atr = self.atr[-1]
        ema = self.ema[-1]
        slope = self.ema_slope[-1]
        upf = self.up_fractal_s[-1]
        dnf = self.dn_fractal_s[-1]

        if np.isnan(atr) or np.isnan(ema) or np.isnan(upf) or np.isnan(dnf) or np.isnan(slope):
            return

        channel_width = upf - dnf
        if channel_width <= 0:
            return

        # Manage existing position
        if self.position:
            # Trailing stop update
            if self.position.is_long:
                new_stop = dnf - self.stop_atr_buffer * atr
                if self.trade_stop is None or new_stop > self.trade_stop:
                    self.trade_stop = new_stop
                # Channel break exit
                if price < upf - self.channel_break_atr * atr:
                    print(f"🌙 EXIT LONG (channel break) @ {price:.2f} ✨")
                    self.position.close()
                    self.last_exit_bar = i
                    self.trade_stop = None
                    self.trade_target = None
                    return
                # Stop loss
                if price <= self.trade_stop:
                    print(f"🌙 STOP LONG @ {price:.2f} 💥")
                    self.position.close()
                    self.last_exit_bar = i
                    self.trade_stop = None
                    self.trade_target = None
                    return
                # Take profit
                if self.trade_target and price >= self.trade_target:
                    print(f"🌙 TP LONG @ {price:.2f} 🎯")
                    self.position.close()
                    self.last_exit_bar = i
                    self.trade_stop = None
                    self.trade_target = None
                    return
            else:
                new_stop = upf + self.stop_atr_buffer * atr
                if self.trade_stop is None or new_stop < self.trade_stop:
                    self.trade_stop = new_stop
                if price > dnf + self.channel_break_atr * atr:
                    print(f"🌙 EXIT SHORT (channel break) @ {price:.2f} ✨")
                    self.position.close()
                    self.last_exit_bar = i
                    self.trade_stop = None
                    self.trade_target = None
                    return
                if price >= self.trade_stop:
                    print(f"🌙 STOP SHORT @ {price:.2f} 💥")
                    self.position.close()
                    self.last_exit_bar = i
                    self.trade_stop = None
                    self.trade_target = None
                    return
                if self.trade_target and price <= self.trade_target:
                    print(f"🌙 TP SHORT @ {price:.2f} 🎯")
                    self.position.close()
                    self.last_exit_bar = i
                    self.trade_stop = None
                    self.trade_target = None
                    return
            return

        # Cooldown
        if i - self.last_exit_bar < self.cooldown_bars:
            return

        # Consolidation filter
        if channel_width < self.min_channel_atr * atr:
            return

        # Channel width expanding check
        if len(self.up_fractal_s) > 5:
            prev_width = self.up_fractal_s[-5] - self.dn_fractal_s[-5]
            if not np.isnan(prev_width) and channel_width <= prev_width:
                return

        # Long entry
        if price > upf and slope > 0:
            stop = dnf - self.stop_atr_buffer * atr
            risk = price - stop
            if risk <= 0:
                return
            size = int(round((self.equity * self.risk_pct) / risk))
            if size <= 0:
                return
            self.trade_stop = stop
            self.trade_target = price + self.tp_channel_mult * channel_width
            print(f"🚀 LONG ENTRY @ {price:.2f} | stop={stop:.2f} | size={size} 🌙")
            self.buy(size=size)
            self.entry_price = price
            return

        # Short entry
        if price < dnf and slope < 0:
            stop = upf + self.stop_atr_buffer * atr
            risk = stop - price
            if risk <= 0:
                return
            size = int(round((self.equity * self.risk_pct) / risk))
            if size <= 0:
                return
            self.trade_stop = stop
            self.trade_target = price - self.tp_channel_mult * channel_width
            print(f"🔻 SHORT ENTRY @ {price:.2f} | stop={stop:.2f} | size={size} 🌙")
            self.sell(size=size)
            self.entry_price = price


bt = Backtest(data, FractalChannel, cash=1_000_000, commission=0.0002, exclusive_orders=True)
stats = bt.run()
print(stats)
print(stats._strategy)