import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

print("🌙✨ MOON DEV BACKTEST INITIALIZING: VolatilityPulse ✨🌙")

# Load data
data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Rename to proper case
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

# Set datetime index
if 'datetime' in data.columns:
    data['datetime'] = pd.to_datetime(data['datetime'])
    data.set_index('datetime', inplace=True)

data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

print(f"🚀 Data loaded: {len(data)} bars")
print(f"🌙 Date range: {data.index[0]} to {data.index[-1]}")


def bbw_pct_rank(x, window=50):
    s = pd.Series(x)
    return s.rolling(window).apply(lambda w: (w.iloc[-1] >= w).sum() / len(w) * 100, raw=False).values


def rolling_median(x, window=50):
    return pd.Series(x).rolling(window).median().values


def vwap_calc(h, l, c, v):
    tp = (h + l + c) / 3
    return (tp * v).cumsum() / v.cumsum()


def bb_upper_calc(close, period, std):
    return talib.BBANDS(close, timeperiod=period, nbdevup=std, nbdevdn=std, matype=0)[0]


def bb_mid_calc(close, period, std):
    return talib.BBANDS(close, timeperiod=period, nbdevup=std, nbdevdn=std, matype=0)[1]


def bb_lower_calc(close, period, std):
    return talib.BBANDS(close, timeperiod=period, nbdevup=std, nbdevdn=std, matype=0)[2]


def bbw_calc(upper, lower, mid):
    return (upper - lower) / mid


def macd_hist_calc(close):
    macd, macdsig, macdhist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    return macdhist


class VolatilityPulse(Strategy):
    # Parameters
    atr_period = 14
    bb_period = 20
    bb_std = 2
    vol_ma_period = 20
    swing_window = 5
    risk_pct = 0.005  # 0.5% risk
    atr_body_mult = 1.2
    vol_spike_mult = 1.5
    bbw_pct_threshold = 30
    trail_mult = 0.75
    target_mult = 1.75
    time_stop_bars = 8

    def init(self):
        print("🌙 Initializing VolatilityPulse indicators...")

        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        volume = pd.Series(self.data.Volume)

        # ATR execution TF
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        # Bollinger Bands
        self.bb_upper = self.I(bb_upper_calc, close, self.bb_period, self.bb_std)
        self.bb_mid = self.I(bb_mid_calc, close, self.bb_period, self.bb_std)
        self.bb_lower = self.I(bb_lower_calc, close, self.bb_period, self.bb_std)

        # BBW
        self.bbw = self.I(bbw_calc, self.bb_upper, self.bb_lower, self.bb_mid)

        # BBW percentile rank (rolling)
        self.bbw_pct = self.I(bbw_pct_rank, self.bbw, 50)

        # Volume MA
        self.vol_ma = self.I(talib.SMA, volume, timeperiod=self.vol_ma_period)

        # Swing highs/lows
        self.swing_high = self.I(talib.MAX, high, timeperiod=self.swing_window)
        self.swing_low = self.I(talib.MIN, low, timeperiod=self.swing_window)

        # RSI
        self.rsi = self.I(talib.RSI, close, timeperiod=14)

        # MACD histogram
        self.macd_hist = self.I(macd_hist_calc, close)

        # ATR median for regime check
        self.atr_median = self.I(rolling_median, self.atr, 50)

        # VWAP
        self.vwap = self.I(vwap_calc, high, low, close, volume)

        # Trade tracking
        self.entry_price = None
        self.entry_bar = None
        self.stop_price = None
        self.target_price = None
        self.trail_active = False

        print("✨ Indicators initialized! 🌙")

    def next(self):
        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        open_ = self.data.Open[-1]
        volume = self.data.Volume[-1]

        atr = self.atr[-1]
        if np.isnan(atr) or atr <= 0:
            return

        # Position management
        if self.position:
            bars_held = len(self.data) - self.entry_bar

            if self.position.is_long:
                # Trailing stop
                if high >= self.entry_price + atr:
                    self.trail_active = True
                if self.trail_active:
                    new_stop = high - self.trail_mult * atr
                    if new_stop > self.stop_price:
                        self.stop_price = new_stop
                        print(f"🌙 Trailing stop updated to {new_stop:.2f} 🚀")

                # Exit conditions
                if low <= self.stop_price:
                    print(f"🛑 LONG STOP HIT @ {self.stop_price:.2f} (price: {low:.2f})")
                    self.position.close()
                    self._reset_trade()
                elif high >= self.target_price:
                    print(f"🎯 LONG TARGET HIT @ {self.target_price:.2f} (price: {high:.2f}) ✨")
                    self.position.close()
                    self._reset_trade()
                elif bars_held >= self.time_stop_bars and high < self.entry_price + 0.5 * atr:
                    print(f"⏰ LONG TIME STOP after {bars_held} bars")
                    self.position.close()
                    self._reset_trade()
                elif not np.isnan(self.bbw_pct[-1]) and self.bbw_pct[-1] < self.bbw_pct_threshold:
                    print(f"💨 Volatility regime failure (BBW pct: {self.bbw_pct[-1]:.1f}) - exiting long")
                    self.position.close()
                    self._reset_trade()

            elif self.position.is_short:
                if low <= self.entry_price - atr:
                    self.trail_active = True
                if self.trail_active:
                    new_stop = low + self.trail_mult * atr
                    if new_stop < self.stop_price:
                        self.stop_price = new_stop
                        print(f"🌙 Trailing stop updated to {new_stop:.2f} 🚀")

                if high >= self.stop_price:
                    print(f"🛑 SHORT STOP HIT @ {self.stop_price:.2f} (price: {high:.2f})")
                    self.position.close()
                    self._reset_trade()
                elif low <= self.target_price:
                    print(f"🎯 SHORT TARGET HIT @ {self.target_price:.2f} (price: {low:.2f}) ✨")
                    self.position.close()
                    self._reset_trade()
                elif bars_held >= self.time_stop_bars and low > self.entry_price - 0.5 * atr:
                    print(f"⏰ SHORT TIME STOP after {bars_held} bars")
                    self.position.close()
                    self._reset_trade()
                elif not np.isnan(self.bbw_pct[-1]) and self.bbw_pct[-1] < self.bbw_pct_threshold:
                    print(f"💨 Volatility regime failure (BBW pct: {self.bbw_pct[-1]:.1f}) - exiting short")
                    self.position.close()
                    self._reset_trade()
            return

        # Entry logic
        if len(self.data) < 60:
            return

        # Regime check: ATR > median OR BBW rising
        atr_med = self.atr_median[-1]
        bbw_rising = (self.bbw[-1] > self.bbw[-2] > self.bbw[-3]) if not (
            np.isnan(self.bbw[-1]) or np.isnan(self.bbw[-2]) or np.isnan(self.bbw[-3])) else False

        regime_ok = False
        if not np.isnan(atr_med) and atr > atr_med:
            regime_ok = True
        if bbw_rising:
            regime_ok = True

        if not regime_ok:
            return

        # Volatility compression check before breakout
        if np.isnan(self.bbw_pct[-1]) or self.bbw_pct[-1] > 60:
            return

        # Body size
        body = abs(price - open_)
        body_ok = body > self.atr_body_mult * atr

        # Volume spike
        vol_ma = self.vol_ma[-1]
        vol_ok = (not np.isnan(vol_ma)) and (volume > self.vol_spike_mult * vol_ma)

        if not (body_ok and vol_ok):
            return

        # S/R zone from swing levels
        res_zone = self.swing_high[-2] if not np.isnan(self.swing_high[-2]) else high
        sup_zone = self.swing_low[-2] if not np.isnan(self.swing_low[-2]) else low

        # Long entry
        long_trigger = price > res_zone + 0.25 * atr
        long_confluence = (self.rsi[-1] > 55 and self.rsi[-1] > self.rsi[-2]) or (
            self.macd_hist[-1] > 0 and self.macd_hist[-2] <= 0) or price > self.vwap[-1]

        # Short entry
        short_trigger = price < sup_zone - 0.25 * atr
        short_confluence = (self.rsi[-1] < 45 and self.rsi[-1] < self.rsi[-2]) or (
            self.macd_hist[-1] < 0 and self.macd_hist[-2] >= 0) or price < self.vwap[-1]

        # Risk sizing
        equity = self.equity
        risk_amount = equity * self.risk_pct
        stop_dist = 1.0 * atr
        if stop_dist <= 0:
            return
        position_size = int(round(risk_amount / stop_dist))
        if position_size < 1:
            position_size = 1

        if long_trigger and long_confluence:
            self.entry_price = price
            self.entry_bar = len(self.data)
            self.stop_price = price - 1.0 * atr
            self.target_price = price + self.target_mult * atr
            self.trail_active = False
            print(f"🚀🌙 LONG ENTRY @ {price:.2f} | Stop: {self.stop_price:.2f} | Target: {self.target_price:.2f} | Size: {position_size}")
            self.buy(size=position_size)

        elif short_trigger and short_confluence:
            self.entry_price = price
            self.entry_bar = len(self.data)
            self.stop_price = price + 1.0 * atr
            self.target_price = price - self.target_mult * atr
            self.trail_active = False
            print(f"🔻🌙 SHORT ENTRY @ {price:.2f} | Stop: {self.stop_price:.2f} | Target: {self.target_price:.2f} | Size: {position_size}")
            self.sell(size=position_size)

    def _reset_trade(self):
        self.entry_price = None
        self.entry_bar = None
        self.stop_price = None
        self.target_price = None
        self.trail_active = False


print("🌙 Running VolatilityPulse backtest... 🚀✨")
bt = Backtest(data, VolatilityPulse, cash=1_000_000, commission=0.0005)
stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙✨ Moon Dev backtest complete! ✨🌙")