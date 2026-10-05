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
    'datetime': 'Datetime',
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

# Set datetime index
if 'Datetime' in data.columns:
    data['Datetime'] = pd.to_datetime(data['Datetime'])
    data = data.set_index('Datetime')

data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

print("🌙 Moon Dev Backtest Engine Starting... ✨")
print(f"📊 Data loaded: {len(data)} bars")
print(f"🚀 First bar: {data.index[0]}")
print(f"🌙 Last bar: {data.index[-1]}")


class VolatileDivergence(Strategy):
    """
    VolatileDivergence Strategy 🌙
    Combines volatility breakout detection with RSI divergence confirmation.
    """
    # Strategy parameters
    atr_period = 14
    rsi_period = 14
    bb_period = 20
    bb_std = 2.0
    vol_ma_period = 20
    ema_period = 50
    pivot_window = 5
    divergence_lookback = 15
    atr_multiplier = 1.0
    tp_atr_mult = 2.0
    sl_atr_mult = 1.0
    trail_atr_mult = 1.5
    risk_pct = 0.02
    volume_mult = 1.5
    time_exit_bars = 10

    def init(self):
        print("🌙 Initializing Moon Dev Indicators... ✨")
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        # ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        # RSI
        self.rsi = self.I(talib.RSI, close, timeperiod=self.rsi_period)

        # Bollinger Bands
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS, close, timeperiod=self.bb_period,
            nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0
        )

        # Volume MA
        self.vol_ma = self.I(talib.SMA, volume, timeperiod=self.vol_ma_period)

        # EMA trend filter
        self.ema = self.I(talib.EMA, close, timeperiod=self.ema_period)

        # Pivot highs/lows for divergence
        self.pivot_high = self.I(talib.MAX, high, timeperiod=self.pivot_window)
        self.pivot_low = self.I(talib.MIN, low, timeperiod=self.pivot_window)

        # Track state
        self.entry_price_val = None
        self.entry_bar = None
        self.stop_price = None
        self.tp_price = None
        self.trail_active = False
        self.trail_stop = None

        print("🌙 Indicators ready! Let's go 🚀")

    def _detect_bullish_divergence(self):
        """Price makes lower lows while RSI makes higher lows."""
        try:
            lookback = self.divergence_lookback
            if len(self.data) < lookback + 5:
                return False

            lows = self.data.Low[-lookback:]
            rsi_vals = self.rsi[-lookback:]

            # Find two most recent local lows in price
            price_lows = []
            for i in range(2, len(lows) - 2):
                if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                    price_lows.append((i, lows[i], rsi_vals[i]))

            if len(price_lows) < 2:
                return False

            # Compare last two lows: price lower low, RSI higher low
            p1, p2 = price_lows[-2], price_lows[-1]
            if p2[1] < p1[1] and p2[2] > p1[2]:
                return True
            return False
        except Exception:
            return False

    def _detect_bearish_divergence(self):
        """Price makes higher highs while RSI makes lower highs."""
        try:
            lookback = self.divergence_lookback
            if len(self.data) < lookback + 5:
                return False

            highs = self.data.High[-lookback:]
            rsi_vals = self.rsi[-lookback:]

            price_highs = []
            for i in range(2, len(highs) - 2):
                if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                    price_highs.append((i, highs[i], rsi_vals[i]))

            if len(price_highs) < 2:
                return False

            p1, p2 = price_highs[-2], price_highs[-1]
            if p2[1] > p1[1] and p2[2] < p1[2]:
                return True
            return False
        except Exception:
            return False

    def next(self):
        if len(self.data) < max(self.bb_period, self.ema_period, self.atr_period) + 5:
            return

        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        atr = self.atr[-1]
        rsi = self.rsi[-1]
        vol = self.data.Volume[-1]
        vol_ma = self.vol_ma[-1]
        ema = self.ema[-1]

        if np.isnan(atr) or np.isnan(rsi) or np.isnan(vol_ma) or np.isnan(ema):
            return

        # Skip dead or panic volatility
        if atr <= 0:
            return

        # Manage existing position
        if self.position:
            bars_held = len(self.data) - self.entry_bar

            # Time exit
            if bars_held >= self.time_exit_bars:
                print(f"⏰ Moon Dev Time Exit at {price:.2f} after {bars_held} bars 🌙")
                self.position.close()
                self._reset_state()
                return

            # Trailing stop
            if self.position.is_long:
                if not self.trail_active and price >= self.entry_price_val + atr * self.atr_multiplier:
                    self.trail_active = True
                    self.trail_stop = price - atr * self.trail_atr_mult
                    print(f"🌙 Long trail activated at {price:.2f} 🚀")

                if self.trail_active:
                    new_trail = price - atr * self.trail_atr_mult
                    if new_trail > self.trail_stop:
                        self.trail_stop = new_trail

                    if price <= self.trail_stop:
                        print(f"🛑 Long trail stop hit at {price:.2f} ✨")
                        self.position.close()
                        self._reset_state()
                        return

                if price <= self.stop_price:
                    print(f"🛑 Long SL hit at {price:.2f} 💔")
                    self.position.close()
                    self._reset_state()
                    return

                if price >= self.tp_price:
                    print(f"🎯 Long TP hit at {price:.2f} 🌙✨")
                    self.position.close()
                    self._reset_state()
                    return

                # Divergence invalidation
                if self._detect_bearish_divergence():
                    print(f"⚠️ Bearish divergence invalidation - closing long 🌙")
                    self.position.close()
                    self._reset_state()
                    return

            elif self.position.is_short:
                if not self.trail_active and price <= self.entry_price_val - atr * self.atr_multiplier:
                    self.trail_active = True
                    self.trail_stop = price + atr * self.trail_atr_mult
                    print(f"🌙 Short trail activated at {price:.2f} 🚀")

                if self.trail_active:
                    new_trail = price + atr * self.trail_atr_mult
                    if new_trail < self.trail_stop:
                        self.trail_stop = new_trail

                    if price >= self.trail_stop:
                        print(f"🛑 Short trail stop hit at {price:.2f} ✨")
                        self.position.close()
                        self._reset_state()
                        return

                if price >= self.stop_price:
                    print(f"🛑 Short SL hit at {price:.2f} 💔")
                    self.position.close()
                    self._reset_state()
                    return

                if price <= self.tp_price:
                    print(f"🎯 Short TP hit at {price:.2f} 🌙✨")
                    self.position.close()
                    self._reset_state()
                    return

                if self._detect_bullish_divergence():
                    print(f"⚠️ Bullish divergence invalidation - closing short 🌙")
                    self.position.close()
                    self._reset_state()
                    return

            return

        # Entry logic
        vol_confirm = vol > vol_ma * self.volume_mult

        # LONG: break above upper BB + bullish divergence + trend filter
        if high > self.bb_upper[-1] and price > self.bb_upper[-1]:
            if self._detect_bullish_divergence() and rsi > 50 and price > ema:
                if vol_confirm:
                    stop = price - atr * self.sl_atr_mult
                    tp = price + atr * self.tp_atr_mult
                    risk = price - stop
                    if risk > 0:
                        size = max(1, int(round((self.equity * self.risk_pct) / risk)))
                        if size > 0:
                            print(f"🚀🌙 LONG BREAKOUT! Price={price:.2f} RSI={rsi:.1f} ATR={atr:.2f} Size={size} ✨")
                            self.buy(size=size)
                            self.entry_price_val = price
                            self.entry_bar = len(self.data)
                            self.stop_price = stop
                            self.tp_price = tp
                            self.trail_active = False
                            self.trail_stop = None

        # SHORT: break below lower BB + bearish divergence + trend filter
        elif low < self.bb_lower[-1] and price < self.bb_lower[-1]:
            if self._detect_bearish_divergence() and rsi < 50 and price < ema:
                if vol_confirm:
                    stop = price + atr * self.sl_atr_mult
                    tp = price - atr * self.tp_atr_mult
                    risk = stop - price
                    if risk > 0:
                        size = max(1, int(round((self.equity * self.risk_pct) / risk)))
                        if size > 0:
                            print(f"🔻🌙 SHORT BREAKDOWN! Price={price:.2f} RSI={rsi:.1f} ATR={atr:.2f} Size={size} ✨")
                            self.sell(size=size)
                            self.entry_price_val = price
                            self.entry_bar = len(self.data)
                            self.stop_price = stop
                            self.tp_price = tp
                            self.trail_active = False
                            self.trail_stop = None

    def _reset_state(self):
        self.entry_price_val = None
        self.entry_bar = None
        self.stop_price = None
        self.tp_price = None
        self.trail_active = False
        self.trail_stop = None


print("🌙 Setting up Moon Dev Backtest... ✨")
bt = Backtest(
    data,
    VolatileDivergence,
    cash=1_000_000,
    commission=0.001,
    exclusive=False
)

print("🚀 Running backtest... 🌙")
stats = bt.run()
print(stats)
print(stats._strategy)