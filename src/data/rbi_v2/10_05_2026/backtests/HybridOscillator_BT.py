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

print("🌙 Moon Dev HybridOscillator Backtest Starting... ✨")
print(f"📊 Data loaded: {len(data)} bars")
print(f"🚀 Price range: {data['Close'].min():.2f} - {data['Close'].max():.2f}")


class HybridOscillator(Strategy):
    # Optimization parameters
    rsi_period = 14
    bb_period = 20
    bb_std = 2.0
    atr_period = 14
    atr_tp_mult = 1.5
    atr_sl_mult = 0.75
    atr_trail_mult = 1.0
    vol_ma_period = 20
    vol_threshold = 1.5
    rsi_oversold = 30
    rsi_overbought = 70
    risk_per_trade = 0.02
    max_hold_bars = 96  # 24 hours in 15min bars

    def init(self):
        print("🌙 Initializing indicators... ✨")
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        # RSI
        self.rsi = self.I(talib.RSI, close, timeperiod=self.rsi_period)

        # Bollinger Bands
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS, close, timeperiod=self.bb_period,
            nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0
        )

        # MACD
        self.macd, self.macd_signal, self.macd_hist = self.I(
            talib.MACD, close, fastperiod=12, slowperiod=26, signalperiod=9
        )

        # ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        # Volume MA
        self.vol_ma = self.I(talib.SMA, volume, timeperiod=self.vol_ma_period)

        # EMA ribbon
        self.ema8 = self.I(talib.EMA, close, timeperiod=8)
        self.ema21 = self.I(talib.EMA, close, timeperiod=21)
        self.ema55 = self.I(talib.EMA, close, timeperiod=55)

        # VWAP (session-based approximation using rolling)
        typical_price = (high + low + close) / 3
        tp_vol = typical_price * volume
        self.vwap = self.I(
            lambda tp_v, v, n: pd.Series(tp_v).rolling(n).sum().values / pd.Series(v).rolling(n).sum().values,
            tp_vol, volume, 96
        )

        print("✅ All indicators initialized! 🚀")

        # Track entry info
        self.entry_bar = None
        self.entry_price = None
        self.tp_price = None
        self.sl_price = None
        self.trail_activated = False

    def next(self):
        price = self.data.Close[-1]

        # Skip if not enough data
        if len(self.data) < 60:
            return

        # Check for active position
        if self.position:
            self._manage_position()
            return

        # Entry logic
        self._check_entries()

    def _check_entries(self):
        rsi = self.rsi[-1]
        bb_upper = self.bb_upper[-1]
        bb_lower = self.bb_lower[-1]
        macd_hist = self.macd_hist[-1]
        macd_hist_prev = self.macd_hist[-2] if len(self.macd_hist) > 1 else 0
        atr = self.atr[-1]
        vol = self.data.Volume[-1]
        vol_ma = self.vol_ma[-1]
        close = self.data.Close[-1]
        vwap = self.vwap[-1]
        ema8 = self.ema8[-1]
        ema21 = self.ema21[-1]
        ema55 = self.ema55[-1]

        if np.isnan([rsi, bb_upper, bb_lower, atr, vol_ma, vwap, ema8, ema21, ema55]).any():
            return

        volume_spike = vol > (self.vol_threshold * vol_ma)

        # Long conditions
        long_rsi = rsi < self.rsi_oversold
        long_bb = close <= bb_lower * 1.005  # touch lower band
        long_vwap = close > vwap  # price above VWAP (trend filter)
        long_macd_div = macd_hist > macd_hist_prev  # bullish momentum shift
        long_ema_trend = ema8 > ema55  # higher timeframe trend bias

        # Short conditions
        short_rsi = rsi > self.rsi_overbought
        short_bb = close >= bb_upper * 0.995  # touch upper band
        short_vwap = close < vwap
        short_macd_div = macd_hist < macd_hist_prev
        short_ema_trend = ema8 < ema55

        if long_rsi and long_bb and long_macd_div and volume_spike and long_ema_trend:
            self._enter_long(close, atr)
        elif short_rsi and short_bb and short_macd_div and volume_spike and short_ema_trend:
            self._enter_short(close, atr)

    def _enter_long(self, price, atr):
        # Position sizing: 1,000,000 units as specified
        size = 1000000
        self.buy(size=size)
        self.entry_price = price
        self.tp_price = price + (self.atr_tp_mult * atr)
        self.sl_price = price - (self.atr_sl_mult * atr)
        self.trail_activated = False
        self.entry_bar = len(self.data)
        print(f"🌙✨ LONG ENTRY @ {price:.2f} | TP: {self.tp_price:.2f} | SL: {self.sl_price:.2f} | Size: {size} 🚀")

    def _enter_short(self, price, atr):
        size = 1000000
        self.sell(size=size)
        self.entry_price = price
        self.tp_price = price - (self.atr_tp_mult * atr)
        self.sl_price = price + (self.atr_sl_mult * atr)
        self.trail_activated = False
        self.entry_bar = len(self.data)
        print(f"🌙✨ SHORT ENTRY @ {price:.2f} | TP: {self.tp_price:.2f} | SL: {self.sl_price:.2f} | Size: {size} 🚀")

    def _manage_position(self):
        price = self.data.Close[-1]
        atr = self.atr[-1]
        rsi = self.rsi[-1]

        if np.isnan(atr) or np.isnan(rsi):
            return

        # Time-based exit
        bars_held = len(self.data) - self.entry_bar
        if bars_held >= self.max_hold_bars:
            print(f"⏰ Time exit after {bars_held} bars @ {price:.2f}")
            self.position.close()
            return

        if self.position.is_long:
            # Take profit
            if price >= self.tp_price:
                print(f"🎯 TP hit LONG @ {price:.2f} 🌙")
                self.position.close()
                return

            # Stop loss
            if price <= self.sl_price:
                print(f"🛑 SL hit LONG @ {price:.2f}")
                self.position.close()
                return

            # Trailing stop activation
            if not self.trail_activated and price >= self.entry_price + (self.atr_trail_mult * atr):
                self.trail_activated = True
                self.sl_price = max(self.sl_price, price - (self.atr_sl_mult * atr))
                print(f"🔒 Trailing stop activated LONG @ {price:.2f}, new SL: {self.sl_price:.2f}")

            # Update trailing stop
            if self.trail_activated:
                new_sl = price - (self.atr_sl_mult * atr)
                if new_sl > self.sl_price:
                    self.sl_price = new_sl

            # RSI cross against position
            if rsi > 50 and self.rsi[-2] <= 50:
                print(f"📉 RSI crossed 50 against LONG @ {price:.2f} 🚪")
                self.position.close()
                return

        elif self.position.is_short:
            if price <= self.tp_price:
                print(f"🎯 TP hit SHORT @ {price:.2f} 🌙")
                self.position.close()
                return

            if price >= self.sl_price:
                print(f"🛑 SL hit SHORT @ {price:.2f}")
                self.position.close()
                return

            if not self.trail_activated and price <= self.entry_price - (self.atr_trail_mult * atr):
                self.trail_activated = True
                self.sl_price = min(self.sl_price, price + (self.atr_sl_mult * atr))
                print(f"🔒 Trailing stop activated SHORT @ {price:.2f}, new SL: {self.sl_price:.2f}")

            if self.trail_activated:
                new_sl = price + (self.atr_sl_mult * atr)
                if new_sl < self.sl_price:
                    self.sl_price = new_sl

            if rsi < 50 and self.rsi[-2] >= 50:
                print(f"📈 RSI crossed 50 against SHORT @ {price:.2f} 🚪")
                self.position.close()
                return


# Run backtest
bt = Backtest(
    data,
    HybridOscillator,
    cash=1000000,
    commission=0.001,
    exclusive=False,
    trade_on_close=False,
    hedging=False
)

print("🌙 Starting backtest run... 🚀")
stats = bt.run()
print(stats)
print(stats._strategy)
print("✨ Moon Dev backtest complete! 🌙")