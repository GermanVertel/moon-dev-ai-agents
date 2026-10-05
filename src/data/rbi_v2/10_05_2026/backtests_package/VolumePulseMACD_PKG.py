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
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')

print("🌙 Moon Dev VolumePulseMACD Backtest Initializing... ✨")
print(f"🚀 Data loaded: {len(data)} bars")
print(f"📊 Columns: {list(data.columns)}")


class VolumePulseMACD(Strategy):
    # Strategy parameters
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    vol_ma_period = 20
    vol_ratio_threshold = 0.5
    ema_trend_period = 200
    atr_period = 14
    atr_stop_mult = 1.5
    risk_pct = 0.02
    swing_window = 20
    max_bars_in_trade = 20

    def init(self):
        print("🌙 Initializing indicators... ✨")
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        # MACD
        self.macd, self.macd_signal, self.macd_hist = self.I(
            talib.MACD, close,
            fastperiod=self.macd_fast,
            slowperiod=self.macd_slow,
            signalperiod=self.macd_signal
        )

        # Volume MA
        self.vol_ma = self.I(talib.SMA, volume, timeperiod=self.vol_ma_period)

        # Trend EMA
        self.ema_trend = self.I(talib.EMA, close, timeperiod=self.ema_trend_period)

        # ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        # Swing highs/lows for divergence detection
        self.swing_high = self.I(talib.MAX, high, timeperiod=self.swing_window)
        self.swing_low = self.I(talib.MIN, low, timeperiod=self.swing_window)

        # Track trade state
        self.entry_bar = None
        self.entry_price = None
        self.stop_price = None

    def next(self):
        if len(self.data) < self.ema_trend_period + 5:
            return

        price = self.data.Close[-1]
        macd_now = self.macd[-1]
        macd_prev = self.macd[-2]
        sig_now = self.macd_signal[-1]
        sig_prev = self.macd_signal[-2]
        hist_now = self.macd_hist[-1]
        hist_prev = self.macd_hist[-2]
        vol_now = self.data.Volume[-1]
        vol_ma_now = self.vol_ma[-1]
        ema_now = self.ema_trend[-1]
        atr_now = self.atr[-1]

        if np.isnan(macd_now) or np.isnan(sig_now) or np.isnan(vol_ma_now) or np.isnan(atr_now) or np.isnan(ema_now):
            return

        vol_ratio = vol_now / vol_ma_now if vol_ma_now > 0 else 0

        # Crossover detection (no backtesting.lib)
        bull_cross = macd_prev < sig_prev and macd_now > sig_now
        bear_cross = macd_prev > sig_prev and macd_now < sig_now

        # ============ EXIT LOGIC ============
        if self.position:
            bars_in_trade = len(self.data) - self.entry_bar if self.entry_bar else 0

            # Hard stop check
            if self.position.is_long and price <= self.stop_price:
                print(f"🛑 MOON DEV STOP HIT (LONG) @ {price:.2f} 🌙")
                self.position.close()
                self.entry_bar = None
                return
            if self.position.is_short and price >= self.stop_price:
                print(f"🛑 MOON DEV STOP HIT (SHORT) @ {price:.2f} 🌙")
                self.position.close()
                self.entry_bar = None
                return

            # Long exit: bearish divergence or bearish crossover
            if self.position.is_long:
                # Bearish divergence: price higher high, MACD lower high
                price_hh = self.data.High[-1] >= self.swing_high[-2]
                macd_lh = macd_now < self.macd[-self.swing_window] if not np.isnan(self.macd[-self.swing_window]) else False
                bearish_div = price_hh and macd_lh

                if bear_cross:
                    print(f"🌙 EXIT LONG: Bearish MACD crossover @ {price:.2f} ✨")
                    self.position.close()
                    self.entry_bar = None
                    return
                if bearish_div and hist_now < hist_prev and hist_now < 0:
                    print(f"🌙 EXIT LONG: Bearish divergence confirmed @ {price:.2f} ✨")
                    self.position.close()
                    self.entry_bar = None
                    return
                if bars_in_trade >= self.max_bars_in_trade:
                    print(f"🌙 EXIT LONG: Time-based exit @ {price:.2f} ✨")
                    self.position.close()
                    self.entry_bar = None
                    return

            # Short exit: bullish divergence or bullish crossover
            if self.position.is_short:
                price_ll = self.data.Low[-1] <= self.swing_low[-2]
                macd_hl = macd_now > self.macd[-self.swing_window] if not np.isnan(self.macd[-self.swing_window]) else False
                bullish_div = price_ll and macd_hl

                if bull_cross:
                    print(f"🌙 EXIT SHORT: Bullish MACD crossover @ {price:.2f} ✨")
                    self.position.close()
                    self.entry_bar = None
                    return
                if bullish_div and hist_now > hist_prev and hist_now > 0:
                    print(f"🌙 EXIT SHORT: Bullish divergence confirmed @ {price:.2f} ✨")
                    self.position.close()
                    self.entry_bar = None
                    return
                if bars_in_trade >= self.max_bars_in_trade:
                    print(f"🌙 EXIT SHORT: Time-based exit @ {price:.2f} ✨")
                    self.position.close()
                    self.entry_bar = None
                    return

        # ============ ENTRY LOGIC ============
        if not self.position:
            # LONG ENTRY
            if bull_cross and vol_ratio > self.vol_ratio_threshold:
                if price > ema_now and macd_now > 0:
                    stop = price - (self.atr_stop_mult * atr_now)
                    risk_per_unit = price - stop
                    if risk_per_unit > 0:
                        risk_amount = self.equity * self.risk_pct
                        size = int(round(risk_amount / risk_per_unit))
                        size = max(1, min(size, int(self.equity / price)))
                        if size > 0:
                            print(f"🚀 MOON DEV LONG ENTRY @ {price:.2f} | VolRatio: {vol_ratio:.2f} | MACD: {macd_now:.2f} | Size: {size} 🌙")
                            self.buy(size=size)
                            self.entry_bar = len(self.data)
                            self.entry_price = price
                            self.stop_price = stop

            # SHORT ENTRY
            elif bear_cross and vol_ratio > self.vol_ratio_threshold:
                if price < ema_now and macd_now < 0:
                    stop = price + (self.atr_stop_mult * atr_now)
                    risk_per_unit = stop - price
                    if risk_per_unit > 0:
                        risk_amount = self.equity * self.risk_pct
                        size = int(round(risk_amount / risk_per_unit))
                        size = max(1, min(size, int(self.equity / price)))
                        if size > 0:
                            print(f"🚀 MOON DEV SHORT ENTRY @ {price:.2f} | VolRatio: {vol_ratio:.2f} | MACD: {macd_now:.2f} | Size: {size} 🌙")
                            self.sell(size=size)
                            self.entry_bar = len(self.data)
                            self.entry_price = price
                            self.stop_price = stop


print("🌙 Setting up backtest engine... 🚀")
bt = Backtest(
    data,
    VolumePulseMACD,
    cash=1_000_000,
    commission=0.001,
    exclusive_orders=True
)

stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙 Moon Dev VolumePulseMACD backtest complete! ✨🚀")