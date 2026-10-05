import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Load data
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Map proper case
data = data.rename(columns={
    'datetime': 'Datetime',
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

data['Datetime'] = pd.to_datetime(data['Datetime'])
data = data.set_index('Datetime')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

# Ensure numeric dtypes for talib
for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    data[col] = pd.to_numeric(data[col], errors='coerce').astype('float64')

data = data.dropna()

print("🌙 MOON DEV BACKTEST INITIALIZED 🚀")
print(f"📊 Data loaded: {len(data)} bars")
print(f"📈 Date range: {data.index[0]} to {data.index[-1]}")
print("=" * 60)


class SqueezeFibonacci(Strategy):
    # Strategy parameters
    bb_period = 20
    bb_std = 2.0
    kc_period = 20
    kc_atr_mult = 1.5
    vol_period = 20
    vol_mult = 1.5
    squeeze_bars = 5
    risk_pct = 0.02
    atr_period = 14

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        # Bollinger Bands
        self.bb_upper, self.bb_mid, self.bb_lower = self.I(
            talib.BBANDS, close, timeperiod=self.bb_period,
            nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0
        )

        # Keltner Channels: EMA + ATR
        self.kc_mid = self.I(talib.EMA, close, timeperiod=self.kc_period)
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.kc_period)
        self.kc_upper = self.kc_mid + self.kc_atr_mult * self.atr
        self.kc_lower = self.kc_mid - self.kc_atr_mult * self.atr

        # Volume MA
        self.vol_ma = self.I(talib.SMA, volume, timeperiod=self.vol_period)

        # ATR for risk
        self.atr_risk = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        # Squeeze tracking (highest high / lowest low during squeeze)
        self.squeeze_high = None
        self.squeeze_low = None
        self.squeeze_start_bar = None
        self.squeeze_count = 0

        # Trade management
        self.entry_price = None
        self.stop_price = None
        self.tp1 = None
        self.tp2 = None
        self.tp3 = None
        self.tp1_hit = False
        self.tp2_hit = False
        self.direction = None
        self.entry_bar = None
        self.squeeze_duration = 0
        self.reentry_used = False

        print("🌙 Indicators initialized: BB, KC, ATR, Volume MA ✨")

    def next(self):
        i = len(self.data) - 1
        if i < max(self.bb_period, self.kc_period, self.vol_period) + 2:
            return

        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        vol = self.data.Volume[-1]

        bb_up = self.bb_upper[-1]
        bb_lo = self.bb_lower[-1]
        kc_up = self.kc_upper[-1]
        kc_lo = self.kc_lower[-1]
        vol_ma = self.vol_ma[-1]
        atr = self.atr_risk[-1]

        # Detect squeeze: BB inside KC
        in_squeeze = (bb_up < kc_up) and (bb_lo > kc_lo)

        # Track squeeze range
        if in_squeeze:
            if self.squeeze_count == 0:
                self.squeeze_high = high
                self.squeeze_low = low
                self.squeeze_start_bar = i
            else:
                self.squeeze_high = max(self.squeeze_high, high)
                self.squeeze_low = min(self.squeeze_low, low)
            self.squeeze_count += 1
        else:
            # Squeeze just ended or not active
            if self.squeeze_count > 0:
                self.squeeze_duration = self.squeeze_count
            if not self.position and self.squeeze_count > 0:
                pass  # keep range until breakout or new squeeze

        # ============ MANAGE OPEN POSITION ============
        if self.position:
            # Time stop
            bars_in_trade = i - self.entry_bar
            if not self.tp1_hit and bars_in_trade > 2 * max(self.squeeze_duration, 5):
                print(f"⏰ MOON DEV TIME STOP - exiting at {price:.2f} 🌙")
                self.position.close()
                self._reset_trade_state()
                return

            # Trailing stop after TP1: use 20 EMA
            if self.tp1_hit:
                ema20 = self.kc_mid[-1]
                if self.direction == 'long' and ema20 > self.stop_price:
                    self.stop_price = ema20
                elif self.direction == 'short' and ema20 < self.stop_price:
                    self.stop_price = ema20

            if self.direction == 'long':
                # Stop loss
                if low <= self.stop_price:
                    print(f"🛑 MOON DEV STOP HIT LONG at {self.stop_price:.2f} 💥")
                    self.position.close()
                    self._reset_trade_state()
                    return
                # TP levels
                if not self.tp1_hit and high >= self.tp1:
                    print(f"🎯 TP1 HIT LONG at {self.tp1:.2f} - scaling 50% 🚀")
                    self.position.close(0.5)
                    self.tp1_hit = True
                if self.tp1_hit and not self.tp2_hit and high >= self.tp2:
                    print(f"🎯 TP2 HIT LONG at {self.tp2:.2f} - scaling 30% 🚀🚀")
                    self.position.close(0.6)
                    self.tp2_hit = True
                if self.tp2_hit and high >= self.tp3:
                    print(f"🎯 TP3 HIT LONG at {self.tp3:.2f} - closing rest 🚀🚀🚀")
                    self.position.close()
                    self._reset_trade_state()
                    return

            elif self.direction == 'short':
                if high >= self.stop_price:
                    print(f"🛑 MOON DEV STOP HIT SHORT at {self.stop_price:.2f} 💥")
                    self.position.close()
                    self._reset_trade_state()
                    return
                if not self.tp1_hit and low <= self.tp1:
                    print(f"🎯 TP1 HIT SHORT at {self.tp1:.2f} - scaling 50% 🚀")
                    self.position.close(0.5)
                    self.tp1_hit = True
                if self.tp1_hit and not self.tp2_hit and low <= self.tp2:
                    print(f"🎯 TP2 HIT SHORT at {self.tp2:.2f} - scaling 30% 🚀🚀")
                    self.position.close(0.6)
                    self.tp2_hit = True
                if self.tp2_hit and low <= self.tp3:
                    print(f"🎯 TP3 HIT SHORT at {self.tp3:.2f} - closing rest 🚀🚀🚀")
                    self.position.close()
                    self._reset_trade_state()
                    return
            return

        # ============ ENTRY LOGIC ============
        if self.squeeze_count < self.squeeze_bars:
            return
        if self.squeeze_high is None or self.squeeze_low is None:
            return

        vol_spike = vol > self.vol_mult * vol_ma
        SH = self.squeeze_high
        SL = self.squeeze_low
        R = SH - SL
        if R <= 0:
            return

        # Long breakout
        long_break = price > bb_up and vol_spike
        # Short breakout
        short_break = price < bb_lo and vol_spike

        if long_break:
            stop = max(SL, price - atr)
            if price - stop < atr:
                stop = price - atr
            risk = price - stop
            if risk <= 0:
                return
            size = int(round((self.equity * self.risk_pct) / risk))
            if size < 1:
                size = 1
            self.tp1 = SL + 1.0 * R
            self.tp2 = SL + 1.272 * R
            self.tp3 = SL + 1.618 * R
            self.entry_price = price
            self.stop_price = stop
            self.direction = 'long'
            self.entry_bar = i
            self.tp1_hit = False
            self.tp2_hit = False
            self.buy(size=size)
            print(f"🌙✨ LONG BREAKOUT! Price={price:.2f} SL={stop:.2f} "
                  f"TP1={self.tp1:.2f} TP2={self.tp2:.2f} TP3={self.tp3:.2f} "
                  f"Size={size} 🚀")
            self.squeeze_count = 0

        elif short_break:
            stop = min(SH, price + atr)
            if stop - price < atr:
                stop = price + atr
            risk = stop - price
            if risk <= 0:
                return
            size = int(round((self.equity * self.risk_pct) / risk))
            if size < 1:
                size = 1
            self.tp1 = SH - 1.0 * R
            self.tp2 = SH - 1.272 * R
            self.tp3 = SH - 1.618 * R
            self.entry_price = price
            self.stop_price = stop
            self.direction = 'short'
            self.entry_bar = i
            self.tp1_hit = False
            self.tp2_hit = False
            self.sell(size=size)
            print(f"🌙✨ SHORT BREAKOUT! Price={price:.2f} SL={stop:.2f} "
                  f"TP1={self.tp1:.2f} TP2={self.tp2:.2f} TP3={self.tp3:.2f} "
                  f"Size={size} 🚀")
            self.squeeze_count = 0

    def _reset_trade_state(self):
        self.entry_price = None
        self.stop_price = None
        self.tp1 = None
        self.tp2 = None
        self.tp3 = None
        self.tp1_hit = False
        self.tp2_hit = False
        self.direction = None
        self.squeeze_count = 0
        self.squeeze_high = None
        self.squeeze_low = None


print("🌙 Starting Moon Dev Squeeze-Fibonacci Backtest... 🚀")
bt = Backtest(data, SqueezeFibonacci, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙✨ Backtest complete! Moon Dev out! 🚀🚀🚀")