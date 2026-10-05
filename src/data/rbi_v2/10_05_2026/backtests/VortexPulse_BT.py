import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# 🌙 Moon Dev VortexPulse Backtest 🚀

class VortexPulse(Strategy):
    # Strategy parameters
    vortex_period = 14
    vol_ma_period = 20
    ema_period = 20
    atr_period = 14
    vol_surge_factor = 1.5
    ema_proximity_pct = 0.005  # 0.5%
    risk_pct = 0.01  # 1% risk per trade
    atr_stop_mult = 1.5
    swing_lookback = 20
    time_stop_bars = 10
    min_atr_pct = 0.005  # 0.5% volatility filter

    def init(self):
        print("🌙 Moon Dev VortexPulse initializing... ✨")
        h = self.data.High
        l = self.data.Low
        c = self.data.Close
        v = self.data.Volume

        # Vortex Indicator
        def vortex_calc(high, low, close, period):
            vm_plus = np.abs(high - np.roll(low, 1))
            vm_minus = np.abs(low - np.roll(high, 1))
            tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
            vm_plus[0] = np.nan
            vm_minus[0] = np.nan
            tr[0] = np.nan
            vm_plus_s = pd.Series(vm_plus).rolling(period).sum().values
            vm_minus_s = pd.Series(vm_minus).rolling(period).sum().values
            tr_s = pd.Series(tr).rolling(period).sum().values
            with np.errstate(divide='ignore', invalid='ignore'):
                vi_plus = vm_plus_s / tr_s
                vi_minus = vm_minus_s / tr_s
            return vi_plus, vi_minus

        self.vi_plus, self.vi_minus = self.I(vortex_calc, h, l, c, self.vortex_period, name="VI")

        # Volume MA
        self.vol_ma = self.I(talib.SMA, v, timeperiod=self.vol_ma_period, name="VolMA")

        # EMA
        self.ema = self.I(talib.EMA, c, timeperiod=self.ema_period, name="EMA20")

        # ATR
        self.atr = self.I(talib.ATR, h, l, c, timeperiod=self.atr_period, name="ATR")

        # Swing high/low
        self.swing_high = self.I(talib.MAX, h, timeperiod=self.swing_lookback, name="SwingHigh")
        self.swing_low = self.I(talib.MIN, l, timeperiod=self.swing_lookback, name="SwingLow")

        # State tracking
        self.entry_price = None
        self.stop_price = None
        self.target_127 = None
        self.target_161 = None
        self.target_200 = None
        self.bars_in_trade = 0
        self.partial_exited = False
        self.be_moved = False

        print("🌙 Moon Dev VortexPulse ready! 🚀")

    def next(self):
        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        volume = self.data.Volume[-1]
        ema = self.ema[-1]
        atr = self.atr[-1]
        vi_p = self.vi_plus[-1]
        vi_m = self.vi_minus[-1]
        vi_p_prev = self.vi_plus[-2]
        vi_m_prev = self.vi_minus[-2]
        vol_ma = self.vol_ma[-1]
        swing_hi = self.swing_high[-1]
        swing_lo = self.swing_low[-1]

        if np.isnan(vi_p) or np.isnan(vi_m) or np.isnan(ema) or np.isnan(atr):
            return

        # Manage open position
        if self.position:
            self.bars_in_trade += 1

            if self.position.is_long:
                # Momentum exit
                if crossover(self.vi_minus, self.vi_plus):
                    print(f"🌙 Momentum exit LONG at {price:.2f} 🚀")
                    self.position.close()
                    self._reset()
                    return

                # Move stop to breakeven after 127.2%
                if not self.be_moved and self.target_127 and high >= self.target_127:
                    self.stop_price = max(self.stop_price, self.entry_price)
                    self.be_moved = True
                    print(f"🌙 Stop moved to breakeven for LONG at {self.entry_price:.2f} ✨")

                # Trail with EMA after 161.8%
                if self.target_161 and high >= self.target_161:
                    new_stop = max(self.stop_price, ema)
                    if new_stop > self.stop_price:
                        self.stop_price = new_stop
                        print(f"🌙 Trailing stop to EMA {ema:.2f} for LONG 📈")

                # Partial exits
                if not self.partial_exited and self.target_127 and high >= self.target_127:
                    half = int(round(self.position.size * 0.5))
                    if half > 0:
                        self.position.close(half)
                        self.partial_exited = True
                        print(f"🌙 Took 50% profit at 127.2% ({self.target_127:.2f}) 💰")

                if self.target_161 and high >= self.target_161 and self.partial_exited:
                    thirty = int(round(self.position.size * 0.6))
                    if thirty > 0:
                        self.position.close(thirty)
                        print(f"🌙 Took 30% more at 161.8% ({self.target_161:.2f}) 💰")

                if self.target_200 and high >= self.target_200:
                    print(f"🌙 Full exit at 200% ({self.target_200:.2f}) 🎯")
                    self.position.close()
                    self._reset()
                    return

                # Stop loss
                if low <= self.stop_price:
                    print(f"🌙 STOP LOSS LONG at {self.stop_price:.2f} 💥")
                    self.position.close()
                    self._reset()
                    return

                # Time stop
                if self.bars_in_trade >= self.time_stop_bars and not (self.target_127 and high >= self.target_127):
                    print(f"🌙 Time stop LONG after {self.bars_in_trade} bars ⏰")
                    self.position.close()
                    self._reset()
                    return

            elif self.position.is_short:
                if crossover(self.vi_plus, self.vi_minus):
                    print(f"🌙 Momentum exit SHORT at {price:.2f} 🚀")
                    self.position.close()
                    self._reset()
                    return

                if not self.be_moved and self.target_127 and low <= self.target_127:
                    self.stop_price = min(self.stop_price, self.entry_price)
                    self.be_moved = True
                    print(f"🌙 Stop moved to breakeven for SHORT at {self.entry_price:.2f} ✨")

                if self.target_161 and low <= self.target_161:
                    new_stop = min(self.stop_price, ema)
                    if new_stop < self.stop_price:
                        self.stop_price = new_stop
                        print(f"🌙 Trailing stop to EMA {ema:.2f} for SHORT 📉")

                if not self.partial_exited and self.target_127 and low <= self.target_127:
                    half = int(round(self.position.size * 0.5))
                    if half > 0:
                        self.position.close(half)
                        self.partial_exited = True
                        print(f"🌙 Took 50% profit at 127.2% ({self.target_127:.2f}) 💰")

                if self.target_161 and low <= self.target_161 and self.partial_exited:
                    thirty = int(round(self.position.size * 0.6))
                    if thirty > 0:
                        self.position.close(thirty)
                        print(f"🌙 Took 30% more at 161.8% ({self.target_161:.2f}) 💰")

                if self.target_200 and low <= self.target_200:
                    print(f"🌙 Full exit at 200% ({self.target_200:.2f}) 🎯")
                    self.position.close()
                    self._reset()
                    return

                if high >= self.stop_price:
                    print(f"🌙 STOP LOSS SHORT at {self.stop_price:.2f} 💥")
                    self.position.close()
                    self._reset()
                    return

                if self.bars_in_trade >= self.time_stop_bars and not (self.target_127 and low <= self.target_127):
                    print(f"🌙 Time stop SHORT after {self.bars_in_trade} bars ⏰")
                    self.position.close()
                    self._reset()
                    return

        # Entry logic (only if no position)
        if not self.position:
            # Volatility filter
            if atr < price * self.min_atr_pct:
                return

            vol_surge = volume >= vol_ma * self.vol_surge_factor

            # Long setup
            long_cross = vi_p_prev <= vi_m_prev and vi_p > vi_m
            near_ema_long = (price >= ema) and (abs(price - ema) / ema <= self.ema_proximity_pct or low <= ema <= high)

            if long_cross and vol_surge:
                # Store pending long signal; wait for pullback
                self.pending_long = True
                print(f"🌙 Vortex bullish crossover + volume surge detected! Waiting for EMA pullback... ✨")

            if getattr(self, 'pending_long', False) and near_ema_long:
                self._enter_long(price, atr, swing_lo)
                self.pending_long = False

            # Short setup
            short_cross = vi_m_prev <= vi_p_prev and vi_m > vi_p
            near_ema_short = (price <= ema) and (abs(price - ema) / ema <= self.ema_proximity_pct or low <= ema <= high)

            if short_cross and vol_surge:
                self.pending_short = True
                print(f"🌙 Vortex bearish crossover + volume surge detected! Waiting for EMA rally... ✨")

            if getattr(self, 'pending_short', False) and near_ema_short:
                self._enter_short(price, atr, swing_hi)
                self.pending_short = False

    def _enter_long(self, price, atr, swing_lo):
        stop = max(swing_lo, price - self.atr_stop_mult * atr)
        risk = price - stop
        if risk <= 0:
            return
        risk_amount = self.equity * self.risk_pct
        size = int(round(risk_amount / risk))
        if size < 1:
            return
        # Cap size so we don't exceed equity
        size = min(size, int(self.equity / price))
        if size < 1:
            return

        # Fibonacci extensions from prior swing
        swing_range = price - swing_lo
        self.target_127 = price + swing_range * 0.272
        self.target_161 = price + swing_range * 0.618
        self.target_200 = price + swing_range * 1.0

        self.entry_price = price
        self.stop_price = stop
        self.bars_in_trade = 0
        self.partial_exited = False
        self.be_moved = False

        self.buy(size=size)
        print(f"🌙 LONG ENTRY at {price:.2f} | Size: {size} | Stop: {stop:.2f} | Targets: {self.target_127:.2f}/{self.target_161:.2f}/{self.target_200:.2f} 🚀")

    def _enter_short(self, price, atr, swing_hi):
        stop = min(swing_hi, price + self.atr_stop_mult * atr)
        risk = stop - price
        if risk <= 0:
            return
        risk_amount = self.equity * self.risk_pct
        size = int(round(risk_amount / risk))
        if size < 1:
            return
        size = min(size, int(self.equity / price))
        if size < 1:
            return

        swing_range = swing_hi - price
        self.target_127 = price - swing_range * 0.272
        self.target_161 = price - swing_range * 0.618
        self.target_200 = price - swing_range * 1.0

        self.entry_price = price
        self.stop_price = stop
        self.bars_in_trade = 0
        self.partial_exited = False
        self.be_moved = False

        self.sell(size=size)
        print(f"🌙 SHORT ENTRY at {price:.2f} | Size: {size} | Stop: {stop:.2f} | Targets: {self.target_127:.2f}/{self.target_161:.2f}/{self.target_200:.2f} 🚀")

    def _reset(self):
        self.entry_price = None
        self.stop_price = None
        self.target_127 = None
        self.target_161 = None
        self.target_200 = None
        self.bars_in_trade = 0
        self.partial_exited = False
        self.be_moved = False


# Load data
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')

# 🌙 Data cleaning
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

# Ensure datetime index
if 'Datetime' in data.columns:
    data['Datetime'] = pd.to_datetime(data['Datetime'])
    data = data.set_index('Datetime')

print("🌙 Moon Dev data loaded! Shape:", data.shape, "✨")
print(data.head())

# Run backtest
bt = Backtest(data, VortexPulse, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)