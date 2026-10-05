import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Load data
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Map to proper case
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

# Set datetime index
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')

# Ensure numeric types (fixes talib "input array type is not double")
for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    data[col] = pd.to_numeric(data[col], errors='coerce').astype(np.float64)

data = data.dropna()

print(f"🌙 Moon Dev Data Loaded: {len(data)} bars ✨")
print(f"🚀 Columns: {list(data.columns)}")


class DualTriggerSurge(Strategy):
    # Parameters
    vol_ma_period = 20
    vol_mult = 1.5
    rsi_period = 14
    rsi_long = 55
    rsi_short = 45
    ema_period = 200
    atr_period = 14
    swing_period = 20
    risk_pct = 0.01  # 1% risk per trade
    rr_target1 = 1.0
    rr_target2 = 3.0
    time_stop_bars = 10
    atr_trail_mult = 1.5

    def init(self):
        print("🌙 Initializing DualTrigger Surge indicators... ✨")
        # Wrap in np.asarray(..., dtype=np.float64) to guarantee double type for talib
        self.vol_ma = self.I(talib.SMA, np.asarray(self.data.Volume, dtype=np.float64), timeperiod=self.vol_ma_period)
        self.rsi = self.I(talib.RSI, np.asarray(self.data.Close, dtype=np.float64), timeperiod=self.rsi_period)

        def _macd(close):
            m, s, h = talib.MACD(np.asarray(close, dtype=np.float64), fastperiod=12, slowperiod=26, signalperiod=9)
            return m, s, h

        self.macd, self.macd_signal, self.macd_hist = self.I(
            _macd, np.asarray(self.data.Close, dtype=np.float64)
        )
        self.ema200 = self.I(talib.EMA, np.asarray(self.data.Close, dtype=np.float64), timeperiod=self.ema_period)
        self.atr = self.I(talib.ATR,
                          np.asarray(self.data.High, dtype=np.float64),
                          np.asarray(self.data.Low, dtype=np.float64),
                          np.asarray(self.data.Close, dtype=np.float64),
                          timeperiod=self.atr_period)
        self.swing_high = self.I(talib.MAX, np.asarray(self.data.High, dtype=np.float64), timeperiod=self.swing_period)
        self.swing_low = self.I(talib.MIN, np.asarray(self.data.Low, dtype=np.float64), timeperiod=self.swing_period)

        # Track trade state
        self.entry_price = None
        self.stop_price = None
        self.target1 = None
        self.target2 = None
        self.bars_in_trade = 0
        self.partial_taken = False

    def next(self):
        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        volume = self.data.Volume[-1]
        vol_ma = self.vol_ma[-1]
        rsi = self.rsi[-1]
        macd_hist = self.macd_hist[-1]
        ema200 = self.ema200[-1]
        atr = self.atr[-1]

        if len(self.data) < 2:
            return
        ref_high = self.swing_high[-2]  # prior swing high (exclude current bar)
        ref_low = self.swing_low[-2]

        if np.isnan(vol_ma) or np.isnan(rsi) or np.isnan(macd_hist) or np.isnan(ema200) or np.isnan(atr):
            return
        if np.isnan(ref_high) or np.isnan(ref_low):
            return

        # Manage open position
        if self.position:
            self.bars_in_trade += 1
            entry = self.entry_price
            stop = self.stop_price

            if self.position.is_long:
                # Time stop
                if self.bars_in_trade >= self.time_stop_bars and not self.partial_taken:
                    print(f"⏰ Moon Dev Time Stop Hit (Long) @ {price:.2f} 🌙")
                    self.position.close()
                    self._reset_state()
                    return

                # Partial at 1R
                if not self.partial_taken and price >= self.target1:
                    print(f"🎯 Moon Dev Target 1 Hit (Long) @ {price:.2f} — taking partial ✨")
                    self.position.close(portion=0.5)
                    self.partial_taken = True
                    # Trail stop to breakeven
                    self.stop_price = entry

                # Target 2
                if self.partial_taken and price >= self.target2:
                    print(f"🚀 Moon Dev Target 2 Hit (Long) @ {price:.2f} — full exit 🌙")
                    self.position.close()
                    self._reset_state()
                    return

                # ATR trailing stop
                new_stop = price - atr * self.atr_trail_mult
                if new_stop > self.stop_price:
                    self.stop_price = new_stop

                # Stop loss check
                if low <= self.stop_price:
                    print(f"🛑 Moon Dev Stop Loss (Long) @ {self.stop_price:.2f} 💥")
                    self.position.close()
                    self._reset_state()
                    return

            elif self.position.is_short:
                if self.bars_in_trade >= self.time_stop_bars and not self.partial_taken:
                    print(f"⏰ Moon Dev Time Stop Hit (Short) @ {price:.2f} 🌙")
                    self.position.close()
                    self._reset_state()
                    return

                if not self.partial_taken and price <= self.target1:
                    print(f"🎯 Moon Dev Target 1 Hit (Short) @ {price:.2f} — taking partial ✨")
                    self.position.close(portion=0.5)
                    self.partial_taken = True
                    self.stop_price = entry

                if self.partial_taken and price <= self.target2:
                    print(f"🚀 Moon Dev Target 2 Hit (Short) @ {price:.2f} — full exit 🌙")
                    self.position.close()
                    self._reset_state()
                    return

                new_stop = price + atr * self.atr_trail_mult
                if new_stop < self.stop_price:
                    self.stop_price = new_stop

                if high >= self.stop_price:
                    print(f"🛑 Moon Dev Stop Loss (Short) @ {self.stop_price:.2f} 💥")
                    self.position.close()
                    self._reset_state()
                    return
            return

        # Entry logic
        vol_expansion = volume >= self.vol_mult * vol_ma
        momentum_long = (rsi > self.rsi_long) or (macd_hist > 0)
        momentum_short = (rsi < self.rsi_short) or (macd_hist < 0)

        long_breakout = price > ref_high
        short_breakout = price < ref_low

        # Long entry
        if long_breakout and vol_expansion and momentum_long and price > ema200:
            stop = min(low, ref_high)
            risk = price - stop
            if risk <= 0:
                return
            # Position sizing: risk_pct of equity / risk per unit
            equity = self.equity
            risk_amount = equity * self.risk_pct
            size = int(round(risk_amount / risk))
            if size < 1:
                size = 1

            print(f"🌙✨ DUALTRIGGER LONG SIGNAL ✨🌙 | Price: {price:.2f} | RefHigh: {ref_high:.2f} | Vol: {volume:.2f} vs {vol_ma*self.vol_mult:.2f} | RSI: {rsi:.2f} | MACD_H: {macd_hist:.4f} | EMA200: {ema200:.2f} | Size: {size} 🚀")
            self.buy(size=size)
            self.entry_price = price
            self.stop_price = stop
            self.target1 = price + risk * self.rr_target1
            self.target2 = price + risk * self.rr_target2
            self.bars_in_trade = 0
            self.partial_taken = False

        # Short entry
        elif short_breakout and vol_expansion and momentum_short and price < ema200:
            stop = max(high, ref_low)
            risk = stop - price
            if risk <= 0:
                return
            equity = self.equity
            risk_amount = equity * self.risk_pct
            size = int(round(risk_amount / risk))
            if size < 1:
                size = 1

            print(f"🌙✨ DUALTRIGGER SHORT SIGNAL ✨🌙 | Price: {price:.2f} | RefLow: {ref_low:.2f} | Vol: {volume:.2f} vs {vol_ma*self.vol_mult:.2f} | RSI: {rsi:.2f} | MACD_H: {macd_hist:.4f} | EMA200: {ema200:.2f} | Size: {size} 🚀")
            self.sell(size=size)
            self.entry_price = price
            self.stop_price = stop
            self.target1 = price - risk * self.rr_target1
            self.target2 = price - risk * self.rr_target2
            self.bars_in_trade = 0
            self.partial_taken = False

    def _reset_state(self):
        self.entry_price = None
        self.stop_price = None
        self.target1 = None
        self.target2 = None
        self.bars_in_trade = 0
        self.partial_taken = False


print("🌙 Moon Dev Backtest Starting... 🚀")
bt = Backtest(data, DualTriggerSurge, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)