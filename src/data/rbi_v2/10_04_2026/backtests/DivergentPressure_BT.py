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
data.columns = [c.capitalize() for c in data.columns]
data = data.rename(columns={'Datetime': 'Date'})
data['Date'] = pd.to_datetime(data['Date'])
data = data.set_index('Date')

print("🌙 Moon Dev Data Loaded! Shape:", data.shape)
print("✨ Columns:", list(data.columns))


class DivergentPressure(Strategy):
    rsi_period = 14
    ema_fast = 50
    ema_slow = 200
    atr_period = 14
    vol_lookback = 10
    rsi_overbought = 70
    rsi_exit = 50
    atr_sl_mult = 1.5
    atr_tp_mult = 1.5
    risk_pct = 0.02
    max_bars_held = 30

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        self.rsi = self.I(talib.RSI, close, timeperiod=self.rsi_period)
        self.ema_f = self.I(talib.EMA, close, timeperiod=self.ema_fast)
        self.ema_s = self.I(talib.EMA, close, timeperiod=self.ema_slow)
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        # Up/down volume: up volume where close > prev close, else down volume
        close_arr = np.asarray(close, dtype=float)
        vol_arr = np.asarray(volume, dtype=float)
        up_vol = np.where(close_arr > np.roll(close_arr, 1), vol_arr, 0.0)
        dn_vol = np.where(close_arr < np.roll(close_arr, 1), vol_arr, 0.0)
        up_vol[0] = 0.0
        dn_vol[0] = 0.0

        self.up_vol_sum = self.I(lambda: pd.Series(up_vol).rolling(self.vol_lookback).sum().values)
        self.dn_vol_sum = self.I(lambda: pd.Series(dn_vol).rolling(self.vol_lookback).sum().values)

        self.bars_held = 0
        self.entry_price = None
        self.stop_price = None
        self.tp_price = None

        print("🌙✨ DivergentPressure indicators initialized! 🚀")

    def next(self):
        price = self.data.Close[-1]
        rsi = self.rsi[-1]
        ema_f = self.ema_f[-1]
        ema_s = self.ema_s[-1]
        atr = self.atr[-1]
        up_v = self.up_vol_sum[-1]
        dn_v = self.dn_vol_sum[-1]

        if np.isnan(rsi) or np.isnan(ema_f) or np.isnan(ema_s) or np.isnan(atr):
            return

        # Manage open short position
        if self.position:
            self.bars_held += 1
            # Take profit
            if price <= self.tp_price:
                print(f"🌙💰 TP HIT! Closing short at {price:.2f} | RSI={rsi:.2f}")
                self.position.close()
                self.bars_held = 0
                return
            # Stop loss
            if price >= self.stop_price:
                print(f"🌙🛑 SL HIT! Closing short at {price:.2f} | RSI={rsi:.2f}")
                self.position.close()
                self.bars_held = 0
                return
            # RSI exit
            if rsi < self.rsi_exit:
                print(f"🌙📉 RSI EXIT! RSI={rsi:.2f} < {self.rsi_exit} | Closing at {price:.2f}")
                self.position.close()
                self.bars_held = 0
                return
            # Time exit
            if self.bars_held >= self.max_bars_held:
                print(f"🌙⏰ TIME EXIT! Held {self.bars_held} bars | Closing at {price:.2f}")
                self.position.close()
                self.bars_held = 0
                return
            return

        # Entry conditions
        uptrend = price > ema_f and ema_f > ema_s
        overbought = rsi > self.rsi_overbought
        vol_divergence = up_v > dn_v

        if uptrend and overbought and vol_divergence:
            stop_price = price + self.atr_sl_mult * atr
            tp_price = price - self.atr_tp_mult * atr

            # Risk-based position sizing
            risk_per_unit = stop_price - price
            if risk_per_unit <= 0:
                return
            risk_amount = self.equity * self.risk_pct
            size = int(round(risk_amount / risk_per_unit))
            if size <= 0:
                size = 1

            print(f"🌙🚀 SHORT ENTRY! Price={price:.2f} | RSI={rsi:.2f} | UpVol={up_v:.2f} > DnVol={dn_v:.2f}")
            print(f"   SL={stop_price:.2f} | TP={tp_price:.2f} | Size={size}")

            self.sell(size=size)
            self.entry_price = price
            self.stop_price = stop_price
            self.tp_price = tp_price
            self.bars_held = 0


bt = Backtest(data, DivergentPressure, cash=1_000_000, commission=0.002)
stats = bt.run()
print(stats)
print(stats._strategy)