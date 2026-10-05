import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

print("🌙✨ SqueezeIgnition Backtest Initializing... Let's catch the ignition! 🚀")

# Load and clean data
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
print(f"🌙 Data loaded: {len(data)} bars from {data.index[0]} to {data.index[-1]} ✨")


class SqueezeIgnition(Strategy):
    bb_period = 20
    bb_std = 2.0
    rsi_period = 14
    atr_period = 14
    vol_period = 20
    squeeze_threshold = 0.05  # 5% - relaxed from 1% to allow trades
    squeeze_min_bars = 3
    rsi_trigger = 35
    vol_mult = 1.15
    atr_tp_mult = 1.5
    atr_sl_mult = 0.5
    time_stop_bars = 12
    risk_pct = 0.02  # 2% risk per trade

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        # Bollinger Bands
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS, close, timeperiod=self.bb_period,
            nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0
        )

        # RSI
        self.rsi = self.I(talib.RSI, close, timeperiod=self.rsi_period)

        # ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        # Volume SMA - ensure volume is float64 for talib
        self.vol_sma = self.I(talib.SMA, volume.astype(np.float64), timeperiod=self.vol_period)

        # Band Width — guard against zero/NaN middle band
        self.band_width = self.I(
            lambda u, m, l: np.where(m > 0, (u - l) / m, np.nan),
            self.bb_upper, self.bb_middle, self.bb_lower
        )

        print("🌙✨ Indicators initialized: BB, RSI, ATR, VolSMA 🚀")

    def next(self):
        if len(self.data) < max(self.bb_period, self.rsi_period, self.atr_period, self.vol_period) + 5:
            return

        price = self.data.Close[-1]
        atr_val = self.atr[-1]

        # Manage existing position
        if self.position:
            entry_price = self.trades[-1].entry_price
            entry_bar = self.trades[-1].entry_bar
            # Time stop
            if len(self.data) - entry_bar >= self.time_stop_bars:
                print(f"⏰ Time stop hit at {price:.2f} — exiting to avoid stagnation 🌙")
                self.position.close()
                return
            # TP/SL
            tp_price = entry_price + self.atr_tp_mult * atr_val
            sl_price = entry_price - self.atr_sl_mult * atr_val
            if self.data.High[-1] >= tp_price:
                print(f"🎯 Profit target hit! Exit at ~{tp_price:.2f} 🚀")
                self.position.close()
                return
            if self.data.Low[-1] <= sl_price:
                print(f"🛑 Stop loss hit! Exit at ~{sl_price:.2f} 🌙")
                self.position.close()
                return
            return

        # Entry conditions
        if len(self.band_width) < self.squeeze_min_bars + 1:
            return

        # Squeeze persistence check (handle NaN safely)
        recent_bw = np.array(self.band_width[-self.squeeze_min_bars:])
        if np.any(np.isnan(recent_bw)):
            return
        squeeze_ok = np.all(recent_bw < self.squeeze_threshold)

        # RSI cross above trigger
        rsi_prev = self.rsi[-2]
        rsi_curr = self.rsi[-1]
        if np.isnan(rsi_prev) or np.isnan(rsi_curr):
            return
        rsi_ok = rsi_prev < self.rsi_trigger and rsi_curr > self.rsi_trigger

        # Volume surge
        vol_sma_val = self.vol_sma[-1]
        if np.isnan(vol_sma_val) or vol_sma_val <= 0:
            return
        vol_ok = self.data.Volume[-1] > self.vol_mult * vol_sma_val

        if squeeze_ok and rsi_ok and vol_ok:
            # Position sizing based on risk — use fraction of equity for backtesting.py
            stop_distance = self.atr_sl_mult * atr_val
            if stop_distance <= 0 or price <= 0 or np.isnan(stop_distance):
                return
            # Fractional sizing: risk_pct of equity, expressed as fraction of price
            size_fraction = (self.equity * self.risk_pct) / (stop_distance * price)
            # Clamp to (0, 1) — must be strictly between 0 and 1
            size_fraction = min(max(size_fraction, 0.01), 0.99)
            if size_fraction <= 0 or np.isnan(size_fraction):
                return

            print(f"🌙✨ SQUEEZE IGNITION! BW={self.band_width[-1]:.4f}, RSI={self.rsi[-1]:.2f}, "
                  f"Vol={self.data.Volume[-1]:.2f} vs SMA={self.vol_sma[-1]:.2f} 🚀")
            print(f"   Entry: {price:.2f}, ATR: {atr_val:.2f}, Size: {size_fraction:.4f}, "
                  f"SL: {price - stop_distance:.2f}, TP: {price + self.atr_tp_mult*atr_val:.2f}")

            self.buy(size=size_fraction, sl=price - stop_distance,
                     tp=price + self.atr_tp_mult * atr_val)


bt = Backtest(data, SqueezeIgnition, cash=1_000_000, commission=0.001)
stats = bt.run()

print("\n" + "="*60)
print("🌙✨ SqueezeIgnition Backtest Results ✨🚀")
print("="*60)
print(stats)
print("\n" + "="*60)
print(stats._strategy)
print("="*60)