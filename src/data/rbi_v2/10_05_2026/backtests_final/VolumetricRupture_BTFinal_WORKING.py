import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

print("🌙 Moon Dev VolumetricRupture Backtest Initializing... ✨🚀")

data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper mapping to backtesting.py requirements
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

# Ensure float64 dtype for talib compatibility
data['Open'] = data['Open'].astype(np.float64)
data['High'] = data['High'].astype(np.float64)
data['Low'] = data['Low'].astype(np.float64)
data['Close'] = data['Close'].astype(np.float64)
data['Volume'] = data['Volume'].astype(np.float64)

print(f"🌙 Data loaded: {len(data)} rows ✨")
print(f"🚀 Date range: {data.index[0]} to {data.index[-1]}")


class VolumetricRupture(Strategy):
    # Strategy parameters
    vol_ma_period = 20
    vol_multiplier = 1.5
    atr_period = 14
    donchian_period = 20
    ema_fast = 20
    ema_slow = 50
    risk_pct = 0.02
    rr_ratio = 2.0

    def init(self):
        print("🌙 Initializing VolumetricRupture indicators... ✨")
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        # Volume moving average - ensure float64
        self.vol_ma = self.I(talib.SMA, np.asarray(volume, dtype=np.float64),
                             timeperiod=self.vol_ma_period, name="VolMA")

        # ATR for stops
        self.atr = self.I(talib.ATR,
                          np.asarray(high, dtype=np.float64),
                          np.asarray(low, dtype=np.float64),
                          np.asarray(close, dtype=np.float64),
                          timeperiod=self.atr_period, name="ATR")

        # Donchian channels for breakout levels (using shifted to avoid lookahead)
        self.donchian_high = self.I(talib.MAX, np.asarray(high, dtype=np.float64),
                                    timeperiod=self.donchian_period, name="DonchianHigh")
        self.donchian_low = self.I(talib.MIN, np.asarray(low, dtype=np.float64),
                                   timeperiod=self.donchian_period, name="DonchianLow")

        # Trend EMAs
        self.ema_f = self.I(talib.EMA, np.asarray(close, dtype=np.float64),
                            timeperiod=self.ema_fast, name="EMA20")
        self.ema_s = self.I(talib.EMA, np.asarray(close, dtype=np.float64),
                            timeperiod=self.ema_slow, name="EMA50")

        print("🌙 Indicators ready! 🚀")

    def next(self):
        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        vol = self.data.Volume[-1]

        # Need enough history
        if len(self.data) < self.donchian_period + 2:
            return

        # Previous Donchian levels (avoid lookahead — use previous bar's values)
        prev_resistance = self.donchian_high[-2]
        prev_support = self.donchian_low[-2]

        vol_ma = self.vol_ma[-1]
        atr = self.atr[-1]
        ema_f = self.ema_f[-1]
        ema_s = self.ema_s[-1]

        if np.isnan(vol_ma) or np.isnan(atr) or atr <= 0:
            return

        # Volume confirmation
        volume_spike = vol > (vol_ma * self.vol_multiplier)

        # Breakout conditions (close beyond level, not just wick)
        long_breakout = price > prev_resistance and volume_spike
        short_breakout = price < prev_support and volume_spike

        # Trend filter
        uptrend = ema_f > ema_s
        downtrend = ema_f < ema_s

        # Entry logic
        if not self.position:
            if long_breakout and uptrend:
                stop_price = price - (1.5 * atr)
                risk = price - stop_price
                if risk <= 0:
                    return
                risk_amount = self.equity * self.risk_pct
                size = int(round(risk_amount / risk))
                if size > 0:
                    print(f"🌙🚀 LONG BREAKOUT! Price={price:.2f} Res={prev_resistance:.2f} "
                          f"Vol={vol:.2f} VolMA={vol_ma:.2f} Size={size}")
                    self.buy(size=size, sl=stop_price, tp=price + (self.rr_ratio * risk))

            elif short_breakout and downtrend:
                stop_price = price + (1.5 * atr)
                risk = stop_price - price
                if risk <= 0:
                    return
                risk_amount = self.equity * self.risk_pct
                size = int(round(risk_amount / risk))
                if size > 0:
                    print(f"🌙🔻 SHORT BREAKOUT! Price={price:.2f} Sup={prev_support:.2f} "
                          f"Vol={vol:.2f} VolMA={vol_ma:.2f} Size={size}")
                    self.sell(size=size, sl=stop_price, tp=price - (self.rr_ratio * risk))


print("🌙✨ Running backtest... 🚀")
bt = Backtest(data, VolumetricRupture, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙 Moon Dev Backtest Complete! ✨🚀")