import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

print("🌙✨ VolumetricSurge Backtest Initializing... 🚀")

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
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

print(f"🌙 Data loaded: {len(data)} bars 🚀")


class VolumetricSurge(Strategy):
    # Strategy parameters
    vol_fast = 12
    vol_slow = 26
    donchian_period = 20
    atr_period = 14
    atr_stop_mult = 2.0
    pvo_threshold = 0.0
    trend_ema_period = 200
    risk_pct = 0.02

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        # Volume EMAs for PVO (talib — no backtesting.lib)
        self.vol_ema_fast = self.I(talib.EMA, volume, timeperiod=self.vol_fast, name='VolEMA_Fast')
        self.vol_ema_slow = self.I(talib.EMA, volume, timeperiod=self.vol_slow, name='VolEMA_Slow')

        # PVO = (fast - slow) / slow * 100  — computed cleanly with numpy-safe denom
        def _pvo():
            fast = np.asarray(self.vol_ema_fast)
            slow = np.asarray(self.vol_ema_slow)
            denom = np.where(slow == 0, 1, slow)
            return (fast - slow) / denom * 100.0

        self.pvo = self.I(_pvo, name='PVO')

        # Donchian channel (talib MAX/MIN — no backtesting.lib)
        self.donchian_high = self.I(talib.MAX, high, timeperiod=self.donchian_period, name='DonchianHigh')
        self.donchian_low = self.I(talib.MIN, low, timeperiod=self.donchian_period, name='DonchianLow')

        # ATR (talib — no backtesting.lib)
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period, name='ATR')

        # Trend EMA (talib — no backtesting.lib)
        self.trend_ema = self.I(talib.EMA, close, timeperiod=self.trend_ema_period, name='TrendEMA')

        # Trailing stop tracking
        self.trailing_stop = None
        self.entry_price = None

        print("🌙✨ Indicators initialized: PVO, Donchian, ATR, Trend EMA 🚀")

    def next(self):
        price = self.data.Close[-1]
        prev_high = self.donchian_high[-2]
        prev_low = self.donchian_low[-2]
        pvo = self.pvo[-1]
        pvo_prev = self.pvo[-2]
        atr = self.atr[-1]
        trend = self.trend_ema[-1]

        if (np.isnan(pvo) or np.isnan(pvo_prev) or np.isnan(atr)
                or np.isnan(trend) or np.isnan(prev_high) or np.isnan(prev_low)):
            return

        # ============ POSITION MANAGEMENT ============
        if self.position:
            if self.position.is_long:
                # Update trailing stop
                new_stop = price - self.atr_stop_mult * atr
                if self.trailing_stop is None or new_stop > self.trailing_stop:
                    self.trailing_stop = new_stop

                # Exit: price below trailing stop OR PVO crosses below zero
                if price < self.trailing_stop:
                    print(f"🌙💥 LONG EXIT @ {price:.2f} | Trailing stop hit 🚀")
                    self.position.close()
                    self.trailing_stop = None
                elif pvo < 0 and pvo_prev >= 0:
                    print(f"🌙💥 LONG EXIT @ {price:.2f} | PVO crossed below zero 🚀")
                    self.position.close()
                    self.trailing_stop = None

            elif self.position.is_short:
                # Update trailing stop
                new_stop = price + self.atr_stop_mult * atr
                if self.trailing_stop is None or new_stop < self.trailing_stop:
                    self.trailing_stop = new_stop

                # Exit: price above trailing stop OR PVO crosses above zero
                if price > self.trailing_stop:
                    print(f"🌙💥 SHORT EXIT @ {price:.2f} | Trailing stop hit 🚀")
                    self.position.close()
                    self.trailing_stop = None
                elif pvo > 0 and pvo_prev <= 0:
                    print(f"🌙💥 SHORT EXIT @ {price:.2f} | PVO crossed above zero 🚀")
                    self.position.close()
                    self.trailing_stop = None
            return

        # ============ ENTRY LOGIC ============
        # Long entry: close above prior Donchian high, PVO > threshold and rising, price above trend EMA
        long_breakout = price > prev_high
        long_vol_confirm = pvo > self.pvo_threshold and pvo > pvo_prev
        long_trend = price > trend

        # Short entry: close below prior Donchian low, PVO < threshold and falling, price below trend EMA
        short_breakout = price < prev_low
        short_vol_confirm = pvo < self.pvo_threshold and pvo < pvo_prev
        short_trend = price < trend

        if long_breakout and long_vol_confirm and long_trend:
            stop = price - self.atr_stop_mult * atr
            risk = price - stop
            if risk > 0:
                size = int(round(1_000_000 / price))
                if size > 0:
                    print(f"🌙🚀 LONG ENTRY @ {price:.2f} | PVO={pvo:.2f} | Stop={stop:.2f} | Size={size}")
                    self.buy(size=size)
                    self.trailing_stop = stop
                    self.entry_price = price

        elif short_breakout and short_vol_confirm and short_trend:
            stop = price + self.atr_stop_mult * atr
            risk = stop - price
            if risk > 0:
                size = int(round(1_000_000 / price))
                if size > 0:
                    print(f"🌙🚀 SHORT ENTRY @ {price:.2f} | PVO={pvo:.2f} | Stop={stop:.2f} | Size={size}")
                    self.sell(size=size)
                    self.trailing_stop = stop
                    self.entry_price = price


bt = Backtest(data, VolumetricSurge, cash=1_000_000, commission=0.001)

print("🌙✨ Running VolumetricSurge backtest... 🚀")
stats = bt.run()
print(stats)
print(stats._strategy)