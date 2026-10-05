import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Load data
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={'datetime': 'Datetime', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
data = data.set_index(pd.to_datetime(data['Datetime']))
data = data[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)

print("🌙 Moon Dev Backtest Engine warming up... 🚀")
print(f"📊 Loaded {len(data)} bars of data")


class CompressionSurge(Strategy):
    bb_period = 20
    bb_std = 2.0
    bbw_lookback = 20
    vol_mult = 1.5
    vol_period = 20
    atr_period = 14
    atr_mult = 2.0
    risk_pct = 0.01
    time_exit_bars = 20

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

        # Bollinger Band Width
        def bbw(upper, middle, lower):
            return (upper - lower) / middle
        self.bbw = self.I(bbw, self.bb_upper, self.bb_middle, self.bb_lower)

        # BBW lowest over lookback
        self.bbw_min = self.I(talib.MIN, self.bbw, timeperiod=self.bbw_lookback)

        # Volume SMA
        self.vol_sma = self.I(talib.SMA, volume, timeperiod=self.vol_period)

        # ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        # Trackers
        self.highest_close = 0.0
        self.entry_bar = 0
        self.initial_stop = 0.0

        print("✨ Indicators initialized: BB, BBW, Volume SMA, ATR 🌙")

    def next(self):
        price = self.data.Close[-1]

        if not self.position:
            # Check entry conditions
            squeeze = self.bbw[-1] <= self.bbw_min[-1]
            vol_spike = self.data.Volume[-1] >= self.vol_mult * self.vol_sma[-1]
            breakout = self.data.Close[-1] > self.bb_upper[-1]

            if squeeze and vol_spike and breakout:
                atr_val = self.atr[-1]
                if atr_val <= 0 or np.isnan(atr_val):
                    return

                # Risk-based position sizing (fraction of equity for backtesting)
                risk_amount = self.equity * self.risk_pct
                stop_distance = self.atr_mult * atr_val
                if stop_distance <= 0:
                    return
                units = risk_amount / stop_distance
                # Convert to fraction of equity so sizing is valid for backtesting
                size_fraction = (units * price) / self.equity
                size_fraction = max(0.001, min(0.99, size_fraction))

                self.initial_stop = price - stop_distance
                self.highest_close = price
                self.entry_bar = len(self.data)

                print(f"🚀 MOON DEV ENTRY: CompressionSurge LONG at {price:.2f} | "
                      f"BBW={self.bbw[-1]:.4f} | Vol={self.data.Volume[-1]:.0f} "
                      f"(x{self.data.Volume[-1]/self.vol_sma[-1]:.2f}) | "
                      f"ATR={atr_val:.2f} | Size={size_fraction:.4f} 🌙")

                self.buy(size=size_fraction)

        else:
            # Update highest close
            if self.data.Close[-1] > self.highest_close:
                self.highest_close = self.data.Close[-1]

            # Trailing stop
            trailing_stop = self.highest_close - self.atr_mult * self.atr[-1]
            hard_stop = self.initial_stop
            stop_level = max(trailing_stop, hard_stop)

            # Mean reversion exit
            below_middle = self.data.Close[-1] < self.bb_middle[-1]
            # Time exit
            bars_held = len(self.data) - self.entry_bar
            time_exit = bars_held >= self.time_exit_bars

            if self.data.Close[-1] < stop_level:
                print(f"🛑 TRAILING STOP HIT at {self.data.Close[-1]:.2f} "
                      f"(stop={stop_level:.2f}) | HighestClose={self.highest_close:.2f} 🌙")
                self.position.close()
            elif below_middle:
                print(f"📉 MEAN REVERSION EXIT: Close {self.data.Close[-1]:.2f} "
                      f"below middle band {self.bb_middle[-1]:.2f} 🌙")
                self.position.close()
            elif time_exit:
                print(f"⏰ TIME EXIT after {bars_held} bars at {self.data.Close[-1]:.2f} 🌙")
                self.position.close()


print("🌙 Starting CompressionSurge backtest... 🚀")
bt = Backtest(data, CompressionSurge, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)
print("✨ Moon Dev backtest complete! 🌙")