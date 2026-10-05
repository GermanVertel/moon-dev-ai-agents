import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Load and clean data
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
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
data = data.dropna()

# Ensure numeric dtypes (talib requires double arrays) 🌙
for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    data[col] = pd.to_numeric(data[col], errors='coerce').astype('float64')
data = data.dropna()

print("🌙✨ Moon Dev SqueezeConfluence Backtest Loading... 🚀")
print(f"📊 Data shape: {data.shape}")
print(f"📈 Date range: {data.index[0]} to {data.index[-1]}")


class SqueezeConfluence(Strategy):
    # Bollinger Bands
    bb_period = 20
    bb_std = 2.0
    # Keltner Channels
    kc_period = 20
    kc_atr_mult = 1.5
    # Volume
    vol_period = 20
    vol_mult = 1.5
    # Squeeze duration
    min_squeeze_bars = 5
    # Risk
    atr_stop_mult = 1.5
    risk_pct = 0.02
    # Time stop
    time_stop_bars = 10

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

        # Keltner Channels
        self.kc_mid = self.I(talib.EMA, close, timeperiod=self.kc_period)
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.kc_period)
        self.kc_upper = self.I(lambda m, a: m + self.kc_atr_mult * a, self.kc_mid, self.atr)
        self.kc_lower = self.I(lambda m, a: m - self.kc_atr_mult * a, self.kc_mid, self.atr)

        # Volume MA
        self.vol_ma = self.I(talib.SMA, volume, timeperiod=self.vol_period)

        # MACD
        self.macd, self.macd_signal, self.macd_hist = self.I(
            talib.MACD, close, fastperiod=12, slowperiod=26, signalperiod=9
        )

        # EMA trend filter
        self.ema20 = self.I(talib.EMA, close, timeperiod=20)

        print("🌙 Indicators initialized successfully! ✨")

    def next(self):
        if len(self.data) < max(self.bb_period, self.kc_period, self.vol_period) + self.min_squeeze_bars + 1:
            return

        price = self.data.Close[-1]
        vol = self.data.Volume[-1]
        vol_ma = self.vol_ma[-1]

        # Squeeze condition: BB inside KC
        squeeze = (self.bb_upper[-1] < self.kc_upper[-1]) and (self.bb_lower[-1] > self.kc_lower[-1])

        # Check squeeze persisted for min bars
        squeeze_count = 0
        for i in range(1, self.min_squeeze_bars + 1):
            idx = -i
            if (self.bb_upper[idx] < self.kc_upper[idx]) and (self.bb_lower[idx] > self.kc_lower[idx]):
                squeeze_count += 1

        squeeze_persisted = squeeze_count >= self.min_squeeze_bars

        # Volume spike
        vol_spike = vol > self.vol_mult * vol_ma if vol_ma and vol_ma > 0 else False

        # Breakout triggers
        long_breakout = price > self.bb_upper[-1] and price > self.kc_upper[-1]
        short_breakout = price < self.bb_lower[-1] and price < self.kc_lower[-1]

        # Momentum filter
        macd_bull = self.macd_hist[-1] > 0 and self.macd_hist[-1] > self.macd_hist[-2]
        macd_bear = self.macd_hist[-1] < 0 and self.macd_hist[-1] < self.macd_hist[-2]

        # Trend bias
        above_ema = price > self.ema20[-1]
        below_ema = price < self.ema20[-1]

        atr_val = self.atr[-1]

        # === EXIT LOGIC ===
        if self.position:
            entry_price = self.trades[-1].entry_price if self.trades else price
            if self.position.is_long:
                # Exit if price re-enters either band
                if price < self.bb_upper[-1] or price < self.kc_upper[-1]:
                    print(f"🌙 EXIT LONG: Price re-entered band at {price:.2f} 🚀")
                    self.position.close()
                    return
                # Trailing stop at mid BB
                if price < self.bb_mid[-1] and self.bb_mid[-1] > entry_price:
                    print(f"🌙 TRAILING EXIT LONG at {price:.2f} ✨")
                    self.position.close()
                    return

            elif self.position.is_short:
                if price > self.bb_lower[-1] or price > self.kc_lower[-1]:
                    print(f"🌙 EXIT SHORT: Price re-entered band at {price:.2f} 🚀")
                    self.position.close()
                    return
                if price > self.bb_mid[-1] and self.bb_mid[-1] < entry_price:
                    print(f"🌙 TRAILING EXIT SHORT at {price:.2f} ✨")
                    self.position.close()
                    return
            return

        # === ENTRY LOGIC ===
        if squeeze_persisted and vol_spike and atr_val > 0:
            # Position sizing based on risk
            risk_amount = self.equity * self.risk_pct
            stop_distance = self.atr_stop_mult * atr_val
            if stop_distance <= 0:
                return
            position_size = int(round(risk_amount / stop_distance))
            if position_size <= 0:
                position_size = 1

            if long_breakout and macd_bull and above_ema:
                sl = price - stop_distance
                tp = price + 2 * stop_distance
                print(f"🚀🌙 LONG BREAKOUT! Price={price:.2f} Squeeze={squeeze_count}bars VolSpike={vol/vol_ma:.2f}x Size={position_size}")
                self.buy(size=position_size, sl=sl, tp=tp)

            elif short_breakout and macd_bear and below_ema:
                sl = price + stop_distance
                tp = price - 2 * stop_distance
                print(f"🚀🌙 SHORT BREAKOUT! Price={price:.2f} Squeeze={squeeze_count}bars VolSpike={vol/vol_ma:.2f}x Size={position_size}")
                self.sell(size=position_size, sl=sl, tp=tp)


bt = Backtest(data, SqueezeConfluence, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)