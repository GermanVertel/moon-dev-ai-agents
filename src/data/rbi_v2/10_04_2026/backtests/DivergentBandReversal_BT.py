import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

print("🌙✨ Moon Dev's DivergentBandReversal Backtest Loading... 🚀")

# Load data
data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Map columns
data = data.rename(columns={
    'datetime': 'datetime',
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

print(f"🌙 Data loaded: {len(data)} rows ✨")


class DivergentBandReversal(Strategy):
    # Long params
    rsi_period = 5
    low_lookback = 3
    # Short params
    ma_period = 10
    bb_period = 20
    bb_std = 2.0
    slope_lookback = 3
    # Risk
    atr_period = 14
    atr_mult = 1.5
    risk_pct = 0.02

    def init(self):
        print("🌙 Initializing indicators... ✨")
        close = self.data.Close
        low = self.data.Low
        high = self.data.High

        # Long: 5-period RSI
        self.rsi = self.I(talib.RSI, close, timeperiod=self.rsi_period)

        # Price action lows over past 3 periods
        self.low_min = self.I(talib.MIN, low, timeperiod=self.low_lookback)

        # Short: 10-period SMA
        self.sma10 = self.I(talib.SMA, close, timeperiod=self.ma_period)

        # Bollinger Bands 20-period
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS, close, timeperiod=self.bb_period,
            nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0
        )

        # ATR for stops
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        # EMA 50 trend filter
        self.ema50 = self.I(talib.EMA, close, timeperiod=50)

        print("🚀 Indicators ready!")

    def next(self):
        price = self.data.Close[-1]

        # Skip if not enough data
        if len(self.data) < 55:
            return

        # ---- EXITS ----
        if self.position:
            if self.position.is_long:
                # Long exit: RSI > 50 or close above SMA10
                if self.rsi[-1] > 50 or price > self.sma10[-1]:
                    print(f"🌙 Long exit at {price:.2f} | RSI={self.rsi[-1]:.2f} ✨")
                    self.position.close()
            elif self.position.is_short:
                # Short exit: SMA10 crosses below BB middle, or BB middle slope turns up
                bb_slope = self.bb_middle[-1] - self.bb_middle[-1 - self.slope_lookback]
                if self.sma10[-1] < self.bb_middle[-1] or bb_slope >= 0:
                    print(f"🌙 Short exit at {price:.2f} | SMA10={self.sma10[-1]:.2f} BBmid={self.bb_middle[-1]:.2f} 🚀")
                    self.position.close()

        # ---- ENTRIES ----
        if not self.position:
            # LONG: RSI below price action low (divergence/oversold)
            long_signal = self.rsi[-1] < self.low_min[-1]

            # SHORT: SMA10 above declining BB middle
            bb_slope = self.bb_middle[-1] - self.bb_middle[-1 - self.slope_lookback]
            short_signal = (self.sma10[-1] > self.bb_middle[-1]) and (bb_slope < 0)

            # Asymmetric: avoid both, prioritize short momentum if both fire
            if long_signal and not short_signal:
                atr_val = self.atr[-1]
                if atr_val > 0:
                    sl_price = price - self.atr_mult * atr_val
                    risk_per_unit = price - sl_price
                    # Volatility-adjusted sizing
                    equity = self.equity
                    risk_amount = equity * self.risk_pct
                    size = int(round(risk_amount / risk_per_unit)) if risk_per_unit > 0 else 0
                    if size > 0:
                        print(f"🌙🚀 LONG ENTRY @ {price:.2f} | RSI={self.rsi[-1]:.2f} < LowMin={self.low_min[-1]:.2f} | SL={sl_price:.2f} size={size}")
                        self.buy(size=size, sl=sl_price)
            elif short_signal and not long_signal:
                atr_val = self.atr[-1]
                if atr_val > 0:
                    sl_price = price + self.atr_mult * atr_val
                    risk_per_unit = sl_price - price
                    equity = self.equity
                    risk_amount = equity * self.risk_pct
                    size = int(round(risk_amount / risk_per_unit)) if risk_per_unit > 0 else 0
                    if size > 0:
                        print(f"🌙🔻 SHORT ENTRY @ {price:.2f} | SMA10={self.sma10[-1]:.2f} > BBmid={self.bb_middle[-1]:.2f} declining | SL={sl_price:.2f} size={size}")
                        self.sell(size=size, sl=sl_price)


bt = Backtest(data, DivergentBandReversal, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)