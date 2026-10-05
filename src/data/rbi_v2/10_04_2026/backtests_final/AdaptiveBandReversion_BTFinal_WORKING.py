import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# 🌙 Moon Dev's AdaptiveBandReversion Strategy ✨

class AdaptiveBandReversion(Strategy):
    # Strategy parameters
    rsi_period = 14
    bb_length = 20
    bb_std = 2.0
    ema_length = 200
    adx_period = 14
    adx_threshold = 20
    atr_period = 14
    atr_mult = 1.5
    risk_pct = 0.01  # 1% risk per trade
    size = 1_000_000

    def init(self):
        print("🌙 Moon Dev initializing AdaptiveBandReversion...")
        close = self.data.Close
        high = self.data.High
        low = self.data.Low

        # RSI
        self.rsi = self.I(talib.RSI, close, timeperiod=self.rsi_period, name='RSI')
        # Bollinger Bands on RSI
        self.rsi_sma = self.I(talib.SMA, self.rsi, timeperiod=self.bb_length, name='RSI_SMA')
        self.rsi_std = self.I(talib.STDDEV, self.rsi, timeperiod=self.bb_length, nbdev=1, name='RSI_STD')
        self.rsi_upper = self.I(lambda: self.rsi_sma + self.bb_std * self.rsi_std, name='RSI_Upper')
        self.rsi_lower = self.I(lambda: self.rsi_sma - self.bb_std * self.rsi_std, name='RSI_Lower')

        # Trend filter
        self.ema = self.I(talib.EMA, close, timeperiod=self.ema_length, name='EMA')

        # ADX
        self.adx = self.I(talib.ADX, high, low, close, timeperiod=self.adx_period, name='ADX')

        # ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period, name='ATR')

        print("✨ Moon Dev indicators ready!")

    def next(self):
        price = self.data.Close[-1]
        rsi_now = self.rsi[-1]
        rsi_prev = self.rsi[-2]
        upper_now = self.rsi_upper[-1]
        upper_prev = self.rsi_upper[-2]
        lower_now = self.rsi_lower[-1]
        lower_prev = self.rsi_lower[-2]
        mid_now = self.rsi_sma[-1]
        mid_prev = self.rsi_sma[-2]
        ema_now = self.ema[-1]
        adx_now = self.adx[-1]
        atr_now = self.atr[-1]

        if np.isnan([rsi_now, rsi_prev, upper_now, upper_prev, lower_now, lower_prev, mid_now, mid_prev, ema_now, adx_now, atr_now]).any():
            return

        # ---- Manage open position ----
        if self.position:
            if self.position.is_long:
                # Exit if RSI crosses above middle band
                if rsi_prev < mid_prev and rsi_now >= mid_now:
                    print(f"🌙 Moon Dev LONG exit — RSI reverted to midline @ {price:.2f}")
                    self.position.close()
                # Trend invalidation
                elif price < ema_now:
                    print(f"🌙 Moon Dev LONG exit — trend invalidation @ {price:.2f}")
                    self.position.close()
            elif self.position.is_short:
                if rsi_prev > mid_prev and rsi_now <= mid_now:
                    print(f"🌙 Moon Dev SHORT exit — RSI reverted to midline @ {price:.2f}")
                    self.position.close()
                elif price > ema_now:
                    print(f"🌙 Moon Dev SHORT exit — trend invalidation @ {price:.2f}")
                    self.position.close()
            return

        # ---- Entry logic ----
        if adx_now < self.adx_threshold:
            return

        # Long entry: uptrend + RSI crosses up through lower band
        long_signal = (
            price > ema_now and
            rsi_prev < lower_prev and
            rsi_now >= lower_now
        )
        # Short entry: downtrend + RSI crosses down through upper band
        short_signal = (
            price < ema_now and
            rsi_prev > upper_prev and
            rsi_now <= upper_now
        )

        if long_signal:
            stop_price = price - self.atr_mult * atr_now
            risk_per_unit = price - stop_price
            if risk_per_unit <= 0:
                return
            size = int(round((self.equity * self.risk_pct) / risk_per_unit))
            if size <= 0:
                return
            print(f"🚀 Moon Dev LONG entry @ {price:.2f} | RSI {rsi_now:.2f} crossed lower band {lower_now:.2f} | SL {stop_price:.2f} | size {size}")
            self.buy(size=size)
            self.trade = {'sl': stop_price}

        elif short_signal:
            stop_price = price + self.atr_mult * atr_now
            risk_per_unit = stop_price - price
            if risk_per_unit <= 0:
                return
            size = int(round((self.equity * self.risk_pct) / risk_per_unit))
            if size <= 0:
                return
            print(f"🚀 Moon Dev SHORT entry @ {price:.2f} | RSI {rsi_now:.2f} crossed upper band {upper_now:.2f} | SL {stop_price:.2f} | size {size}")
            self.sell(size=size)
            self.trade = {'sl': stop_price}


# ---- Data loading ----
data_path = '/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv'
print("🌙 Loading Moon Dev data...")
data = pd.read_csv(data_path)

# Clean columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Map to backtesting.py required columns
data = data.rename(columns={
    'datetime': 'Date',
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

# Ensure datetime index
if 'Date' in data.columns:
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.set_index('Date')

data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
print(f"✨ Moon Dev data ready: {len(data)} rows")

# ---- Run backtest ----
bt = Backtest(data, AdaptiveBandReversion, cash=1_000_000, commission=0.001, exclusive_orders=True)
stats = bt.run()
print(stats)
print(stats._strategy)