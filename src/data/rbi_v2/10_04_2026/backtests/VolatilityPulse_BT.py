import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

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
data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

print("🌙✨ Moon Dev VolatilityPulse Backtest Loading... 🚀")
print(f"📊 Data shape: {data.shape}")
print(f"📈 Date range: {data.index[0]} to {data.index[-1]}")


class VolatilityPulse(Strategy):
    # Strategy parameters
    fast_ema_period = 20
    slow_ema_period = 50
    bb_period = 20
    bb_std = 2.0
    rsi_period = 14
    atr_period = 14
    bww_lookback = 3
    bww_ma_period = 20
    rsi_exit_threshold = 70
    risk_per_trade = 0.02

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low

        # EMAs
        self.ema_fast = self.I(talib.EMA, close, timeperiod=self.fast_ema_period, name='EMA20')
        self.ema_slow = self.I(talib.EMA, close, timeperiod=self.slow_ema_period, name='EMA50')

        # Bollinger Bands
        self.bb_upper = self.I(talib.BBANDS, close, timeperiod=self.bb_period,
                               nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0,
                               name='BB_Upper', which=0)
        self.bb_middle = self.I(talib.BBANDS, close, timeperiod=self.bb_period,
                                nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0,
                                name='BB_Middle', which=1)
        self.bb_lower = self.I(talib.BBANDS, close, timeperiod=self.bb_period,
                               nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0,
                               name='BB_Lower', which=2)

        # BWW = (Upper - Lower) / Middle
        self.bww = self.I(lambda u, l, m: (u - l) / m,
                          self.bb_upper, self.bb_lower, self.bb_middle,
                          name='BWW')

        # BWW moving average
        self.bww_ma = self.I(talib.SMA, self.bww, timeperiod=self.bww_ma_period, name='BWW_MA')

        # RSI
        self.rsi = self.I(talib.RSI, close, timeperiod=self.rsi_period, name='RSI')

        # ATR for stop loss
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period, name='ATR')

        # Swing low for stop loss (recent 10-bar low)
        self.swing_low = self.I(talib.MIN, low, timeperiod=10, name='SwingLow')

        print("🌙 VolatilityPulse indicators initialized! ✨")

    def next(self):
        price = self.data.Close[-1]

        # Skip if not enough data
        if len(self.data) < self.slow_ema_period + self.bww_ma_period + 5:
            return

        # ============ LONG ENTRY ============
        if not self.position:
            # EMA crossover: fast above slow, and previously below
            ema_cross_up = (self.ema_fast[-1] > self.ema_slow[-1] and
                            self.ema_fast[-2] <= self.ema_slow[-2])

            # BWW rising over last N bars
            bww_rising = all(self.bww[-i] > self.bww[-i - 1]
                             for i in range(1, self.bww_lookback + 1))

            # BWW above its moving average
            bww_above_ma = self.bww[-1] > self.bww_ma[-1]

            # RSI > 50 and rising
            rsi_bullish = self.rsi[-1] > 50 and self.rsi[-1] > self.rsi[-2]

            # Avoid overbought extremes
            not_overbought = self.rsi[-1] < 75

            if ema_cross_up and bww_rising and bww_above_ma and rsi_bullish and not_overbought:
                # Risk-based position sizing
                stop_price = min(self.swing_low[-1], self.ema_slow[-1] - 1.5 * self.atr[-1])
                risk_per_unit = price - stop_price

                if risk_per_unit > 0:
                    risk_amount = self.equity * self.risk_per_trade
                    position_size = int(round(risk_amount / risk_per_unit))
                    position_size = min(position_size, 1000000)

                    if position_size > 0:
                        self.buy(size=position_size, sl=stop_price)
                        print(f"🚀🌙 MOON DEV LONG ENTRY! Price: {price:.2f} | "
                              f"Size: {position_size} | SL: {stop_price:.2f} | "
                              f"RSI: {self.rsi[-1]:.2f} | BWW: {self.bww[-1]:.4f}")

        # ============ LONG EXIT ============
        else:
            # BWW contraction for M consecutive bars
            bww_contracting = all(self.bww[-i] < self.bww[-i - 1]
                                  for i in range(1, self.bww_lookback + 1))

            # BWW falls below its moving average
            bww_below_ma = self.bww[-1] < self.bww_ma[-1]

            # RSI bearish divergence: price higher high, RSI lower high
            lookback = 14
            bearish_divergence = False
            if len(self.data) > lookback + 2:
                recent_high_idx = int(np.argmax(self.data.High[-lookback:]))
                if recent_high_idx >= 2:
                    prev_high_idx = int(np.argmax(self.data.High[-lookback:-recent_high_idx - 1])) if recent_high_idx < lookback - 2 else 0
                    price_hh = self.data.High[-lookback + recent_high_idx] > self.data.High[-lookback + prev_high_idx]
                    rsi_lh = self.rsi[-lookback + recent_high_idx] < self.rsi[-lookback + prev_high_idx]
                    bearish_divergence = price_hh and rsi_lh

            # RSI overbought then crosses back down
            rsi_tp_exit = self.rsi[-2] > 80 and self.rsi[-1] < 70

            # Opposite EMA crossover
            ema_cross_down = (self.ema_fast[-1] < self.ema_slow[-1] and
                              self.ema_fast[-2] >= self.ema_slow[-2])

            if bww_contracting or bww_below_ma or bearish_divergence or rsi_tp_exit or ema_cross_down:
                self.position.close()
                reason = ("BWW Contraction" if bww_contracting else
                          "BWW Below MA" if bww_below_ma else
                          "RSI Divergence" if bearish_divergence else
                          "RSI Take Profit" if rsi_tp_exit else
                          "EMA Cross Down")
                print(f"🌙✨ MOON DEV EXIT! Reason: {reason} | "
                      f"Price: {price:.2f} | RSI: {self.rsi[-1]:.2f} | "
                      f"BWW: {self.bww[-1]:.4f}")


# Run backtest
print("🌙🚀 Starting Moon Dev VolatilityPulse Backtest...")
bt = Backtest(data, VolatilityPulse, cash=1000000, commission=0.002)
stats = bt.run()
print(stats)
print(stats._strategy)