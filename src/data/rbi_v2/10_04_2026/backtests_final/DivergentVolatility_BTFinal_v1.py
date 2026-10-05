import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

print("🌙✨ DivergentVolatility Strategy Loading... 🚀")

# Load data
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
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
print(f"🌙 Data loaded: {len(data)} rows ✨")


class DivergentVolatility(Strategy):
    rsi_period = 14
    rsi_overbought = 70
    rsi_exit = 30
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    atr_period = 14
    atr_sma_period = 20
    atr_multiplier = 1.5
    swing_lookback = 20
    ema_period = 9
    risk_pct = 0.02
    rr_target = 2.0

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low

        # RSI
        self.rsi = self.I(talib.RSI, close, timeperiod=self.rsi_period, name='RSI')

        # MACD
        macd_result = self.I(
            talib.MACD, close,
            fastperiod=self.macd_fast,
            slowperiod=self.macd_slow,
            signalperiod=self.macd_signal,
            name='MACD'
        )
        self.macd = macd_result[0]
        self.macd_signal_line = macd_result[1]
        self.macd_hist = macd_result[2]

        # ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period, name='ATR')
        self.atr_sma = self.I(talib.SMA, self.atr, timeperiod=self.atr_sma_period, name='ATR_SMA')

        # EMA
        self.ema = self.I(talib.EMA, close, timeperiod=self.ema_period, name='EMA')

        # Swing highs/lows
        self.swing_high = self.I(talib.MAX, high, timeperiod=self.swing_lookback, name='SwingHigh')
        self.swing_low = self.I(talib.MIN, low, timeperiod=self.swing_lookback, name='SwingLow')

        print("🌙✨ Indicators initialized! 🚀")

    def next(self):
        price = self.data.Close[-1]

        # Skip if indicators not ready
        if len(self.data) < max(self.swing_lookback, self.atr_sma_period) + 5:
            return

        # ---------- EXIT LOGIC ----------
        if self.position:
            atr_val = self.atr[-1]
            atr_sma_val = self.atr_sma[-1]

            # Primary exit: ATR spike
            if atr_sma_val > 0 and atr_val > self.atr_multiplier * atr_sma_val:
                print(f"🌙💥 ATR Volatility Exit! ATR={atr_val:.2f} > {self.atr_multiplier}×SMA={atr_sma_val:.2f} @ {price:.2f}")
                self.position.close()
                return

            # Secondary exit: RSI drops below 30
            if self.rsi[-1] < self.rsi_exit:
                print(f"🌙📉 RSI Exit! RSI={self.rsi[-1]:.2f} < {self.rsi_exit} @ {price:.2f}")
                self.position.close()
                return

        # ---------- ENTRY LOGIC ----------
        if not self.position:
            rsi_val = self.rsi[-1]
            hist_now = self.macd_hist[-1]
            hist_prev = self.macd_hist[-2]
            hist_prev2 = self.macd_hist[-3]

            # RSI overbought
            rsi_ob = rsi_val > self.rsi_overbought

            # Bearish divergence: price higher high, MACD histogram lower high
            price_hh = self.data.High[-1] >= self.swing_high[-2]
            macd_lh = hist_now < hist_prev and hist_prev < hist_prev2

            # Optional confirmation: bearish close or below EMA
            bearish_close = self.data.Close[-1] < self.data.Open[-1]
            below_ema = self.data.Close[-1] < self.ema[-1]

            if rsi_ob and price_hh and macd_lh and (bearish_close or below_ema):
                atr_val = self.atr[-1]
                if atr_val <= 0:
                    return

                stop_price = self.data.High[-1] + 1.0 * atr_val
                risk = stop_price - price
                if risk <= 0:
                    return

                # Position sizing: risk 2% of equity -> fraction of equity
                equity = self.equity
                risk_amount = equity * self.risk_pct
                position_size = risk_amount / risk

                # Convert to fraction of equity for the backtesting framework
                size_fraction = position_size / equity
                if size_fraction <= 0:
                    return
                if size_fraction > 0.99:
                    size_fraction = 0.99

                print(f"🌙🚀 SHORT SIGNAL! RSI={rsi_val:.2f} | HH+MACD Divergence | Price={price:.2f} | Size={size_fraction:.4f}")

                self.sell(size=size_fraction, sl=stop_price, tp=price - self.rr_target * risk)


print("🌙✨ Running Backtest... 🚀")
bt = Backtest(data, DivergentVolatility, cash=1_000_000, commission=0.002)
stats = bt.run()
print(stats)
print(stats._strategy)