import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Load and prepare data
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
data['Datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('Datetime')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

print("🌙✨ Moon Dev ReversalConfirmation Backtest Initializing... 🚀")
print(f"📊 Data loaded: {len(data)} bars")


class ReversalConfirmation(Strategy):
    rsi_period = 14
    atr_period = 14
    swing_window = 20
    rsi_oversold = 30
    rsi_overbought = 70
    risk_pct = 0.02
    rr_target = 2.0

    def init(self):
        print("🌙 Initializing indicators...")
        # ✅ RSI via talib
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=self.rsi_period, name="RSI")
        # ✅ ATR via talib
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period, name="ATR")
        # ✅ Rolling swing high/low via talib MAX/MIN
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=self.swing_window, name="SwingHigh")
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=self.swing_window, name="SwingLow")
        # ✅ SMA via talib
        self.sma200 = self.I(talib.SMA, self.data.Close, timeperiod=200, name="SMA200")
        print("✅ Indicators ready! 🌙✨")

    def next(self):
        # Need enough history for indicators and lookback
        if len(self.data) < max(self.swing_window + 2, 202):
            return

        price = self.data.Close[-1]
        rsi = self.rsi[-1]
        rsi_prev = self.rsi[-2]
        atr = self.atr[-1]
        neckline = self.swing_high[-1]
        head = self.swing_low[-1]

        if np.isnan(rsi) or np.isnan(atr) or np.isnan(neckline) or np.isnan(head):
            return

        # --- ENTRY LOGIC ---
        if not self.position:
            # Breakout above neckline + RSI reversal confirmation
            breakout = price > neckline and self.data.Close[-2] <= self.swing_high[-2]
            # RSI crossover detection
            rsi_bullish = (rsi > 50 and rsi_prev <= 50) or (rsi > self.rsi_oversold and rsi_prev <= self.rsi_oversold)
            # Head must be below neckline (valid IHS shape)
            valid_pattern = head < neckline

            if breakout and rsi_bullish and valid_pattern:
                stop_loss = price - (2 * atr)
                risk_per_unit = price - stop_loss
                if risk_per_unit <= 0:
                    return
                account = self.equity
                risk_amount = account * self.risk_pct
                size = int(round(risk_amount / risk_per_unit))
                if size < 1:
                    size = 1
                # Cap size to avoid insane leverage
                max_size = int(account / price)
                if size > max_size:
                    size = max_size
                if size < 1:
                    return
                target = neckline + (neckline - head)  # measured move
                print(f"🌙🚀 MOON DEV ENTRY! Price={price:.2f} Neckline={neckline:.2f} Head={head:.2f} RSI={rsi:.2f} ATR={atr:.2f} Size={size} SL={stop_loss:.2f} TP={target:.2f}")
                self.buy(size=size, sl=stop_loss, tp=target)

        # --- EXIT LOGIC ---
        else:
            # RSI bearish reversal exit
            rsi_bearish = (rsi < 50 and rsi_prev >= 50) or (rsi < self.rsi_overbought and rsi_prev >= self.rsi_overbought)
            if rsi_bearish:
                print(f"🌙✨ MOON DEV EXIT — RSI bearish reversal at price={price:.2f} RSI={rsi:.2f}")
                self.position.close()


bt = Backtest(data, ReversalConfirmation, cash=1_000_000, commission=0.002)
stats = bt.run()
print(stats)
print(stats._strategy)