import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Load and clean data
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
print("🌙✨ EchoDivergence data loaded successfully! 🚀")
print(data.head())


class EchoDivergence(Strategy):
    # Strategy parameters
    bb_period = 20
    bb_std = 2.0
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    rsi_period = 14
    atr_period = 14
    swing_lookback = 30
    atr_stop_mult = 1.5
    atr_be_mult = 1.0
    risk_pct = 0.02

    def init(self):
        print("🌙 Initializing EchoDivergence indicators... ✨")

        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        volume = pd.Series(self.data.Volume)

        # Bollinger Bands
        self.bb_upper, self.bb_mid, self.bb_lower = self.I(
            talib.BBANDS, close, timeperiod=self.bb_period,
            nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0
        )

        # MACD
        self.macd, self.macd_signal_line, self.macd_hist = self.I(
            talib.MACD, close,
            fastperiod=self.macd_fast,
            slowperiod=self.macd_slow,
            signalperiod=self.macd_signal
        )

        # RSI
        self.rsi = self.I(talib.RSI, close, timeperiod=self.rsi_period)

        # ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        # Swing highs and lows
        self.swing_high = self.I(talib.MAX, high, timeperiod=self.swing_lookback)
        self.swing_low = self.I(talib.MIN, low, timeperiod=self.swing_lookback)

        # Volume SMA for decreasing-volume filter
        self.vol_sma = self.I(talib.SMA, volume, timeperiod=self.swing_lookback)

        # State tracking
        self.entry_price_val = None
        self.stop_price = None
        self.breakeven_activated = False

        print("🌙✨ Indicators ready! Let's echo some divergences 🚀")

    def _bearish_macd_divergence(self, i):
        """Detect bearish MACD divergence: price higher high, MACD lower high."""
        if i < self.swing_lookback * 2:
            return False

        window = self.swing_lookback
        recent_high = np.array(self.data.High[i - window:i + 1])
        recent_macd = np.array(self.macd[i - window:i + 1])

        if len(recent_high) < 5:
            return False

        if np.any(np.isnan(recent_macd)):
            return False

        half = len(recent_high) // 2
        first_high_idx = int(np.argmax(recent_high[:half]))
        second_high_idx = int(np.argmax(recent_high[half:])) + half

        if recent_high[second_high_idx] > recent_high[first_high_idx] and recent_macd[second_high_idx] < recent_macd[first_high_idx]:
            return True
        return False

    def _bullish_rsi_divergence(self, i):
        """Detect bullish RSI divergence: price lower low, RSI higher low."""
        if i < self.swing_lookback * 2:
            return False

        window = self.swing_lookback
        recent_low = np.array(self.data.Low[i - window:i + 1])
        recent_rsi = np.array(self.rsi[i - window:i + 1])

        if np.any(np.isnan(recent_rsi)):
            return False

        half = len(recent_low) // 2
        first_low_idx = int(np.argmin(recent_low[:half]))
        second_low_idx = int(np.argmin(recent_low[half:])) + half

        if recent_low[second_low_idx] < recent_low[first_low_idx] and recent_rsi[second_low_idx] > recent_rsi[first_low_idx]:
            return True
        return False

    def next(self):
        i = len(self.data) - 1

        # Manage open short position
        if self.position:
            price = self.data.Close[-1]

            # Breakeven logic
            if not self.breakeven_activated and self.entry_price_val is not None:
                atr_val = self.atr[-1]
                if not np.isnan(atr_val) and price <= self.entry_price_val - (self.atr_be_mult * atr_val):
                    self.stop_price = self.entry_price_val
                    self.breakeven_activated = True
                    print(f"🌙✨ Breakeven activated at {self.stop_price:.2f} 🚀")

            # Check stop loss (short: price above stop)
            if self.stop_price is not None and price >= self.stop_price:
                self.position.close()
                print(f"🛑 Stop hit at {price:.2f} — exiting short 🌙")
                self.entry_price_val = None
                self.stop_price = None
                self.breakeven_activated = False
                return

            # Check RSI bullish divergence exit
            if self._bullish_rsi_divergence(i):
                self.position.close()
                print(f"🌙✨ Bullish RSI divergence detected — closing short at {price:.2f} 🚀")
                self.entry_price_val = None
                self.stop_price = None
                self.breakeven_activated = False
                return

            return

        # Entry logic
        if i < self.swing_lookback * 2 + 5:
            return

        price = self.data.Close[-1]
        bb_lower = self.bb_lower[-1]
        atr_val = self.atr[-1]
        rsi_val = self.rsi[-1]
        vol_sma_val = self.vol_sma[-1]

        if np.isnan(bb_lower) or np.isnan(atr_val) or np.isnan(rsi_val) or np.isnan(vol_sma_val):
            return

        # Condition 1: MACD bearish divergence
        macd_div = self._bearish_macd_divergence(i)

        # Condition 2: Close below lower Bollinger Band
        bb_breakdown = price < bb_lower

        # Condition 3: Decreasing volume vs prior rally
        vol_decreasing = self.data.Volume[-1] < vol_sma_val

        # Condition 4: RSI filter
        rsi_filter = rsi_val < 50

        if macd_div and bb_breakdown and vol_decreasing and rsi_filter:
            # Position sizing: risk 2% of equity, expressed as a fraction of equity
            equity = self.equity
            risk_amount = equity * self.risk_pct
            stop_distance = self.atr_stop_mult * atr_val

            if stop_distance <= 0:
                return

            # Convert risk-based unit size to fraction of equity (backtesting.py requires 0<size<1)
            units = risk_amount / stop_distance
            position_size = (units * price) / equity

            # Cap so we don't exceed equity
            if position_size >= 1:
                position_size = 0.99
            if position_size <= 0:
                return

            # Ensure position_size is a valid fraction
            position_size = float(position_size)

            self.sell(size=position_size, sl=self.stop_price)
            self.entry_price_val = price
            self.stop_price = price + stop_distance
            self.breakeven_activated = False

            print(f"🌙🔻 SHORT ENTRY @ {price:.2f} | Size: {position_size:.4f} | Stop: {self.stop_price:.2f} | ATR: {atr_val:.2f} 🚀")


# Run backtest
bt = Backtest(data, EchoDivergence, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)