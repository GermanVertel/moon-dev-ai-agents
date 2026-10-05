import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

print("🌙 Moon Dev's FiboBand Reversal Backtest loading... ✨🚀")

# Load data
data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper case mapping
data = data.rename(columns={
    'datetime': 'Datetime',
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

data['Datetime'] = pd.to_datetime(data['Datetime'])
data = data.set_index('Datetime')

print(f"🌙 Data loaded: {len(data)} bars from {data.index[0]} to {data.index[-1]} ✨")


class FiboBandReversal(Strategy):
    # Bollinger Bands
    bb_period = 20
    bb_std = 2.0
    # RSI
    rsi_period = 14
    rsi_overbought = 70
    # ATR
    atr_period = 14
    atr_buffer = 0.5
    # Swing detection
    swing_lookback = 10
    # Risk
    risk_pct = 0.02
    rr_target = 2.0
    max_hold_bars = 15
    # ADX filter to avoid strong trends
    adx_period = 14
    adx_threshold = 30

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low

        # Bollinger Bands
        self.bb_upper, self.bb_mid, self.bb_lower = self.I(
            talib.BBANDS, close, timeperiod=self.bb_period,
            nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0
        )

        # RSI
        self.rsi = self.I(talib.RSI, close, timeperiod=self.rsi_period)

        # ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        # ADX
        self.adx = self.I(talib.ADX, high, low, close, timeperiod=self.adx_period)

        # Swing high/low
        self.swing_high = self.I(talib.MAX, high, timeperiod=self.swing_lookback)
        self.swing_low = self.I(talib.MIN, low, timeperiod=self.swing_lookback)

        # State tracking
        self.touched_upper = False
        self.fib_high = None
        self.entry_bar = None
        self.stop_price = None
        self.tp_price = None

        print("🌙✨ Indicators initialized! Ready to hunt reversals 🚀")

    def next(self):
        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]

        upper = self.bb_upper[-1]
        mid = self.bb_mid[-1]
        lower = self.bb_lower[-1]
        rsi = self.rsi[-1]
        atr = self.atr[-1]
        adx = self.adx[-1]

        if np.isnan(upper) or np.isnan(rsi) or np.isnan(atr) or np.isnan(adx):
            return

        # ========== MANAGE OPEN POSITION ==========
        if self.position:
            bars_held = len(self.data) - self.entry_bar

            # Stop loss: close above fib high
            if self.fib_high is not None and price > self.fib_high:
                print(f"🛑 Stop hit! Price {price:.2f} > Fib high {self.fib_high:.2f} 🌙")
                self.position.close()
                self.touched_upper = False
                self.fib_high = None
                return

            # Take profit: touch lower band
            if price <= lower:
                print(f"🎯 TP hit! Price {price:.2f} touched lower band {lower:.2f} ✨")
                self.position.close()
                self.touched_upper = False
                self.fib_high = None
                return

            # Max holding period
            if bars_held >= self.max_hold_bars:
                print(f"⏰ Max hold {bars_held} bars reached, closing short 🌙")
                self.position.close()
                self.touched_upper = False
                self.fib_high = None
                return

            return

        # ========== ENTRY LOGIC ==========
        # Avoid strong trending regimes
        if adx > self.adx_threshold:
            return

        # Track upper band touch
        if high >= upper:
            self.touched_upper = True
            print(f"🌙 Price touched upper BB at {high:.2f} (upper={upper:.2f}) ✨")

        # After touch, look for fib retracement high confirmation
        if self.touched_upper and self.fib_high is None:
            # Confirm swing high formed after touch
            if self.swing_high[-1] < high and self.swing_high[-1] > 0:
                self.fib_high = self.swing_high[-1]
                print(f"📐 Fib retracement high confirmed at {self.fib_high:.2f} 🚀")

        # Entry trigger: touched upper, fib high set, RSI overbought, bearish close
        if (self.touched_upper and self.fib_high is not None
                and rsi > self.rsi_overbought
                and price < self.data.Close[-2]
                and price < self.fib_high):

            # Risk management position sizing
            stop_price = self.fib_high + self.atr_buffer * atr
            risk_per_unit = stop_price - price

            if risk_per_unit <= 0:
                return

            # Use fraction-based sizing (percentage of equity)
            # Risk-based fraction of equity
            position_fraction = self.risk_pct

            # Clamp to valid fraction range (0, 1)
            position_fraction = min(max(position_fraction, 0.01), 0.95)

            self.stop_price = stop_price
            self.tp_price = lower
            self.entry_bar = len(self.data)

            print(f"🚀 SHORT ENTRY! Price={price:.2f} FibHigh={self.fib_high:.2f} "
                  f"Stop={stop_price:.2f} Size={position_fraction:.4f} RSI={rsi:.1f} 🌙")

            self.sell(size=position_fraction, sl=stop_price)


# Run backtest
bt = Backtest(
    data,
    FiboBandReversal,
    cash=1_000_000,
    commission=0.002,
    exclusive_orders=True
)

print("🌙✨ Running FiboBand Reversal backtest... 🚀")
stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙 Moon Dev backtest complete! ✨🚀")