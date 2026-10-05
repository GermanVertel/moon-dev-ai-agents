import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# ============================================================
# 🌙 Moon Dev's CompressionPulse Backtest 🌙
# ============================================================

DATA_PATH = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"

print("🌙 Moon Dev: Loading data from the lunar data vault...")
data = pd.read_csv(DATA_PATH)

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Ensure proper column mapping
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

print(f"🚀 Moon Dev: Data loaded with {len(data)} bars. Ready for liftoff!")


def _bbands_upper(close, timeperiod, nbdevup, nbdevdn, matype):
    """Wrapper to extract upper Bollinger Band."""
    u, m, l = talib.BBANDS(close, timeperiod=timeperiod, nbdevup=nbdevup, nbdevdn=nbdevdn, matype=matype)
    return u


def _bbands_middle(close, timeperiod, nbdevup, nbdevdn, matype):
    """Wrapper to extract middle Bollinger Band."""
    u, m, l = talib.BBANDS(close, timeperiod=timeperiod, nbdevup=nbdevup, nbdevdn=nbdevdn, matype=matype)
    return m


def _bbands_lower(close, timeperiod, nbdevup, nbdevdn, matype):
    """Wrapper to extract lower Bollinger Band."""
    u, m, l = talib.BBANDS(close, timeperiod=timeperiod, nbdevup=nbdevup, nbdevdn=nbdevdn, matype=matype)
    return l


class CompressionPulse(Strategy):
    """
    🌙 CompressionPulse Strategy 🌙
    Volatility squeeze breakout with RSI momentum filter.
    Long-only, ATR-based symmetric exits (2x ATR).
    """

    # Strategy parameters
    bb_period = 20
    bb_std = 2.0
    rsi_period = 14
    atr_period = 14
    squeeze_lookback = 30
    atr_mult = 2.0
    risk_pct = 0.01  # risk 1% of equity per trade
    cooldown_bars = 5
    use_middle_filter = True

    def init(self):
        print("🌙 Moon Dev: Initializing CompressionPulse indicators...")

        close = self.data.Close
        high = self.data.High
        low = self.data.Low

        # Bollinger Bands (using talib wrappers to extract each band)
        self.bb_upper = self.I(_bbands_upper, close, self.bb_period,
                               self.bb_std, self.bb_std, 0, name="BB_Upper")
        self.bb_middle = self.I(_bbands_middle, close, self.bb_period,
                                self.bb_std, self.bb_std, 0, name="BB_Middle")
        self.bb_lower = self.I(_bbands_lower, close, self.bb_period,
                               self.bb_std, self.bb_std, 0, name="BB_Lower")

        # Bollinger Band Width
        self.bbw = self.I(lambda u, l, m: (u - l) / m,
                          self.bb_upper, self.bb_lower, self.bb_middle,
                          name="BBW")

        # RSI
        self.rsi = self.I(talib.RSI, close, timeperiod=self.rsi_period, name="RSI")

        # ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period, name="ATR")

        # Rolling min of BBW
        self.bbw_min = self.I(talib.MIN, self.bbw, timeperiod=self.squeeze_lookback, name="BBW_Min")

        # Trade state
        self.entry_price = None
        self.target = None
        self.stop = None
        self.last_exit_bar = -999

        print("✨ Moon Dev: Indicators ready. Squeeze detectors online!")

    def next(self):
        price = self.data.Close[-1]
        bbw_now = self.bbw[-1]
        bbw_min = self.bbw_min[-1]
        rsi_now = self.rsi[-1]
        atr_now = self.atr[-1]
        mid_now = self.bb_middle[-1]

        if np.isnan(bbw_now) or np.isnan(bbw_min) or np.isnan(rsi_now) or np.isnan(atr_now):
            return

        bar_index = len(self.data) - 1

        # ============================================================
        # Manage open position
        # ============================================================
        if self.position:
            high_now = self.data.High[-1]
            low_now = self.data.Low[-1]

            if high_now >= self.target:
                print(f"🎯 Moon Dev: TARGET HIT @ {self.target:.2f} (high={high_now:.2f}) — locking in lunar profits! 🌙💰")
                self.position.close()
                self.last_exit_bar = bar_index
                self.entry_price = None
                return

            if low_now <= self.stop:
                print(f"🛑 Moon Dev: STOP HIT @ {self.stop:.2f} (low={low_now:.2f}) — ejecting to safety! 🚨")
                self.position.close()
                self.last_exit_bar = bar_index
                self.entry_price = None
                return

            return

        # ============================================================
        # Cooldown check
        # ============================================================
        if bar_index - self.last_exit_bar < self.cooldown_bars:
            return

        # ============================================================
        # Entry conditions
        # ============================================================
        squeeze = bbw_now <= bbw_min
        rsi_bull = rsi_now > 50
        mid_filter = (price > mid_now) if self.use_middle_filter else True

        if squeeze and rsi_bull and mid_filter:
            entry_price = price
            target = entry_price + self.atr_mult * atr_now
            stop = entry_price - self.atr_mult * atr_now

            risk_per_unit = entry_price - stop
            if risk_per_unit <= 0:
                return

            equity = self.equity
            risk_amount = equity * self.risk_pct
            position_size = risk_amount / risk_per_unit
            position_size = int(round(position_size))

            if position_size < 1:
                print(f"⚠️ Moon Dev: Position size too small ({position_size}) — skipping signal.")
                return

            self.entry_price = entry_price
            self.target = target
            self.stop = stop

            print(f"🚀 Moon Dev: COMPRESSION PULSE DETECTED! BBW={bbw_now:.5f} (min={bbw_min:.5f}) "
                  f"RSI={rsi_now:.2f} ATR={atr_now:.2f}")
            print(f"   📈 Entry={entry_price:.2f} | Target={target:.2f} | Stop={stop:.2f} | Size={position_size}")

            self.buy(size=position_size)


# ============================================================
# Run the backtest
# ============================================================
print("🌙 Moon Dev: Launching CompressionPulse backtest...")
bt = Backtest(
    data,
    CompressionPulse,
    cash=1_000_000,
    commission=0.001,
    trade_on_close=True,
)

stats = bt.run()
print("\n" + "=" * 60)
print("🌙 Moon Dev's CompressionPulse — Final Lunar Stats 🌙")
print("=" * 60)
print(stats)
print(stats._strategy)
print("=" * 60)
print("🚀 Moon Dev: Backtest complete. To the moon! 🌙✨")