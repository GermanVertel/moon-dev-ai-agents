import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

print("🌙✨ Starting Moon Dev's VolumetricSqueeze Backtest ✨🌙")

# Load and clean data
data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

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

# Set datetime index
if 'datetime' in data.columns:
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')

print(f"🌙 Data loaded: {len(data)} candles")
print(f"🌙 Date range: {data.index[0]} to {data.index[-1]}")


class VolumetricSqueeze(Strategy):
    """
    🌙 VolumetricSqueeze Strategy 🌙
    Triple confluence: BB breakout + Volume confirmation + Bullish Engulfing
    """
    # Strategy parameters
    bb_period = 20
    bb_std = 2.0
    vol_period = 20
    vol_multiplier = 1.5
    atr_period = 14
    atr_multiplier = 1.5
    risk_pct = 0.01  # 1% risk per trade
    trend_period = 200
    time_exit_bars = 8

    def init(self):
        print("🌙 Initializing VolumetricSqueeze indicators...")

        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        volume = pd.Series(self.data.Volume)
        open_ = pd.Series(self.data.Open)

        # Bollinger Bands
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS, close, timeperiod=self.bb_period,
            nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0,
            name=['BB_Upper', 'BB_Middle', 'BB_Lower']
        )

        # Volume SMA
        self.vol_sma = self.I(talib.SMA, volume, timeperiod=self.vol_period, name='Vol_SMA')

        # ATR for trailing stop
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period, name='ATR')

        # Trend filter - 200 SMA
        self.trend_sma = self.I(talib.SMA, close, timeperiod=self.trend_period, name='Trend_SMA')

        # RSI for confirmation
        self.rsi = self.I(talib.RSI, close, timeperiod=14, name='RSI')

        # Store raw data for engulfing detection
        self._open = open_.values
        self._close = close.values
        self._high = high.values
        self._low = low.values

        # Track trade state
        self.entry_bar = None
        self.entry_price = None
        self.stop_price = None
        self.highest_high = None

        print("🌙✨ Indicators initialized successfully! ✨🌙")

    def is_bullish_engulfing(self, i):
        """Check for bullish engulfing pattern at bar i"""
        if i < 1:
            return False

        curr_open = self._open[i]
        curr_close = self._close[i]
        prev_open = self._open[i - 1]
        prev_close = self._close[i - 1]

        # Current candle bullish
        curr_bullish = curr_close > curr_open
        # Prior candle bearish
        prev_bearish = prev_close < prev_open

        if not (curr_bullish and prev_bearish):
            return False

        # Current body engulfs prior body
        curr_body_low = min(curr_open, curr_close)
        curr_body_high = max(curr_open, curr_close)
        prev_body_low = min(prev_open, prev_close)
        prev_body_high = max(prev_open, prev_close)

        engulfs = (curr_body_low <= prev_body_low) and (curr_body_high >= prev_body_high)

        # Body strength check - avoid weak patterns with long wicks
        curr_body = abs(curr_close - curr_open)
        curr_range = self._high[i] - self._low[i]
        if curr_range > 0:
            body_ratio = curr_body / curr_range
            if body_ratio < 0.5:
                return False

        return engulfs

    def next(self):
        i = len(self.data) - 1

        # Manage open position
        if self.position:
            # Update highest high for trailing
            if self.highest_high is None or self._high[i] > self.highest_high:
                self.highest_high = self._high[i]

            # Time-based exit
            bars_in_trade = i - self.entry_bar if self.entry_bar is not None else 0
            if bars_in_trade >= self.time_exit_bars:
                # Check if made new high recently
                recent_high = max(self._high[i - 3:i + 1]) if i >= 3 else self._high[i]
                if self.highest_high <= recent_high:
                    print(f"🌙⏰ Time-based exit at {self._close[i]:.2f}")
                    self.position.close()
                    self._reset_trade_state()
                    return

            # Trailing stop using ATR
            if self.atr[i] > 0:
                new_stop = self._close[i] - self.atr_multiplier * self.atr[i]
                if self.stop_price is None or new_stop > self.stop_price:
                    self.stop_price = new_stop

            # Check stop loss hit
            if self.stop_price is not None and self._low[i] <= self.stop_price:
                print(f"🌙🛑 Stop loss hit at {self.stop_price:.2f}")
                self.position.close()
                self._reset_trade_state()
                return

            # Take profit at middle band
            if (self.entry_price is not None and
                    self._close[i] >= self.bb_middle[i] and
                    self.entry_price < self.bb_middle[i]):
                print(f"🌙🎯 Take profit at middle band {self.bb_middle[i]:.2f}")
                self.position.close()
                self._reset_trade_state()
                return

            # Momentum failure exit - close back below upper band with declining volume
            if i >= 1:
                if (self._close[i] < self.bb_upper[i] and
                        self._close[i - 1] >= self.bb_upper[i - 1] and
                        self.data.Volume[i] < self.data.Volume[i - 1]):
                    print(f"🌙📉 Momentum failure exit at {self._close[i]:.2f}")
                    self.position.close()
                    self._reset_trade_state()
                    return
            return

        # Entry logic
        if i < max(self.trend_period, self.bb_period, self.vol_period) + 1:
            return

        # Skip if band width is extremely wide (late-stage breakout)
        if self.bb_middle[i] == 0:
            return
        band_width = (self.bb_upper[i] - self.bb_lower[i]) / self.bb_middle[i]
        if band_width > 0.15:  # 15% width filter
            return

        # Condition 1: Close > Upper Bollinger Band
        bb_breakout = self._close[i] > self.bb_upper[i]

        # Condition 2: Volume confirmation
        vol_confirm = self.data.Volume[i] > self.vol_multiplier * self.vol_sma[i]

        # Condition 3: Bullish Engulfing
        engulfing = self.is_bullish_engulfing(i)

        # Confirmation enhancements
        rsi_confirm = self.rsi[i] > 50
        trend_confirm = self._close[i] > self.trend_sma[i]

        if bb_breakout and vol_confirm and engulfing and rsi_confirm and trend_confirm:
            print(f"🌙🚀 SIGNAL! BB Breakout + Volume + Engulfing at {self._close[i]:.2f}")
            print(f"   📊 BB Upper: {self.bb_upper[i]:.2f}, Vol: {self.data.Volume[i]:.2f} vs Avg: {self.vol_sma[i]:.2f}")
            print(f"   📈 RSI: {self.rsi[i]:.2f}, Trend SMA: {self.trend_sma[i]:.2f}")

            # Calculate stop loss - below engulfing candle low or ATR based
            stop_candle = self._low[i]
            stop_atr = self._close[i] - self.atr_multiplier * self.atr[i]
            stop_price = min(stop_candle, stop_atr)

            # Risk-based position sizing
            risk_per_unit = self._close[i] - stop_price
            if risk_per_unit <= 0:
                return

            equity = self.equity
            risk_amount = equity * self.risk_pct
            position_size = int(round(risk_amount / risk_per_unit))

            # Cap position size
            max_size = int(equity * 0.95 / self._close[i])
            position_size = min(position_size, max_size)

            if position_size < 1:
                position_size = 1

            print(f"   💰 Position size: {position_size} units, Stop: {stop_price:.2f}")

            self.buy(size=position_size)
            self.entry_bar = i
            self.entry_price = self._close[i]
            self.stop_price = stop_price
            self.highest_high = self._high[i]

    def _reset_trade_state(self):
        self.entry_bar = None
        self.entry_price = None
        self.stop_price = None
        self.highest_high = None


print("🌙✨ Running VolumetricSqueeze Backtest... ✨🌙")

bt = Backtest(
    data,
    VolumetricSqueeze,
    cash=1_000_000,
    commission=0.002,
)

stats = bt.run()
print(stats)
print(stats._strategy)