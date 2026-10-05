import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

print("🌙✨ Moon Dev's VolatilityCompressionMomentum Backtest Starting! 🚀🚀")

# Load data
data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Rename columns properly
data = data.rename(columns={
    'open': 'Open', 'high': 'High', 'low': 'Low',
    'close': 'Close', 'volume': 'Volume'
})

# Set datetime index
if 'datetime' in data.columns:
    data = data.set_index(pd.to_datetime(data['datetime']))
    data = data.drop(columns=['datetime'])

# Ensure required columns
required = ['Open', 'High', 'Low', 'Close', 'Volume']
for col in required:
    if col not in data.columns:
        raise ValueError(f"Missing required column: {col}")

# Drop any NaN rows
data = data.dropna()

print(f"🌙 Data loaded: {len(data)} bars")
print(f"🚀 Columns: {list(data.columns)}")


class VolatilityCompressionMomentum(Strategy):
    # Strategy parameters
    bb_period = 20
    bb_std = 2.0
    atr_period = 14
    atr_pct_threshold = 10.0
    ema_period = 20
    ema_distance_threshold = 0.07  # 7%
    stop_atr_mult = 1.5
    time_stop_bars = 18
    risk_pct = 0.02  # 2% risk per trade

    def init(self):
        print("🌙 Initializing indicators...")
        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)

        # Bollinger Bands
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS, close, timeperiod=self.bb_period,
            nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0
        )

        # ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        # EMA
        self.ema = self.I(talib.EMA, close, timeperiod=self.ema_period)

        # Volume SMA for optional confirmation
        self.vol_sma = self.I(talib.SMA, pd.Series(self.data.Volume), timeperiod=20)

        print("✨ Indicators initialized!")

    def next(self):
        price = self.data.Close[-1]

        # Skip if indicators not ready
        if np.isnan(self.bb_upper[-1]) or np.isnan(self.atr[-1]) or np.isnan(self.ema[-1]):
            return

        # Compute ATR%
        atr_pct = (self.atr[-1] / price) * 100 if price > 0 else 999

        # EMA distance
        ema_dist = (price - self.ema[-1]) / self.ema[-1] if self.ema[-1] > 0 else 0

        # ============ EXIT LOGIC ============
        if self.position:
            entry_price = self.trades[-1].entry_price if self.trades else price

            # Track entry ATR for stop
            if not hasattr(self, '_entry_atr'):
                self._entry_atr = self.atr[-1]
            if not hasattr(self, '_entry_bar'):
                self._entry_bar = len(self.data)

            bars_since_entry = len(self.data) - self._entry_bar

            # 1. Hard stop-loss: entry - 1.5 * ATR
            stop_price = entry_price - (self.stop_atr_mult * self._entry_atr)

            # 2. Primary exit: Close below upper BB (reversion)
            bb_reversion = price < self.bb_upper[-1]

            # 3. EMA distance exit (mean-reversion adjustment)
            ema_exit = ema_dist > self.ema_distance_threshold

            # 4. EMA cross below - force exit
            ema_cross_exit = price < self.ema[-1]

            # 5. Time stop
            time_exit = bars_since_entry >= self.time_stop_bars

            # 6. Hard stop hit
            stop_hit = price <= stop_price

            if stop_hit:
                print(f"🛑 Stop hit at {price:.2f} (stop: {stop_price:.2f}) 🌙")
                self.position.close()
                self._cleanup()
            elif ema_cross_exit:
                print(f"📉 EMA cross exit at {price:.2f} (EMA: {self.ema[-1]:.2f}) 🌙")
                self.position.close()
                self._cleanup()
            elif ema_exit:
                print(f"⚡ EMA distance exit at {price:.2f} (dist: {ema_dist*100:.2f}%) 🌙")
                self.position.close()
                self._cleanup()
            elif bb_reversion:
                print(f"✅ BB reversion exit at {price:.2f} (upper: {self.bb_upper[-1]:.2f}) ✨")
                self.position.close()
                self._cleanup()
            elif time_exit:
                print(f"⏰ Time stop exit at {price:.2f} after {bars_since_entry} bars 🌙")
                self.position.close()
                self._cleanup()

        # ============ ENTRY LOGIC ============
        else:
            # Reset entry tracking
            self._cleanup()

            # Condition 1: Close above upper BB
            bb_breakout = price > self.bb_upper[-1]

            # Condition 2: ATR% < threshold
            low_vol = atr_pct < self.atr_pct_threshold

            # Condition 3: Close > EMA (trend alignment)
            trend_align = price > self.ema[-1]

            # Optional: Volume expansion
            vol_expansion = self.data.Volume[-1] > self.vol_sma[-1] if not np.isnan(self.vol_sma[-1]) else True

            if bb_breakout and low_vol and trend_align:
                # Position sizing based on risk
                stop_price = price - (self.stop_atr_mult * self.atr[-1])
                risk_per_unit = price - stop_price

                if risk_per_unit > 0:
                    risk_amount = self.equity * self.risk_pct
                    position_size = risk_amount / risk_per_unit
                    position_size = int(round(position_size))
                    position_size = min(position_size, 1000000)

                    if position_size > 0:
                        print(f"🚀🌙 ENTRY SIGNAL! Price: {price:.2f} | BB_upper: {self.bb_upper[-1]:.2f} | ATR%: {atr_pct:.2f}% | Size: {position_size}")
                        self.buy(size=position_size)
                        self._entry_atr = self.atr[-1]
                        self._entry_bar = len(self.data)

    def _cleanup(self):
        if hasattr(self, '_entry_atr'):
            del self._entry_atr
        if hasattr(self, '_entry_bar'):
            del self._entry_bar


print("🌙 Running backtest...")
bt = Backtest(
    data,
    VolatilityCompressionMomentum,
    cash=1_000_000,
    commission=0.001
)

stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙✨ Backtest complete! 🚀🚀")