import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

print("🌙✨ Moon Dev Backtest AI Initializing... ✨🌙")
print("🚀 Loading VolatilityCompression Strategy...")

# Load data
data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
print(f"📂 Loading data from: {data_path}")
data = pd.read_csv(data_path)

# Clean columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Set datetime index
if 'datetime' in data.columns:
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')

# Map columns to proper case
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

# Ensure only OHLCV columns
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

print(f"✅ Data loaded: {len(data)} rows")
print(f"📊 Columns: {list(data.columns)}")
print(f"🌙 Moon Dev says: Let's backtest this volatility compression strategy! 🚀")


class VolatilityCompression(Strategy):
    """
    🌙 VolatilityCompression Strategy
    Targets mean reversion in volatility by shorting when volatility spikes
    to statistically extreme levels with ATR confirmation.
    """

    # Strategy parameters
    bb_period = 20
    bb_std = 2.0
    vvix_percentile = 90
    vvix_lookback = 252
    risk_per_trade = 0.02  # 2% of capital
    stop_loss_pct = 0.18  # 18% stop loss
    max_holding_days = 12  # Maximum holding period

    def init(self):
        print("🌙 Initializing indicators...")

        close = self.data.Close
        high = self.data.High
        low = self.data.Low

        # 20-day SMA (Bollinger midline)
        self.sma20 = self.I(talib.SMA, close, timeperiod=self.bb_period)

        # 20-day StdDev
        self.std20 = self.I(talib.STDDEV, close, timeperiod=self.bb_period, nbdev=1)

        # Upper Bollinger Band
        def calc_upper_band(sma, std):
            return sma + (self.bb_std * std)

        self.upper_bb = self.I(calc_upper_band, self.sma20, self.std20)

        # ATR as volatility proxy
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=14)

        # VVIX percentile threshold (90th percentile of ATR over lookback)
        def calc_vvix_p90(atr_arr):
            result = np.full(len(atr_arr), np.nan)
            lookback = self.vvix_lookback
            pct = self.vvix_percentile
            for i in range(lookback, len(atr_arr)):
                window = atr_arr[i-lookback:i]
                if len(window) > 0 and not np.all(np.isnan(window)):
                    result[i] = np.nanpercentile(window, pct)
            return result

        self.vvix_p90 = self.I(calc_vvix_p90, self.atr)

        # Entry day high tracking
        self.entry_high = None
        self.entry_bar = None

        print("✅ Indicators initialized!")
        print(f"🌙 BB Period: {self.bb_period}, BB StdDev: {self.bb_std}")
        print(f"🌙 VVIX Percentile: {self.vvix_percentile}, Lookback: {self.vvix_lookback}")
        print(f"🌙 Risk per trade: {self.risk_per_trade*100}%")
        print(f"🌙 Max holding: {self.max_holding_days} bars")

    def next(self):
        # Skip if not enough data
        if len(self.data) < self.bb_period + 5:
            return

        # Check if indicators are valid
        if (np.isnan(self.sma20[-1]) or np.isnan(self.upper_bb[-1]) or
            np.isnan(self.vvix_p90[-1]) or np.isnan(self.atr[-1])):
            return

        current_close = self.data.Close[-1]
        current_high = self.data.High[-1]
        current_low = self.data.Low[-1]
        current_sma = self.sma20[-1]
        current_upper_bb = self.upper_bb[-1]
        current_atr = self.atr[-1]
        current_vvix_p90 = self.vvix_p90[-1]

        # Position management
        if self.position:
            # Track entry high for stop loss
            if self.entry_high is None:
                self.entry_high = current_high
                self.entry_bar = len(self.data)
            else:
                self.entry_high = max(self.entry_high, current_high)

            bars_held = len(self.data) - self.entry_bar

            # Stop Loss: if price closes above entry-day high by additional 18%
            stop_level = self.entry_high * (1 + self.stop_loss_pct)
            if current_close > stop_level:
                print(f"🛑 STOP LOSS HIT! Close: {current_close:.2f} > Stop: {stop_level:.2f}")
                print(f"🌙 Moon Dev: Cutting losses, protecting capital! 💰")
                self.position.close()
                self.entry_high = None
                self.entry_bar = None
                return

            # Maximum holding period
            if bars_held >= self.max_holding_days:
                print(f"⏰ MAX HOLDING PERIOD REACHED ({bars_held} bars)")
                print(f"🌙 Moon Dev: Time to exit, thesis invalidated! 🚪")
                self.position.close()
                self.entry_high = None
                self.entry_bar = None
                return

            # Exit Rule: Close when price closes BELOW SMA(20)
            if current_close < current_sma:
                print(f"✅ EXIT SIGNAL - Mean reversion complete!")
                print(f"🌙 Close: {current_close:.2f} < SMA20: {current_sma:.2f}")
                print(f"🌙 Moon Dev: Taking profits, mean reversion worked! 🎯")
                self.position.close()
                self.entry_high = None
                self.entry_bar = None
                return

        else:
            # Entry Logic: SHORT volatility
            # Primary Trigger: Close ABOVE upper Bollinger Band
            bb_breakout = current_close > current_upper_bb

            # Confirmation Filter: ATR >= 90th percentile (VVIX proxy)
            vvix_confirmation = current_atr >= current_vvix_p90

            if bb_breakout and vvix_confirmation:
                print(f"🚀 ENTRY SIGNAL - Volatility Compression Setup!")
                print(f"🌙 Close: {current_close:.2f} > Upper BB: {current_upper_bb:.2f}")
                print(f"🌙 ATR: {current_atr:.4f} >= VVIX P90: {current_vvix_p90:.4f}")
                print(f"🌙 Moon Dev: Shorting volatility at extremes! 📉")

                # Position sizing: use fraction of equity (0 < size < 1)
                position_size = float(self.risk_per_trade)
                position_size = min(position_size, 0.99)
                position_size = max(position_size, 0.01)

                print(f"🌙 Position size: {position_size*100:.1f}% of equity")

                # FIX: For short trades, use self.sell with sl parameter for stop loss
                # The stop loss is handled manually in next(), so we just enter short
                self.sell(size=position_size)
                self.entry_high = current_high
                self.entry_bar = len(self.data)


print("🚀 Running initial backtest with default parameters...")
print("🌙 Moon Dev: Let's see how this volatility strategy performs! ✨")

bt = Backtest(
    data,
    VolatilityCompression,
    cash=1000000,
    commission=0.001,
    exclusive_orders=True
)

stats = bt.run()

print("\n" + "="*80)
print("🌙✨ MOON DEV BACKTEST RESULTS - VOLATILITY COMPRESSION STRATEGY ✨🌙")
print("="*80)
print(stats)
print("\n" + "="*80)
print("🌙 STRATEGY DETAILS:")
print("="*80)
print(stats._strategy)
print("="*80)
print("🌙 Moon Dev: Backtest complete! May the volatility be ever in your favor! 🚀")
print("="*80)