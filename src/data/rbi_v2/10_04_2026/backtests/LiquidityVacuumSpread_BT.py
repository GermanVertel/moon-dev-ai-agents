import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

print("🌙✨ LiquidityVacuumSpread Backtest Initializing... 🚀")

# Load data
data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper column mapping
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

print(f"🌙 Data loaded: {len(data)} bars")
print(f"✨ Columns: {list(data.columns)}")


class LiquidityVacuumSpread(Strategy):
    # Strategy parameters
    ema_period = 20
    rsi_period = 14
    vol_short = 5
    vol_long = 20
    volume_avg_period = 20

    # Liquidity thresholds
    volume_ratio_threshold = 0.6
    rsi_low = 30
    rsi_high = 55

    # Risk management
    risk_pct = 0.0075  # 0.75% of equity
    stop_loss_pct = 0.5  # 50% of position
    tp_mult = 1.75  # profit target multiple

    def init(self):
        print("🚀 Initializing indicators...")

        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        volume = pd.Series(self.data.Volume)

        # 20 EMA
        self.ema20 = self.I(talib.EMA, self.data.Close, timeperiod=self.ema_period)

        # RSI(14)
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=self.rsi_period)

        # Volume average (20-day)
        self.vol_avg = self.I(talib.SMA, self.data.Volume, timeperiod=self.volume_avg_period)

        # Realized volatility - 5 day and 20 day
        # Use ATR as proxy for realized vol
        self.atr5 = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.vol_short)
        self.atr20 = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.vol_long)

        print("✨ Indicators ready! Moon Dev power engaged 🌙")

    def next(self):
        # Skip if not enough data
        if len(self.data) < self.vol_long + 5:
            return

        price = self.data.Close[-1]
        ema = self.ema20[-1]
        ema_prev = self.ema20[-2] if len(self.ema20) > 1 else ema
        rsi = self.rsi[-1]
        vol = self.data.Volume[-1]
        vol_avg = self.vol_avg[-1]
        atr5 = self.atr5[-1]
        atr20 = self.atr20[-1]

        # Guard against NaN
        if any(np.isnan([ema, ema_prev, rsi, vol_avg, atr5, atr20])):
            return

        # Liquidity ratio
        vol_ratio = vol / vol_avg if vol_avg > 0 else 1.0

        # Volatility compression ratio
        vol_compression = atr5 / atr20 if atr20 > 0 else 1.0

        # EMA slope
        ema_declining = ema <= ema_prev

        # ---- ENTRY CONDITIONS ----
        liquidity_low = vol_ratio < self.volume_ratio_threshold
        below_ema = price < ema
        ema_weak = ema_declining
        rsi_ok = self.rsi_low <= rsi <= self.rsi_high
        vol_compressed = vol_compression < 1.0

        entry_signal = liquidity_low and below_ema and ema_weak and rsi_ok and vol_compressed

        # ---- EXIT CONDITIONS ----
        if self.position:
            # Volume normalization exit
            if vol_ratio > 1.0:
                print(f"🌙 Liquidity returned! vol_ratio={vol_ratio:.2f} - EXITING")
                self.position.close()
                return

            # Price above EMA exit (thesis invalidated)
            if price > ema:
                print(f"🌙 Price broke above EMA - EXITING")
                self.position.close()
                return

        # ---- EXECUTE ENTRY ----
        if not self.position and entry_signal:
            # Position sizing: risk_pct of equity
            # Using size=1_000_000 as requested
            size = 1_000_000

            # Stop loss and take profit levels
            # For a short/bearish trade, stop is above entry
            stop_price = price * (1 + self.stop_loss_pct * 0.02)  # ~1% stop
            tp_price = price * (1 - self.stop_loss_pct * 0.04)  # ~2% target

            print(f"🚀🌙 ENTRY SIGNAL! Price={price:.2f} EMA={ema:.2f} RSI={rsi:.1f} "
                  f"vol_ratio={vol_ratio:.2f} vol_comp={vol_compression:.2f}")

            # Since we can't short directly in backtesting.py easily with puts,
            # we simulate the bear put spread as a short position
            self.sell(size=size, sl=stop_price, tp=tp_price)


print("🌙 Running backtest...")
bt = Backtest(data, LiquidityVacuumSpread, cash=10_000_000, commission=0.002)
stats = bt.run()
print(stats)
print(stats._strategy)
print("🚀✨ Moon Dev Backtest Complete! 🌙")