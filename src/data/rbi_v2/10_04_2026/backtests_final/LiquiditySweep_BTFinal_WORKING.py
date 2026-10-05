import pandas as pd
import numpy as np
import talib
from backtesting import Backtest, Strategy

# ============================================================
# 🌙 MOON DEV - LIQUIDITY SWEEP STRATEGY 🌙
# ============================================================

DATA_PATH = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"

print("🌙✨ Moon Dev Backtest Engine Warming Up... ✨🌙")

# Load and clean data
data = pd.read_csv(DATA_PATH)
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Ensure proper column mapping
data = data.rename(columns={
    'datetime': 'Date',
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

data['Date'] = pd.to_datetime(data['Date'])
data = data.set_index('Date')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
data = data.dropna()

print(f"🚀 Loaded {len(data)} bars of data")
print(f"📊 Date range: {data.index[0]} → {data.index[-1]}")


class LiquiditySweep(Strategy):
    """
    🌙 LiquiditySweep Strategy 🌙
    Volume-confirmed Donchian breakout with ATR risk management.
    """
    # Tunable params
    donchian_period = 20
    vma_period = 20
    atr_period = 14
    rvol_threshold = 2.0
    ema_period = 200
    stop_atr_mult = 1.5
    trail_atr_mult = 2.0
    risk_pct = 0.01
    time_stop_bars = 5
    wick_max_pct = 0.40

    def init(self):
        print("🌙 Initializing Moon Dev LiquiditySweep indicators...")

        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        volume = pd.Series(self.data.Volume)

        # 📊 Donchian Channels (use prior N-bar high/low to avoid lookahead)
        self.dc_upper = self.I(
            lambda s: pd.Series(s).rolling(self.donchian_period).max().shift(1).values,
            high
        )
        self.dc_lower = self.I(
            lambda s: pd.Series(s).rolling(self.donchian_period).min().shift(1).values,
            low
        )

        # 📊 Volume MA & RVOL
        self.vma = self.I(talib.SMA, volume, timeperiod=self.vma_period)
        self.rvol = self.I(
            lambda v, m: np.where(m > 0, np.array(v) / np.array(m), 0.0),
            volume, self.vma
        )

        # 📊 ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        # 📊 Trend filter EMA
        self.ema200 = self.I(talib.EMA, close, timeperiod=self.ema_period)

        # 📊 Volume max of prior 5 bars
        self.vol_max5 = self.I(
            lambda v: pd.Series(v).rolling(5).max().shift(1).values,
            volume
        )

        # State
        self.entry_price = None
        self.stop_price = None
        self.target1 = None
        self.target2 = None
        self.bars_in_trade = 0
        self.scaled_out = False
        self.trail_high = None
        self.trail_low = None

        print("✨ Indicators ready. Let the liquidity hunt begin! 🎯")

    def next(self):
        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        open_ = self.data.Open[-1]

        # ---------- Manage existing position ----------
        if self.position:
            self.bars_in_trade += 1

            if self.position.is_long:
                # Update trailing high
                self.trail_high = max(self.trail_high, high)

                # Chandelier trailing stop
                chandelier = self.trail_high - self.trail_atr_mult * self.atr[-1]
                if chandelier > self.stop_price:
                    self.stop_price = chandelier

                # Scale out at 2R
                if (not self.scaled_out) and price >= self.target1:
                    print(f"💰🌙 [LONG] 2R hit at {price:.2f} — scaling out 50%")
                    self.position.close(portion=0.5)
                    self.scaled_out = True

                # Stop hit
                if price <= self.stop_price:
                    print(f"🛑 [LONG] Trail/Stop hit at {price:.2f}")
                    self.position.close()
                    self._reset_trade()
                    return

                # Time stop
                if self.bars_in_trade >= self.time_stop_bars and price < self.entry_price:
                    print(f"⏰ [LONG] Time stop — momentum faded")
                    self.position.close()
                    self._reset_trade()
                    return

            elif self.position.is_short:
                self.trail_low = min(self.trail_low, low)

                chandelier = self.trail_low + self.trail_atr_mult * self.atr[-1]
                if chandelier < self.stop_price:
                    self.stop_price = chandelier

                if (not self.scaled_out) and price <= self.target1:
                    print(f"💰🌙 [SHORT] 2R hit at {price:.2f} — scaling out 50%")
                    self.position.close(portion=0.5)
                    self.scaled_out = True

                if price >= self.stop_price:
                    print(f"🛑 [SHORT] Trail/Stop hit at {price:.2f}")
                    self.position.close()
                    self._reset_trade()
                    return

                if self.bars_in_trade >= self.time_stop_bars and price > self.entry_price:
                    print(f"⏰ [SHORT] Time stop — momentum faded")
                    self.position.close()
                    self._reset_trade()
                    return

            return  # Don't stack positions

        # ---------- Entry Logic ----------
        if len(self.data) < self.ema_period + 2:
            return

        # Skip if indicators not ready
        if np.isnan(self.dc_upper[-1]) or np.isnan(self.atr[-1]) or np.isnan(self.ema200[-1]):
            return

        rvol = self.rvol[-1]
        vol_max5 = self.vol_max5[-1]

        # Bar range & wick analysis
        bar_range = high - low
        if bar_range <= 0:
            return

        upper_wick = high - max(open_, price)
        lower_wick = min(open_, price) - low
        close_position = (price - low) / bar_range  # 0 = bottom, 1 = top

        # ---------- LONG ----------
        long_breakout = price > self.dc_upper[-1]
        long_rvol = rvol >= self.rvol_threshold
        long_vol_confirm = self.data.Volume[-1] > vol_max5
        long_trend = price > self.ema200[-1]
        long_strong_close = close_position >= 0.70
        long_ok_wick = upper_wick <= self.wick_max_pct * bar_range

        if long_breakout and long_rvol and long_vol_confirm and long_trend and long_strong_close and long_ok_wick:
            stop = price - self.stop_atr_mult * self.atr[-1]
            risk = price - stop
            if risk <= 0:
                return

            size = int(round(1_000_000 / price))
            if size <= 0:
                return

            self.entry_price = price
            self.stop_price = stop
            self.target1 = price + 2 * risk
            self.target2 = price + 3.5 * risk
            self.bars_in_trade = 0
            self.scaled_out = False
            self.trail_high = high

            print(f"🚀🌙 [LONG SIGNAL] Breakout @ {price:.2f} | RVOL={rvol:.2f} | ATR={self.atr[-1]:.2f} | Size={size}")
            self.buy(size=size)
            return

        # ---------- SHORT ----------
        short_breakout = price < self.dc_lower[-1]
        short_rvol = rvol >= self.rvol_threshold
        short_vol_confirm = self.data.Volume[-1] > vol_max5
        short_trend = price < self.ema200[-1]
        short_strong_close = close_position <= 0.30
        short_ok_wick = lower_wick <= self.wick_max_pct * bar_range

        if short_breakout and short_rvol and short_vol_confirm and short_trend and short_strong_close and short_ok_wick:
            stop = price + self.stop_atr_mult * self.atr[-1]
            risk = stop - price
            if risk <= 0:
                return

            size = int(round(1_000_000 / price))
            if size <= 0:
                return

            self.entry_price = price
            self.stop_price = stop
            self.target1 = price - 2 * risk
            self.target2 = price - 3.5 * risk
            self.bars_in_trade = 0
            self.scaled_out = False
            self.trail_low = low

            print(f"🔻🌙 [SHORT SIGNAL] Breakdown @ {price:.2f} | RVOL={rvol:.2f} | ATR={self.atr[-1]:.2f} | Size={size}")
            self.sell(size=size)
            return

    def _reset_trade(self):
        self.entry_price = None
        self.stop_price = None
        self.target1 = None
        self.target2 = None
        self.bars_in_trade = 0
        self.scaled_out = False
        self.trail_high = None
        self.trail_low = None


# ============================================================
# 🌙 RUN THE BACKTEST 🌙
# ============================================================
print("🚀🌙 Launching Moon Dev LiquiditySweep Backtest...")
bt = Backtest(
    data,
    LiquiditySweep,
    cash=1_000_000,
    commission=0.0002,
    exclusive_orders=True
)

stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙✨ Backtest complete. Moon Dev out! ✨🌙")