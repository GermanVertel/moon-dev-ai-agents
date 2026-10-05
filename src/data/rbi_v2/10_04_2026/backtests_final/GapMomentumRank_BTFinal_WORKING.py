import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# 🌙 Moon Dev's GapMomentumRank Backtest 🌙
print("🚀 Initializing Moon Dev's GapMomentumRank Strategy...")
print("✨ Loading celestial data from the moon base...")

# Load data
data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# 🌙 Data cleaning ritual
print("🧹 Cleaning data columns...")
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

print(f"🌙 Data loaded: {len(data)} rows from {data.index[0]} to {data.index[-1]}")
print(f"✨ Columns: {list(data.columns)}")


class GapMomentumRank(Strategy):
    """
    🌙 GapMomentumRank Strategy 🌙
    Cross-sectional momentum based on normalized gap size.
    Since we have single-asset data (BTC-USD), we simulate the
    cross-sectional ranking by using rolling percentile ranks of
    the gap value over a lookback window.
    """

    # Strategy parameters
    gap_lookback = 20          # Rolling window for gap percentile ranking
    min_abs_gap = 0.5          # Minimum absolute normalized gap threshold
    atr_period = 14            # ATR period for volatility
    atr_stop_mult = 1.5        # ATR multiplier for stops
    profit_target_mult = 1.5   # Profit target = 1.5x prior day range
    top_pct = 0.20             # Top 20% long candidates
    bottom_pct = 0.20          # Bottom 20% short candidates
    risk_pct = 0.02            # 2% risk per trade
    max_hold_bars = 96         # Max hold (96 x 15m = 24 hours)

    def init(self):
        print("🌙 Initializing indicators with Moon Dev magic...")

        # Prior day high/low/close - use previous bar values (shifted)
        # For 15m data, "prior day" approximated by rolling daily window
        # We use rolling 96 bars (24h) for daily high/low/close
        self.prior_high = self.I(
            lambda h: pd.Series(h).rolling(96).max().shift(1).values,
            self.data.High
        )
        self.prior_low = self.I(
            lambda l: pd.Series(l).rolling(96).min().shift(1).values,
            self.data.Low
        )
        self.prior_close = self.I(
            lambda c: pd.Series(c).shift(96).values,
            self.data.Close
        )

        # ATR for volatility-based sizing and stops
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close,
                          timeperiod=self.atr_period)

        # Volume moving average for liquidity filter
        self.vol_ma = self.I(talib.SMA, self.data.Volume, timeperiod=20)

        # 50-period MA for trend filter
        self.ma50 = self.I(talib.SMA, self.data.Close, timeperiod=50)

        # Compute normalized gap and its rolling percentile rank
        def compute_gap_rank(open_, prior_close, prior_high, prior_low):
            prior_range = prior_high - prior_low
            prior_range = np.where(prior_range <= 0, np.nan, prior_range)
            gap = (open_ - prior_close) / prior_range
            gap = np.where(np.isnan(gap), 0, gap)
            # Rolling percentile rank of gap over lookback
            gap_series = pd.Series(gap)
            rank = gap_series.rolling(self.gap_lookback).rank(pct=True).values
            return rank

        self.gap_rank = self.I(
            compute_gap_rank,
            self.data.Open,
            self.prior_close,
            self.prior_high,
            self.prior_low
        )

        # Compute raw gap for entry threshold checks
        def compute_gap(open_, prior_close, prior_high, prior_low):
            prior_range = prior_high - prior_low
            prior_range = np.where(prior_range <= 0, np.nan, prior_range)
            gap = (open_ - prior_close) / prior_range
            return np.where(np.isnan(gap), 0, gap)

        self.gap = self.I(
            compute_gap,
            self.data.Open,
            self.prior_close,
            self.prior_high,
            self.prior_low
        )

        self.bar_count = 0
        self.entry_bar = 0
        self.entry_price = 0.0
        print("✨ All indicators initialized! Ready to ride the momentum wave 🚀")

    def next(self):
        self.bar_count += 1

        # Skip if indicators not ready
        if len(self.data) < 200:
            return

        price = self.data.Close[-1]
        current_gap = self.gap[-1]
        current_rank = self.gap_rank[-1]
        current_atr = self.atr[-1]
        vol = self.data.Volume[-1]
        vol_avg = self.vol_ma[-1]
        ma = self.ma50[-1]

        if np.isnan(current_rank) or np.isnan(current_atr) or current_atr <= 0:
            return

        # 🌙 Liquidity filter: volume above average
        if vol_avg > 0 and vol < vol_avg * 0.5:
            return

        # ============ EXIT LOGIC ============
        if self.position:
            bars_held = self.bar_count - self.entry_bar

            # Time-based exit
            if bars_held >= self.max_hold_bars:
                print(f"⏰ Moon Dev time exit at bar {self.bar_count} | Price: {price:.2f}")
                self.position.close()
                return

            # Stop-loss / profit target using ATR
            if self.position.is_long:
                stop_price = self.entry_price - self.atr_stop_mult * current_atr
                target_price = self.entry_price + self.profit_target_mult * current_atr

                if self.data.Low[-1] <= stop_price:
                    print(f"🛑 Moon Dev LONG stop hit | Entry: {self.entry_price:.2f} | Stop: {stop_price:.2f}")
                    self.position.close()
                    return
                if self.data.High[-1] >= target_price:
                    print(f"🎯 Moon Dev LONG target hit | Entry: {self.entry_price:.2f} | Target: {target_price:.2f}")
                    self.position.close()
                    return

            elif self.position.is_short:
                stop_price = self.entry_price + self.atr_stop_mult * current_atr
                target_price = self.entry_price - self.profit_target_mult * current_atr

                if self.data.High[-1] >= stop_price:
                    print(f"🛑 Moon Dev SHORT stop hit | Entry: {self.entry_price:.2f} | Stop: {stop_price:.2f}")
                    self.position.close()
                    return
                if self.data.Low[-1] <= target_price:
                    print(f"🎯 Moon Dev SHORT target hit | Entry: {self.entry_price:.2f} | Target: {target_price:.2f}")
                    self.position.close()
                    return

        # ============ ENTRY LOGIC ============
        if not self.position:
            # Long: high gap rank (top 20%) AND positive gap above threshold
            if current_rank >= (1 - self.top_pct) and current_gap > self.min_abs_gap:
                # Trend filter: only long above 50-MA
                if not np.isnan(ma) and price > ma:
                    # Risk-based position sizing
                    risk_amount = self.equity * self.risk_pct
                    stop_distance = self.atr_stop_mult * current_atr
                    if stop_distance > 0:
                        position_size = int(round(risk_amount / stop_distance))
                        position_size = max(1, min(position_size, int(self.equity / price)))
                        print(f"🚀 Moon Dev LONG signal! Rank: {current_rank:.2f} | Gap: {current_gap:.2f} | Size: {position_size}")
                        self.buy(size=position_size)
                        self.entry_bar = self.bar_count
                        self.entry_price = price

            # Short: low gap rank (bottom 20%) AND negative gap below threshold
            elif current_rank <= self.bottom_pct and current_gap < -self.min_abs_gap:
                # Trend filter: only short below 50-MA
                if not np.isnan(ma) and price < ma:
                    risk_amount = self.equity * self.risk_pct
                    stop_distance = self.atr_stop_mult * current_atr
                    if stop_distance > 0:
                        position_size = int(round(risk_amount / stop_distance))
                        position_size = max(1, min(position_size, int(self.equity / price)))
                        print(f"🔻 Moon Dev SHORT signal! Rank: {current_rank:.2f} | Gap: {current_gap:.2f} | Size: {position_size}")
                        self.sell(size=position_size)
                        self.entry_bar = self.bar_count
                        self.entry_price = price


print("🌙 Setting up backtest engine...")
bt = Backtest(
    data,
    GapMomentumRank,
    cash=1_000_000,
    commission=0.001,
    exclusive_orders=True
)

print("🚀 Running Moon Dev's GapMomentumRank backtest...")
stats = bt.run()
print(stats)
print(stats._strategy)