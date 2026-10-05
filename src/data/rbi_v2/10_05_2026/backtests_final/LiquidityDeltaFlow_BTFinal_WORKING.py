import talib
import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy

# 🌙 Moon Dev's LiquidityDeltaFlow Backtest 🌙
print("🚀 Moon Dev is warming up the engines...")

# Load data
data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper case mapping
data.columns = [col.capitalize() for col in data.columns]
if 'Datetime' in data.columns:
    data = data.drop(columns=['Datetime'])

print(f"🌙 Data loaded: {len(data)} bars")
print(f"✨ Columns: {list(data.columns)}")


class LiquidityDeltaFlow(Strategy):
    """
    🌙 Moon Dev's LiquidityDeltaFlow Strategy 🌙

    Since we only have OHLCV data (no L2 order book), we approximate:
    - Bid-Ask Spread proxy: high-low range normalized by close (Corwin-Schultz style)
    - Cumulative Net Delta proxy: signed volume using tick rule (close vs prev close)
    - OFI proxy: volume-weighted directional pressure
    """

    # Parameters
    lookback_n = 10              # lookback for spread change & CND
    spread_threshold = 0.15      # % contraction/expansion threshold
    trend_period = 20            # EMA trend filter
    atr_period = 14              # ATR for stops
    max_hold_bars = 20           # time-based exit
    risk_pct = 0.02              # 2% risk per trade
    spread_pctile = 90           # skip if spread in top decile
    size_cap_mult = 3.0          # max size multiplier

    def init(self):
        print("🌙 Initializing LiquidityDeltaFlow indicators...")

        # Trend filter: EMA-20
        self.ema = self.I(talib.EMA, self.data.Close, timeperiod=self.trend_period)
        print("✨ EMA-20 ready")

        # Volatility: ATR
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close,
                          timeperiod=self.atr_period)
        print("✨ ATR ready")

        # Bid-Ask Spread proxy: (High - Low) / Close * 10000 (bps)
        high = self.data.High
        low = self.data.Low
        close = self.data.Close
        spread = (high - low) / close * 10000
        self.spread = self.I(lambda x: x, spread, name='SpreadProxy')
        print("✨ Spread proxy ready")

        # Rolling median of spread
        self.spread_median = self.I(talib.SMA, self.spread, timeperiod=self.lookback_n * 3)
        print("✨ Spread median ready")

        # Spread change over lookback (pct change)
        spread_series = pd.Series(np.asarray(spread))
        spread_change = spread_series.pct_change(self.lookback_n) * 100
        self.spread_change = self.I(lambda x: x, spread_change.values, name='SpreadChange')
        print("✨ Spread change ready")

        # Cumulative Net Delta proxy: tick rule signed volume
        close_series = pd.Series(np.asarray(close))
        price_diff = close_series.diff()
        sign = np.sign(price_diff).fillna(0)
        signed_vol = sign * pd.Series(np.asarray(self.data.Volume))
        cnd = signed_vol.rolling(self.lookback_n).sum()
        self.cnd = self.I(lambda x: x, cnd.values, name='CND')
        print("✨ Cumulative Net Delta ready")

        # Rolling median |CND| for sizing normalization
        abs_cnd = cnd.abs()
        self.cnd_median = self.I(lambda x: x, abs_cnd.rolling(self.lookback_n * 3).median().values,
                                  name='CND_Median')
        print("✨ CND median ready")

        # Spread percentile threshold (rolling)
        self.spread_p90 = self.I(lambda x: x, spread_series.rolling(self.lookback_n * 5).quantile(0.9).values,
                                  name='Spread_P90')
        print("✨ Spread percentile ready")

        # Track bars in trade
        self.bar_count = 0
        self.entry_price = 0
        print("🚀 All indicators initialized! Let's fly!")

    def next(self):
        price = self.data.Close[-1]

        # Skip if not enough history
        if len(self.data) < self.lookback_n * 5 + 5:
            return

        # Get indicator values
        spread_chg = self.spread_change[-1]
        cnd_val = self.cnd[-1]
        cnd_med = self.cnd_median[-1]
        ema_val = self.ema[-1]
        atr_val = self.atr[-1]
        spread_val = self.spread[-1]
        spread_p90 = self.spread_p90[-1]

        if np.isnan(spread_chg) or np.isnan(cnd_val) or np.isnan(ema_val) or np.isnan(atr_val):
            return

        # Liquidity filter: skip if spread is in top decile
        if not np.isnan(spread_p90) and spread_val > spread_p90:
            return

        # Increment bar count if in position
        if self.position:
            self.bar_count += 1

        # ============ EXIT LOGIC ============
        if self.position.is_long:
            # Exit: spread stops contracting OR CND flips negative OR time exit
            if spread_chg >= 0 or cnd_val < 0 or self.bar_count >= self.max_hold_bars:
                print(f"🌙 EXIT LONG @ {price:.2f} | spread_chg={spread_chg:.3f} cnd={cnd_val:.2f} bars={self.bar_count}")
                self.position.close()
                self.bar_count = 0
                return

        if self.position.is_short:
            # Exit: spread stops widening OR CND flips positive OR time exit
            if spread_chg <= 0 or cnd_val > 0 or self.bar_count >= self.max_hold_bars:
                print(f"🌙 EXIT SHORT @ {price:.2f} | spread_chg={spread_chg:.3f} cnd={cnd_val:.2f} bars={self.bar_count}")
                self.position.close()
                self.bar_count = 0
                return

        # ============ ENTRY LOGIC ============
        if self.position:
            return

        # LONG: spread contracting + positive CND + price above EMA
        if (spread_chg < -self.spread_threshold and
            cnd_val > 0 and
            price > ema_val):

            # Position sizing based on |CND| normalized
            if not np.isnan(cnd_med) and cnd_med > 0:
                size_mult = min(abs(cnd_val) / cnd_med, self.size_cap_mult)
            else:
                size_mult = 1.0

            # Risk-based sizing
            risk_amount = self.equity * self.risk_pct
            stop_distance = atr_val * 1.5
            if stop_distance > 0:
                position_size = int(round((risk_amount / stop_distance) * size_mult))
            else:
                position_size = 0

            if position_size > 0:
                sl = price - stop_distance
                tp = price + stop_distance * 2
                print(f"🚀 LONG ENTRY @ {price:.2f} | size={position_size} | SL={sl:.2f} TP={tp:.2f} | spread_chg={spread_chg:.3f} cnd={cnd_val:.2f}")
                self.buy(size=position_size, sl=sl, tp=tp)
                self.bar_count = 0
                self.entry_price = price

        # SHORT: spread widening + negative CND + price below EMA
        elif (spread_chg > self.spread_threshold and
              cnd_val < 0 and
              price < ema_val):

            if not np.isnan(cnd_med) and cnd_med > 0:
                size_mult = min(abs(cnd_val) / cnd_med, self.size_cap_mult)
            else:
                size_mult = 1.0

            risk_amount = self.equity * self.risk_pct
            stop_distance = atr_val * 1.5
            if stop_distance > 0:
                position_size = int(round((risk_amount / stop_distance) * size_mult))
            else:
                position_size = 0

            if position_size > 0:
                sl = price + stop_distance
                tp = price - stop_distance * 2
                print(f"🔻 SHORT ENTRY @ {price:.2f} | size={position_size} | SL={sl:.2f} TP={tp:.2f} | spread_chg={spread_chg:.3f} cnd={cnd_val:.2f}")
                self.sell(size=position_size, sl=sl, tp=tp)
                self.bar_count = 0
                self.entry_price = price


# 🌙 Run backtest
print("\n🌙 Moon Dev launching backtest...")
bt = Backtest(data, LiquidityDeltaFlow, cash=1_000_000, commission=0.001, exclusive_orders=True)
stats = bt.run()
print(stats)
print(stats._strategy)
print("\n✨ Moon Dev backtest complete! 🚀🌙")