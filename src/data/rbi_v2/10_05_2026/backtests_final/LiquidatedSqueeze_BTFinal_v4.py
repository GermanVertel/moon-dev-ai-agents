import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# ============================================================
# 🌙 MOON DEV - LiquidatedSqueeze Strategy 🌙
# ============================================================

print("🌙 Moon Dev Backtest AI warming up...")
print("🚀 Loading LiquidatedSqueeze strategy...")

data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"

# Load data
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

data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
print(f"✨ Data loaded: {len(data)} bars from {data.index[0]} to {data.index[-1]}")


class LiquidatedSqueeze(Strategy):
    """
    🌙 LiquidatedSqueeze Strategy 🌙
    Short-side, event-driven setup combining:
    - Bollinger Band squeeze (volatility compression)
    - Breakdown below lower BB & squeeze range low
    - Negative funding proxy (downside momentum)
    - Volume profile HVN target
    """

    # --- Parameters ---
    bb_period = 20
    bb_std = 2.0
    squeeze_lookback = 150
    squeeze_percentile = 20  # bottom 20%
    rsi_period = 14
    rsi_threshold = 40
    atr_period = 14
    atr_stop_mult = 1.2
    hvn_lookback = 100
    risk_pct = 0.02  # 2% risk per trade
    time_stop_bars = 20

    def init(self):
        print("🌙 Initializing Moon Dev indicators...")

        close = self.data.Close
        high = self.data.High
        low = self.data.Low

        # Bollinger Bands
        def bb_bands(close_arr):
            upper, middle, lower = talib.BBANDS(
                close_arr, timeperiod=self.bb_period,
                nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0
            )
            return upper, middle, lower

        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            bb_bands, close
        )

        # BandWidth
        def calc_bw(upper, middle, lower):
            return (upper - lower) / middle

        self.bandwidth = self.I(calc_bw, self.bb_upper, self.bb_middle, self.bb_lower)

        # BandWidth rolling percentile threshold
        def rolling_pct(bw, lookback, pct):
            s = pd.Series(bw)
            return s.rolling(lookback, min_periods=lookback).quantile(pct / 100.0).values

        self.bw_threshold = self.I(
            rolling_pct, self.bandwidth, self.squeeze_lookback, self.squeeze_percentile
        )

        # RSI
        self.rsi = self.I(talib.RSI, close, timeperiod=self.rsi_period)

        # ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        # Rolling low (squeeze range low)
        self.range_low = self.I(talib.MIN, low, timeperiod=self.squeeze_lookback)

        # Rolling high (squeeze range high) for stop reference
        self.range_high = self.I(talib.MAX, high, timeperiod=self.squeeze_lookback)

        # HVN proxy: rolling VWAP-like volume weighted price for target
        def hvn(close_arr, vol_arr, lookback):
            c = pd.Series(close_arr)
            v = pd.Series(vol_arr)
            num = (c * v).rolling(lookback, min_periods=lookback).sum()
            den = v.rolling(lookback, min_periods=lookback).sum()
            return (num / den).values

        self.hvn = self.I(hvn, close, self.data.Volume, self.hvn_lookback)

        # Funding proxy: use price momentum (downside momentum = negative funding)
        # Z-score of returns as a proxy for funding z-score
        def ret_zscore(close_arr, lookback=30):
            c = pd.Series(close_arr)
            ret = c.pct_change()
            m = ret.rolling(lookback, min_periods=lookback).mean()
            s = ret.rolling(lookback, min_periods=lookback).std()
            return ((ret - m) / s).values

        self.ret_z = self.I(ret_zscore, close, 30)

        print("✨ Indicators ready! Let's squeeze some shorts 🚀")

    def next(self):
        # Need enough data
        if len(self.data) < self.squeeze_lookback + 5:
            return

        price = self.data.Close[-1]

        # Skip if any indicator NaN
        vals = [self.bandwidth[-1], self.bw_threshold[-1], self.bb_lower[-1],
                self.range_low[-1], self.rsi[-1], self.atr[-1], self.hvn[-1],
                self.ret_z[-1]]
        if any(pd.isna(v) or np.isnan(v) for v in vals):
            return

        # ---------------- ENTRY LOGIC (SHORT) ----------------
        if not self.position:
            # 1. Squeeze active
            squeeze_active = self.bandwidth[-1] <= self.bw_threshold[-1]

            # 2. Breakdown: close below lower BB AND below squeeze range low
            breakdown = (price < self.bb_lower[-1]) and (price < self.range_low[-1])

            # 3. Liquidation alignment proxy
            liq_align = price <= self.range_low[-1]

            # 4. Funding confirmation proxy: negative momentum z-score
            funding_conf = self.ret_z[-1] < -1.0

            # 5. Momentum filter: RSI < 40
            momentum_ok = self.rsi[-1] < self.rsi_threshold

            if squeeze_active and breakdown and liq_align and funding_conf and momentum_ok:
                # --- Risk management ---
                stop_price = price + self.atr_stop_mult * self.atr[-1]
                # Target = nearest HVN above entry (mean reversion level)
                target_price = self.hvn[-1]
                if target_price <= price:
                    target_price = price + 2.0 * self.atr[-1]

                risk_per_unit = stop_price - price
                if risk_per_unit <= 0:
                    return

                equity = self.equity
                risk_amount = equity * self.risk_pct
                position_size = int(round(risk_amount / risk_per_unit))
                if position_size <= 0:
                    return

                # Ensure size is valid: cap by affordable units at current price
                max_affordable = int(equity / price)
                if position_size > max_affordable:
                    position_size = max_affordable
                if position_size <= 0:
                    return

                print(f"🌙 SHORT SIGNAL! Price={price:.2f} | BW={self.bandwidth[-1]:.4f} "
                      f"<= {self.bw_threshold[-1]:.4f} | RSI={self.rsi[-1]:.1f} | "
                      f"Z={self.ret_z[-1]:.2f} | Stop={stop_price:.2f} | "
                      f"Target(HVN)={target_price:.2f} | Size={position_size} 🚀")

                self.sell(size=position_size, sl=stop_price, tp=target_price)
                self.entry_bar = len(self.data)
                self.entry_price = price

        # ---------------- EXIT LOGIC ----------------
        else:
            # Time stop
            bars_in_trade = len(self.data) - getattr(self, 'entry_bar', len(self.data))
            if bars_in_trade >= self.time_stop_bars:
                print(f"⏰ Time stop hit after {bars_in_trade} bars — closing short 🌙")
                self.position.close()
                return

            # Hard invalidation: close back above lower BB
            if price > self.bb_lower[-1]:
                print(f"🛑 Invalidation: price {price:.2f} back above lower BB "
                      f"{self.bb_lower[-1]:.2f} — closing short")
                self.position.close()
                return

            # Funding normalization proxy: momentum z back to neutral
            if self.ret_z[-1] > 0.5:
                print(f"💤 Funding normalized (Z={self.ret_z[-1]:.2f}) — closing short")
                self.position.close()
                return


# ============================================================
# 🚀 RUN BACKTEST
# ============================================================
print("🌙 Launching Moon Dev Backtest...")
bt = Backtest(
    data,
    LiquidatedSqueeze,
    cash=1_000_000,
    commission=0.001,
    trade_on_close=True,
)

stats = bt.run()
print(stats)
print(stats._strategy)
print("✨ Moon Dev backtest complete! 🌙🚀")