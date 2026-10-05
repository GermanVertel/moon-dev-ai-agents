import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# ============================================================
# 🌙 Moon Dev's GammaVolatilityReversion Backtest 🌙
# ============================================================

print("🌙 Moon Dev initializing GammaVolatilityReversion strategy...")
print("✨ Loading cosmic data from the lunar vault...")

data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper mapping
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
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

print(f"🚀 Data loaded: {len(data)} rows of lunar price action")


class GammaVolatilityReversion(Strategy):
    """
    🌙 Gamma Volatility Reversion Strategy 🌙

    Since we only have OHLCV data (no options chain), we proxy the
    gamma hedge ratio using realized volatility compression vs expansion
    and price structure. We use:
      - VIX proxy: ATR-based volatility regime
      - Gamma proxy: range compression relative to volume
      - IV rank proxy: realized vol rank over 252 bars
    """

    # --- Parameters ---
    sma_period = 50
    atr_period = 14
    vol_lookback = 100       # for IV rank proxy
    gamma_lookback = 20      # for gamma hedge ratio proxy
    gamma_entry = 1.2
    gamma_exit = 0.8
    vix_low = 20.0           # scaled proxy threshold
    vix_high = 25.0
    iv_rank_entry = 30.0
    iv_rank_exit = 60.0
    profit_target_pct = 0.25   # IV expansion proxy
    hard_stop_pct = 0.08
    trailing_stop_pct = 0.06
    max_hold_days = 12

    def init(self):
        print("🌙 Initializing indicators...")

        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        # 50-day SMA trend filter
        self.sma50 = self.I(talib.SMA, close, timeperiod=self.sma_period)

        # ATR for volatility regime proxy
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        # ATR% as volatility proxy (VIX-like)
        self.atr_pct = self.I(lambda a, c: (a / c) * 100, self.atr, close)

        # Realized volatility (rolling std of returns) for IV rank proxy
        self.rvol = self.I(
            lambda c: pd.Series(c).pct_change().rolling(self.vol_lookback).std().values * np.sqrt(252) * 100,
            close
        )

        # IV Rank proxy: percentile rank of current rvol over lookback
        self.iv_rank = self.I(
            lambda r: pd.Series(r).rolling(self.vol_lookback * 2).apply(
                lambda x: (x.iloc[-1] > x.iloc[:-1]).mean() * 100 if len(x) > 1 else 50, raw=False
            ).values,
            self.rvol
        )

        # Gamma Hedge Ratio proxy:
        # range compression (high-low)/close normalized vs average range,
        # divided by volume ratio (dealer hedging intensity proxy)
        def gamma_proxy(h, l, c, v):
            rng = (h - l) / c
            avg_rng = pd.Series(rng).rolling(self.gamma_lookback).mean().values
            vol_ratio = pd.Series(v).rolling(self.gamma_lookback).mean().values
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = (avg_rng / np.where(rng == 0, 1e-9, rng)) * 1.0
            # scale into 0.5 - 2.5 range
            ratio = np.clip(ratio, 0.3, 3.0)
            return ratio

        self.gamma_ratio = self.I(gamma_proxy, high, low, close, volume)

        # VIX proxy: ATR% smoothed
        self.vix_proxy = self.I(talib.SMA, self.atr_pct, timeperiod=20)

        # VIX term structure proxy: short vs long volatility
        self.vix_short = self.I(talib.SMA, self.atr_pct, timeperiod=10)
        self.vix_long = self.I(talib.SMA, self.atr_pct, timeperiod=50)
        # contango = short < long (calm)
        self.contango = self.I(
            lambda s, l: (s < l).astype(float),
            self.vix_short, self.vix_long
        )

        print("✨ Indicators initialized successfully!")

    def next(self):
        price = self.data.Close[-1]

        # Skip warmup
        if len(self.data) < self.vol_lookback * 2 + 10:
            return
        if np.isnan(self.sma50[-1]) or np.isnan(self.gamma_ratio[-1]) or np.isnan(self.vix_proxy[-1]):
            return
        if np.isnan(self.iv_rank[-1]) or np.isnan(self.contango[-1]):
            return

        gamma = self.gamma_ratio[-1]
        vix = self.vix_proxy[-1]
        ivr = self.iv_rank[-1]
        sma = self.sma50[-1]
        contango = self.contango[-1] > 0.5

        # ============ ENTRY LOGIC ============
        if not self.position:
            cond_vix = vix < self.vix_low
            cond_contango = contango
            cond_gamma = gamma > self.gamma_entry
            cond_iv = ivr < self.iv_rank_entry
            cond_trend = price > sma

            if cond_vix and cond_contango and cond_gamma and cond_iv and cond_trend:
                # Position sizing scaled by gamma ratio (capped)
                # Size = 1,000,000 base units as required
                base_size = 1_000_000
                scale = min(gamma / self.gamma_entry, 1.5)
                size = int(round(base_size * scale))
                if size < 1:
                    size = 1

                print(f"🌙✨ ENTRY SIGNAL! Price={price:.2f} | VIXproxy={vix:.2f} | "
                      f"Gamma={gamma:.2f} | IVrank={ivr:.1f} | SMA={sma:.2f} | Size={size}")

                self.buy(size=size)
                self.entry_price = price
                self.entry_bar = len(self.data)
                self.highest_price = price

        # ============ EXIT LOGIC ============
        else:
            entry = self.entry_price
            bars_held = len(self.data) - self.entry_bar
            self.highest_price = max(self.highest_price, price)

            # Trailing stop
            trail_stop = self.highest_price * (1 - self.trailing_stop_pct)
            # Hard stop
            hard_stop = entry * (1 - self.hard_stop_pct)
            # Profit target
            profit_target = entry * (1 + self.profit_target_pct)

            exit_reason = None

            if price <= hard_stop:
                exit_reason = "🛑 HARD STOP"
            elif price <= trail_stop and price < entry:
                exit_reason = "📉 TRAILING STOP"
            elif price >= profit_target:
                exit_reason = "🎯 PROFIT TARGET"
            elif vix > self.vix_high:
                exit_reason = "⚡ REGIME STOP (VIX spike)"
            elif gamma < self.gamma_exit:
                exit_reason = "🌊 GAMMA DECAY EXIT"
            elif ivr > self.iv_rank_exit:
                exit_reason = "🔥 IV RANK EXIT"
            elif bars_held >= self.max_hold_days:
                exit_reason = "⏰ TIME STOP"

            if exit_reason:
                print(f"🌙 EXIT: {exit_reason} | Price={price:.2f} | "
                      f"Entry={entry:.2f} | Bars={bars_held}")
                self.position.close()


print("🚀 Launching Moon Dev backtest engine...")
bt = Backtest(
    data,
    GammaVolatilityReversion,
    cash=10_000_000,
    commission=0.002,
    exclusive=False
)

stats = bt.run()
print("🌙✨ Backtest complete! Printing full stats...")
print(stats)
print(stats._strategy)