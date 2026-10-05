import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 🌙 MOON DEV - TEMPORAL SPREAD HARVEST (SPY/INDEX PROXY) 🌙
# ============================================================
# Note: backtesting.py works on the underlying price series.
# We simulate the put credit spread payoff using the underlying
# proxy: short OTM put ~1 SD below, long put further OTM.
# Since options data isn't available, we approximate the
# spread P&L using the underlying's move relative to the
# short strike (delta-based approximation).
# ============================================================

class TemporalSpreadHarvest(Strategy):
    # --- Strategy Parameters ---
    risk_pct = 0.015          # 1.5% equity risk per trade
    sd_mult = 1.0             # 1 SD OTM for short put
    spread_width_mult = 0.5   # long put further OTM
    min_credit_ratio = 0.30   # credit >= 30% of width
    profit_target = 0.50      # close at 50% of credit
    stop_loss_value = 1.50    # value-based stop at 150% of credit
    pip_buffer = 0.5          # pip buffer above short strike
    max_hold_bars = 2         # 30 min = 2 bars of 15m
    iv_lookback = 20          # for IV proxy (ATR-based)

    def init(self):
        # 🌙 Moon Dev indicators — all via talib, wrapped in self.I ✨
        self.ema = self.I(talib.EMA, self.data.Close, timeperiod=20, name='EMA20')
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close,
                          timeperiod=14, name='ATR14')
        self.sma_iv = self.I(talib.SMA, self.atr, timeperiod=self.iv_lookback,
                             name='SMA_ATR')

        # Trade state
        self.entry_bar = None
        self.short_strike = None
        self.long_strike = None
        self.credit = None
        self.width = None
        self.entry_price = None

        print("🌙✨ TemporalSpreadHarvest initialized — Moon Dev ready! 🚀")

    def next(self):
        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        atr = self.atr[-1]

        if np.isnan(atr) or atr <= 0:
            return

        # ------------------ MANAGE OPEN POSITION ------------------
        if self.position:
            bars_held = len(self.data) - self.entry_bar
            # Approximate current spread value using underlying move
            # Short put loses value as underlying drops toward strike
            price_move = self.entry_price - price  # positive if price drops
            # Spread value approximation: credit + delta * adverse move
            # delta proxy ~ 0.25 for 1SD OTM put
            spread_value = self.credit + 0.25 * max(price_move, 0) - 0.05 * max(-price_move, 0)
            spread_value = max(spread_value, 0.0)

            # 🎯 Profit target: spread value <= 50% of credit
            if spread_value <= self.credit * self.profit_target:
                self.position.close()
                print(f"🌙💰 PROFIT TARGET HIT | value={spread_value:.4f} "
                      f"credit={self.credit:.4f} | bars={bars_held}")
                self._reset()
                return

            # 🛑 Value-based stop: spread value >= 150% of credit
            if spread_value >= self.credit * self.stop_loss_value:
                self.position.close()
                print(f"🌙🛑 VALUE STOP | value={spread_value:.4f} "
                      f"credit={self.credit:.4f} | bars={bars_held}")
                self._reset()
                return

            # 🛑 Price-based stop: price touches short strike + pip
            if price <= self.short_strike + self.pip_buffer:
                self.position.close()
                print(f"🌙⚠️ PRICE STOP | price={price:.2f} "
                      f"short_strike={self.short_strike:.2f} | bars={bars_held}")
                self._reset()
                return

            # ⏰ Time stop: 30 minutes = 2 bars of 15m
            if bars_held >= self.max_hold_bars:
                self.position.close()
                print(f"🌙⏰ TIME STOP | bars={bars_held} | "
                      f"price={price:.2f}")
                self._reset()
                return
            return

        # ------------------ ENTRY LOGIC ------------------
        # Only enter during European open window (07:00–10:00 UTC proxy)
        # 15m bars: hour 7,8,9 UTC
        try:
            current_hour = self.data.index[-1].hour
        except Exception:
            current_hour = 8

        if current_hour not in (7, 8, 9):
            return

        # Short put ~1 SD OTM (using ATR as IV proxy)
        sd_move = self.sd_mult * atr
        short_strike = price - sd_move
        width = self.spread_width_mult * sd_move
        long_strike = short_strike - width

        # Estimate credit: roughly 30-40% of width for 1SD short put
        credit = 0.35 * width

        # Credit requirement filter
        if credit < self.min_credit_ratio * width:
            return

        # Risk = width - credit (max defined loss)
        max_loss_per_unit = width - credit
        if max_loss_per_unit <= 0:
            return

        # Position sizing: risk_pct of equity / max_loss_per_unit
        risk_amount = self.equity * self.risk_pct
        size = int(round(risk_amount / max_loss_per_unit))
        if size < 1:
            size = 1

        # Enter long (bullish bias — selling puts)
        self.buy(size=size)
        self.entry_bar = len(self.data)
        self.short_strike = short_strike
        self.long_strike = long_strike
        self.credit = credit
        self.width = width
        self.entry_price = price

        print(f"🚀🌙 ENTERED BULL PUT SPREAD | price={price:.2f} "
              f"short={short_strike:.2f} long={long_strike:.2f} "
              f"credit={credit:.4f} size={size}")

    def _reset(self):
        self.entry_bar = None
        self.short_strike = None
        self.long_strike = None
        self.credit = None
        self.width = None
        self.entry_price = None


# ============================================================
# DATA LOADING
# ============================================================
data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper case mapping
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

# Datetime index
if 'datetime' in data.columns:
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')

data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

print(f"🌙 Loaded {len(data)} bars from Moon Dev data vault 📊")

# ============================================================
# RUN BACKTEST
# ============================================================
bt = Backtest(
    data,
    TemporalSpreadHarvest,
    cash=1_000_000,
    commission=0.0002,
    exclusive_orders=True
)

stats = bt.run()
print(stats)
print(stats._strategy)