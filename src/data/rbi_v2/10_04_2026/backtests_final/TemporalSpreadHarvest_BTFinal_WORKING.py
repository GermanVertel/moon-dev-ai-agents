import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 🌙 MOON DEV - TEMPORAL SPREAD HARVEST (SPY/INDEX PROXY) 🌙
# ============================================================

class TemporalSpreadHarvest(Strategy):
    # --- Strategy Parameters ---
    risk_pct = 0.015
    sd_mult = 1.0
    spread_width_mult = 0.5
    min_credit_ratio = 0.30
    profit_target = 0.50
    stop_loss_value = 1.50
    pip_buffer = 0.5
    max_hold_bars = 2
    iv_lookback = 20

    def init(self):
        # 🌙 Moon Dev indicators ✨
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
            price_move = self.entry_price - price
            spread_value = self.credit + 0.25 * max(price_move, 0) - 0.05 * max(-price_move, 0)
            spread_value = max(spread_value, 0.0)

            # 🎯 Profit target
            if spread_value <= self.credit * self.profit_target:
                self.position.close()
                print(f"🌙💰 PROFIT TARGET HIT | value={spread_value:.4f} "
                      f"credit={self.credit:.4f} | bars={bars_held}")
                self._reset()
                return

            # 🛑 Value-based stop
            if spread_value >= self.credit * self.stop_loss_value:
                self.position.close()
                print(f"🌙🛑 VALUE STOP | value={spread_value:.4f} "
                      f"credit={self.credit:.4f} | bars={bars_held}")
                self._reset()
                return

            # 🛑 Price-based stop
            if price <= self.short_strike + self.pip_buffer:
                self.position.close()
                print(f"🌙⚠️ PRICE STOP | price={price:.2f} "
                      f"short_strike={self.short_strike:.2f} | bars={bars_held}")
                self._reset()
                return

            # ⏰ Time stop
            if bars_held >= self.max_hold_bars:
                self.position.close()
                print(f"🌙⏰ TIME STOP | bars={bars_held} | "
                      f"price={price:.2f}")
                self._reset()
                return
            return

        # ------------------ ENTRY LOGIC ------------------
        try:
            current_hour = self.data.index[-1].hour
        except Exception:
            current_hour = 8

        if current_hour not in (7, 8, 9):
            return

        sd_move = self.sd_mult * atr
        short_strike = price - sd_move
        width = self.spread_width_mult * sd_move
        long_strike = short_strike - width

        credit = 0.35 * width

        if credit < self.min_credit_ratio * width:
            return

        max_loss_per_unit = width - credit
        if max_loss_per_unit <= 0:
            return

        # Position sizing: fraction of equity (0 < size < 1)
        # Risk fraction = risk_pct / max_loss_per_unit relative to price
        risk_fraction = (self.equity * self.risk_pct) / (max_loss_per_unit * price)
        # Clamp to valid fraction range
        size = min(max(risk_fraction, 0.01), 0.95)

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
              f"credit={credit:.4f} size={size:.4f}")

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