import pandas as pd
import numpy as np
import talib
from backtesting import Backtest, Strategy

# ── Data Loading & Cleaning ────────────────────────────────────────────────
data = pd.read_csv(
    '/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv'
)

data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

data = data.rename(columns={
    'datetime': 'Datetime',
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume',
})

data['Datetime'] = pd.to_datetime(data['Datetime'])
data = data.set_index('Datetime')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)

print("🌙✨ Data loaded and cleaned! Shape:", data.shape)
print(data.head())


# ── Strategy ───────────────────────────────────────────────────────────────
class VolatilitySqueeze(Strategy):
    atr_period = 14
    bb_period = 20
    bb_std = 2.0
    vol_sma_period = 10
    vol_surge_mult = 2.0
    ema_period = 20
    risk_pct = 0.02
    atr_sl_mult = 1.5
    time_stop_bars = 15

    def init(self):
        # ATR (14)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close,
                          timeperiod=self.atr_period)

        # Bollinger Bands applied to the ATR series
        self.atr_sma = self.I(talib.SMA, self.atr, timeperiod=self.bb_period)
        self.atr_std = self.I(talib.STDDEV, self.atr, timeperiod=self.bb_period, nbdev=1)
        self.atr_upper = self.I(lambda: self.atr_sma + self.bb_std * self.atr_std)
        self.atr_lower = self.I(lambda: self.atr_sma - self.bb_std * self.atr_std)

        # Volume SMA (10)
        self.vol_sma = self.I(talib.SMA, self.data.Volume, timeperiod=self.vol_sma_period)

        # EMA (20) for directional bias
        self.ema = self.I(talib.EMA, self.data.Close, timeperiod=self.ema_period)

        self.entry_bar = None
        print("🌙✨ VolatilitySqueeze indicators initialized! 🚀")

    def next(self):
        # Need at least 2 bars for prev_high/prev_low and enough for indicators
        if len(self.data) < max(self.atr_period, self.bb_period, self.vol_sma_period, self.ema_period) + 2:
            return

        price = self.data.Close[-1]
        prev_high = self.data.High[-2]
        prev_low = self.data.Low[-2]

        atr_val = self.atr[-1]
        atr_lower = self.atr_lower[-1]
        atr_sma = self.atr_sma[-1]
        vol = self.data.Volume[-1]
        vol_avg = self.vol_sma[-1]
        ema_val = self.ema[-1]

        # ── Manage existing position ───────────────────────────────────────
        if self.position:
            # Exit when ATR crosses back above its 20-period MA
            if atr_val > atr_sma:
                print(f"🌙✨ EXIT: ATR({atr_val:.2f}) > ATR_SMA({atr_sma:.2f}) — volatility normalized! Closing. 🚀")
                self.position.close()
                self.entry_bar = None
                return

            # Time stop
            if self.entry_bar is not None and (len(self.data) - self.entry_bar) >= self.time_stop_bars:
                print(f"⏰ EXIT: Time stop hit ({self.time_stop_bars} bars) — expansion thesis failed. Closing. 🌙")
                self.position.close()
                self.entry_bar = None
                return

            return

        # ── Entry logic ────────────────────────────────────────────────────
        if (np.isnan(atr_lower) or np.isnan(atr_sma) or np.isnan(vol_avg)
                or np.isnan(atr_val) or np.isnan(ema_val)):
            return

        squeeze = atr_val < atr_lower
        vol_surge = vol >= self.vol_surge_mult * vol_avg

        if not (squeeze and vol_surge):
            return

        # Directional resolution
        long_signal = price > prev_high and price > ema_val
        short_signal = price < prev_low and price < ema_val

        if long_signal:
            sl = price - self.atr_sl_mult * atr_val
            # Tighter of ATR stop vs squeeze range extreme
            squeeze_low = float(np.min(self.data.Low[-5:]))
            sl = max(sl, squeeze_low)

            risk_per_unit = price - sl
            if risk_per_unit <= 0:
                return

            equity = self.equity
            risk_amount = equity * self.risk_pct
            size = int(round(risk_amount / risk_per_unit))
            if size < 1:
                size = 1

            # Ensure size doesn't exceed affordable units
            max_affordable = int(equity / price)
            if max_affordable < 1:
                return
            size = min(size, max_affordable)

            # Convert size to fraction of equity for backtesting.py
            size_frac = size / max_affordable
            if size_frac >= 1:
                size_frac = 0.99
            if size_frac <= 0:
                size_frac = 0.01

            print(f"🌙🚀 LONG ENTRY: price={price:.2f} | ATR={atr_val:.2f} < BB_lower={atr_lower:.2f} | "
                  f"Vol={vol:.2f} >= 2x avg={vol_avg:.2f} | SL={sl:.2f} | size_frac={size_frac:.4f}")
            self.buy(size=size_frac, sl=sl)
            self.entry_bar = len(self.data)

        elif short_signal:
            sl = price + self.atr_sl_mult * atr_val
            squeeze_high = float(np.max(self.data.High[-5:]))
            sl = min(sl, squeeze_high)

            risk_per_unit = sl - price
            if risk_per_unit <= 0:
                return

            equity = self.equity
            risk_amount = equity * self.risk_pct
            size = int(round(risk_amount / risk_per_unit))
            if size < 1:
                size = 1

            # Ensure size doesn't exceed affordable units
            max_affordable = int(equity / price)
            if max_affordable < 1:
                return
            size = min(size, max_affordable)

            # Convert size to fraction of equity for backtesting.py
            size_frac = size / max_affordable
            if size_frac >= 1:
                size_frac = 0.99
            if size_frac <= 0:
                size_frac = 0.01

            print(f"🌙🔻 SHORT ENTRY: price={price:.2f} | ATR={atr_val:.2f} < BB_lower={atr_lower:.2f} | "
                  f"Vol={vol:.2f} >= 2x avg={vol_avg:.2f} | SL={sl:.2f} | size_frac={size_frac:.4f}")
            self.sell(size=size_frac, sl=sl)
            self.entry_bar = len(self.data)


# ── Run Backtest ───────────────────────────────────────────────────────────
bt = Backtest(data, VolatilitySqueeze, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)