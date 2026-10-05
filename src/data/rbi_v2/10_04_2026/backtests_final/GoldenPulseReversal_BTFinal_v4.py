import pandas as pd
import numpy as np
import talib
from backtesting import Backtest, Strategy

# ============================================================
# 🌙 GoldenPulse Reversal Strategy - Moon Dev Backtest 🌙
# ============================================================

print("🌙✨ Moon Dev Backtest AI warming up the engines... ✨🌙")

# Load data
data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Map to proper case
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

# Ensure numeric types
for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')

data = data.dropna()

print(f"🚀 Data loaded: {len(data)} rows | {data.index[0]} -> {data.index[-1]}")


class GoldenPulseReversal(Strategy):
    # Strategy parameters
    ma_short = 50
    ma_long = 200
    ema_trigger = 20
    atr_period = 14
    atr_stop_mult = 1.5
    risk_pct = 0.02
    rr_ratio = 2.0

    def init(self):
        print("🌙 Initializing GoldenPulse indicators...")
        close = self.data.Close
        high = self.data.High
        low = self.data.Low

        # Trend MAs
        self.ma50 = self.I(talib.SMA, close, timeperiod=self.ma_short)
        self.ma200 = self.I(talib.SMA, close, timeperiod=self.ma_long)
        self.ema20 = self.I(talib.EMA, close, timeperiod=self.ema_trigger)

        # ATR for risk management
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        # Spread = MA50 - MA200 (computed via numpy arrays to avoid pandas ops)
        def _spread():
            ma50_arr = np.asarray(self.ma50)
            ma200_arr = np.asarray(self.ma200)
            return ma50_arr - ma200_arr
        self.spread = self.I(_spread, name="Spread")

        print("✨ Indicators ready: MA50, MA200, EMA20, ATR, Spread")

    def next(self):
        # Need enough history
        if len(self.data) < self.ma_long + 2:
            return

        # Current and previous values
        spread_now = self.spread[-1]
        spread_prev = self.spread[-2]
        ema_now = self.ema20[-1]
        ema_prev = self.ema20[-2]
        ma50_now = self.ma50[-1]
        ma200_now = self.ma200[-1]
        price = self.data.Close[-1]
        atr_now = self.atr[-1]

        # Skip if any NaN
        if (np.isnan(spread_now) or np.isnan(spread_prev) or
                np.isnan(ema_now) or np.isnan(ema_prev) or
                np.isnan(atr_now) or
                np.isnan(ma50_now) or np.isnan(ma200_now)):
            return

        # ===================== ENTRY =====================
        if not self.position:
            # Bullish crossover: spread crosses above EMA20
            bullish_cross = (spread_prev <= ema_prev) and (spread_now > ema_now)
            # Confirmation: MA50 > MA200 (bullish structure)
            trend_ok = ma50_now > ma200_now

            if bullish_cross and trend_ok:
                # Risk-based position sizing as fraction of equity
                stop_distance = atr_now * self.atr_stop_mult
                if stop_distance > 0 and price > stop_distance:
                    # Fraction of equity sized so that stop loss = risk_pct of equity
                    size_frac = (self.equity * self.risk_pct) / (stop_distance * price)
                    size_frac = float(np.clip(size_frac, 0.001, 0.99))
                    sl = price - stop_distance
                    tp = price + stop_distance * self.rr_ratio
                    print(f"🌙🚀 LONG ENTRY | Price={price:.2f} | Spread={spread_now:.2f} > EMA20={ema_now:.2f} | SL={sl:.2f} TP={tp:.2f} | Size={size_frac:.4f}")
                    self.buy(size=size_frac, sl=sl, tp=tp)

        # ===================== EXIT =====================
        else:
            # Exit when spread crosses back below EMA20
            bearish_cross = (spread_prev >= ema_prev) and (spread_now < ema_now)
            # Trend failure exit
            trend_fail = ma50_now < ma200_now

            if bearish_cross or trend_fail:
                reason = "Spread<EMA20" if bearish_cross else "MA50<MA200"
                print(f"🌙🛑 EXIT ({reason}) | Price={price:.2f} | Spread={spread_now:.2f}")
                self.position.close()


# ============================================================
# 🚀 Run the backtest
# ============================================================
print("🌙✨ Launching GoldenPulse Reversal backtest... ✨🌙")

bt = Backtest(
    data,
    GoldenPulseReversal,
    cash=1_000_000,
    commission=0.002,
    exclusive_orders=True
)

stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙✨ Backtest complete. Moon Dev out! ✨🌙")