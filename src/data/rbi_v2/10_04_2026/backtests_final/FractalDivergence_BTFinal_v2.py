import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Load data
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Map to proper column names
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')

print("🌙 Moon Dev FractalDivergence Backtest Loading... ✨")
print(f"📊 Data loaded: {len(data)} bars from {data.index[0]} to {data.index[-1]}")
print(f"🚀 Columns: {list(data.columns)}")


class FractalDivergence(Strategy):
    # Strategy parameters
    rsi_period = 14
    ema_fast_period = 50
    ema_slow_period = 200
    atr_period = 14
    fractal_window = 2  # 2 bars each side = 5-bar fractal
    rr_ratio = 2.0
    risk_pct = 0.02
    sl_buffer_atr = 0.5
    max_concurrent = 3

    def init(self):
        print("🌙 Initializing Moon Dev FractalDivergence indicators... ✨")

        # RSI
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=self.rsi_period, name='RSI')

        # EMAs
        self.ema_fast = self.I(talib.EMA, self.data.Close, timeperiod=self.ema_fast_period, name='EMA50')
        self.ema_slow = self.I(talib.EMA, self.data.Close, timeperiod=self.ema_slow_period, name='EMA200')

        # ATR
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close,
                          timeperiod=self.atr_period, name='ATR')

        # Fractals using talib MAX/MIN (5-bar pattern)
        self.fractal_high = self.I(talib.MAX, self.data.High,
                                   timeperiod=2 * self.fractal_window + 1, name='FractalHigh')
        self.fractal_low = self.I(talib.MIN, self.data.Low,
                                  timeperiod=2 * self.fractal_window + 1, name='FractalLow')

        # Track state
        self.last_fractal_high_price = None
        self.last_fractal_high_rsi = None
        self.last_fractal_high_idx = None
        self.last_fractal_low_price = None
        self.last_fractal_low_rsi = None
        self.last_fractal_low_idx = None

        # Pending breakout levels
        self.long_trigger_level = None
        self.long_sl = None
        self.long_tp = None
        self.short_trigger_level = None
        self.short_sl = None
        self.short_tp = None

        # Track active trades for R-multiple management
        self.entry_price = None
        self.initial_sl = None
        self.breakeven_set = False

        print("🌙 Indicators ready! RSI, EMA50/200, ATR, Fractals loaded 🚀")

    def next(self):
        # Need enough bars for indicators
        if len(self.data) < self.ema_slow_period + 5:
            return

        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        rsi_val = self.rsi[-1]
        atr_val = self.atr[-1]
        ema_f = self.ema_fast[-1]
        ema_s = self.ema_slow[-1]

        if np.isnan(rsi_val) or np.isnan(atr_val) or np.isnan(ema_s):
            return

        # Detect fractal formation at index -1 - fractal_window (the center of the 5-bar pattern)
        center = -1 - self.fractal_window

        # Check for up fractal at center (swing high)
        up_fractal = False
        down_fractal = False

        if len(self.data) > 2 * self.fractal_window + 1:
            # Compute window bounds safely using absolute indices
            n = len(self.data)
            center_abs = n - 1 - self.fractal_window
            start_abs = center_abs - self.fractal_window
            end_abs = center_abs + self.fractal_window + 1

            if start_abs >= 0 and end_abs <= n:
                window_high = max(self.data.High[start_abs:end_abs])
                window_low = min(self.data.Low[start_abs:end_abs])

                if self.data.High[center] >= window_high:
                    up_fractal = True
                if self.data.Low[center] <= window_low:
                    down_fractal = True

        # --- Fractal + RSI Divergence Detection ---
        if up_fractal:
            fractal_price = self.data.High[center]
            fractal_rsi = self.rsi[center]
            # Bearish divergence: price higher high, RSI lower high
            if (self.last_fractal_high_price is not None and
                    fractal_price > self.last_fractal_high_price and
                    fractal_rsi < self.last_fractal_high_rsi):
                print(f"🌙✨ BEARISH DIVERGENCE detected! Price HH {fractal_price:.2f} > {self.last_fractal_high_price:.2f}, "
                      f"RSI LH {fractal_rsi:.2f} < {self.last_fractal_high_rsi:.2f}")
                # Set short trigger on break below fractal low
                self.short_trigger_level = self.data.Low[center]
                sl = fractal_price + self.sl_buffer_atr * atr_val
                risk = sl - self.short_trigger_level
                self.short_sl = sl
                self.short_tp = self.short_trigger_level - self.rr_ratio * risk
            self.last_fractal_high_price = fractal_price
            self.last_fractal_high_rsi = fractal_rsi

        if down_fractal:
            fractal_price = self.data.Low[center]
            fractal_rsi = self.rsi[center]
            # Bullish divergence: price lower low, RSI higher low
            if (self.last_fractal_low_price is not None and
                    fractal_price < self.last_fractal_low_price and
                    fractal_rsi > self.last_fractal_low_rsi):
                print(f"🌙✨ BULLISH DIVERGENCE detected! Price LL {fractal_price:.2f} < {self.last_fractal_low_price:.2f}, "
                      f"RSI HL {fractal_rsi:.2f} > {self.last_fractal_low_rsi:.2f}")
                # Set long trigger on break above fractal high
                self.long_trigger_level = self.data.High[center]
                sl = fractal_price - self.sl_buffer_atr * atr_val
                risk = self.long_trigger_level - sl
                self.long_sl = sl
                self.long_tp = self.long_trigger_level + self.rr_ratio * risk
            self.last_fractal_low_price = fractal_price
            self.last_fractal_low_rsi = fractal_rsi

        # --- Entry Logic (breakout confirmation) ---
        # Long entry: price above EMA200, break above fractal high
        if (not self.position and
                self.long_trigger_level is not None and
                price > self.long_trigger_level and
                price > ema_s and
                rsi_val < 70 and
                len(self.trades) < self.max_concurrent):
            # Position sizing: risk 2% of equity
            risk_per_unit = self.long_trigger_level - self.long_sl
            if risk_per_unit > 0:
                risk_amount = self.equity * self.risk_pct
                size = int(round(risk_amount / risk_per_unit))
                if size > 0:
                    # Ensure SL < LIMIT < TP for long orders
                    sl_val = self.long_sl
                    tp_val = self.long_tp
                    if sl_val < price < tp_val:
                        print(f"🚀🌙 LONG ENTRY! Price {price:.2f} broke fractal high {self.long_trigger_level:.2f} | "
                              f"SL={sl_val:.2f} TP={tp_val:.2f} Size={size}")
                        self.buy(size=size, sl=sl_val, tp=tp_val)
                        self.entry_price = price
                        self.initial_sl = sl_val
                        self.breakeven_set = False
                        self.long_trigger_level = None
                    else:
                        print(f"🌙⚠️ Long order skipped: SL ({sl_val:.2f}) < LIMIT ({price:.2f}) < TP ({tp_val:.2f}) not satisfied")

        # Short entry: price below EMA200, break below fractal low
        if (not self.position and
                self.short_trigger_level is not None and
                price < self.short_trigger_level and
                price < ema_s and
                rsi_val > 30 and
                len(self.trades) < self.max_concurrent):
            risk_per_unit = self.short_sl - self.short_trigger_level
            if risk_per_unit > 0:
                risk_amount = self.equity * self.risk_pct
                size = int(round(risk_amount / risk_per_unit))
                if size > 0:
                    # Ensure TP < LIMIT < SL for short orders
                    sl_val = self.short_sl
                    tp_val = self.short_tp
                    if tp_val < price < sl_val:
                        print(f"🔻🌙 SHORT ENTRY! Price {price:.2f} broke fractal low {self.short_trigger_level:.2f} | "
                              f"SL={sl_val:.2f} TP={tp_val:.2f} Size={size}")
                        self.sell(size=size, sl=sl_val, tp=tp_val)
                        self.entry_price = price
                        self.initial_sl = sl_val
                        self.breakeven_set = False
                        self.short_trigger_level = None
                    else:
                        print(f"🌙⚠️ Short order skipped: TP ({tp_val:.2f}) < LIMIT ({price:.2f}) < SL ({sl_val:.2f}) not satisfied")

        # --- Trade Management: Move SL to breakeven after 1R profit ---
        if self.position:
            if self.position.is_long and not self.breakeven_set:
                r_dist = self.entry_price - self.initial_sl
                if high >= self.entry_price + r_dist:
                    print(f"🌙✨ Long 1R reached! Moving SL to breakeven {self.entry_price:.2f}")
                    self.position.sl = self.entry_price
                    self.breakeven_set = True
            elif self.position.is_short and not self.breakeven_set:
                r_dist = self.initial_sl - self.entry_price
                if low <= self.entry_price - r_dist:
                    print(f"🌙✨ Short 1R reached! Moving SL to breakeven {self.entry_price:.2f}")
                    self.position.sl = self.entry_price
                    self.breakeven_set = True


print("🌙🚀 Starting Moon Dev FractalDivergence Backtest... ✨")
bt = Backtest(data, FractalDivergence, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)