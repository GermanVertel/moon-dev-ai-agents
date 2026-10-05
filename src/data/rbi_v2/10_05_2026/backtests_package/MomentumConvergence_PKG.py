import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# ============================================================
# 🌙 Moon Dev's MomentumConvergence Backtest 🌙
# ============================================================

print("🌙✨ Moon Dev Backtest Engine Starting... Loading cosmic data! ✨🌙")

# Load data
data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean columns
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

# Ensure datetime index
if 'datetime' in data.columns:
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')

data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
print(f"🚀 Data loaded: {len(data)} rows from the lunar archives!")


class MomentumConvergence(Strategy):
    # Strategy parameters
    rsi_period = 14
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    atr_period = 14
    breakout_period = 20
    vol_ma_period = 20
    ema_period = 10

    rsi_long_thresh = 55
    rsi_short_thresh = 45
    rsi_overbought = 75
    rsi_oversold = 25

    atr_sl_mult = 1.0
    atr_tp_mult = 2.0
    risk_pct = 0.02
    volume_mult = 1.5
    time_exit_bars = 10

    def init(self):
        print("🌙 Initializing Moon Dev indicators... ✨")
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        # RSI
        self.rsi = self.I(talib.RSI, close, timeperiod=self.rsi_period, name="RSI")

        # MACD
        self.macd, self.macd_signal_line, self.macd_hist = self.I(
            talib.MACD, close,
            fastperiod=self.macd_fast,
            slowperiod=self.macd_slow,
            signalperiod=self.macd_signal,
            name="MACD"
        )

        # ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period, name="ATR")

        # Breakout levels
        self.hh = self.I(talib.MAX, high, timeperiod=self.breakout_period, name="HH20")
        self.ll = self.I(talib.MIN, low, timeperiod=self.breakout_period, name="LL20")

        # Volume MA
        self.vol_ma = self.I(talib.SMA, volume, timeperiod=self.vol_ma_period, name="VolMA")

        # EMA trail
        self.ema = self.I(talib.EMA, close, timeperiod=self.ema_period, name="EMA10")

        print("🚀 Indicators ready for liftoff! 🌙")

    def next(self):
        price = self.data.Close[-1]

        # Skip if not enough data
        if len(self.data) < self.breakout_period + 5:
            return

        # Check for NaN
        if (np.isnan(self.rsi[-1]) or np.isnan(self.macd[-1]) or
                np.isnan(self.macd_signal_line[-1]) or np.isnan(self.macd_hist[-1]) or
                np.isnan(self.atr[-1]) or np.isnan(self.hh[-1]) or
                np.isnan(self.ll[-1]) or np.isnan(self.vol_ma[-1])):
            return

        # ---------------- MANAGE OPEN POSITION ----------------
        if self.position:
            entry_price = self.position.entry_price
            is_long = self.position.is_long
            bars_held = len(self.data) - self.trades[-1].entry_bar if self.trades else 0

            # Momentum exit: RSI crosses back below 50 (long) or above 50 (short)
            if is_long and self.rsi[-1] < 50:
                print(f"🌙 Momentum Exit LONG @ {price:.2f} | RSI={self.rsi[-1]:.2f}")
                self.position.close()
                return
            if not is_long and self.rsi[-1] > 50:
                print(f"🌙 Momentum Exit SHORT @ {price:.2f} | RSI={self.rsi[-1]:.2f}")
                self.position.close()
                return

            # MACD histogram flip
            if is_long and self.macd_hist[-1] < 0 and self.macd_hist[-2] >= 0:
                print(f"🌙 MACD Flip Exit LONG @ {price:.2f}")
                self.position.close()
                return
            if not is_long and self.macd_hist[-1] > 0 and self.macd_hist[-2] <= 0:
                print(f"🌙 MACD Flip Exit SHORT @ {price:.2f}")
                self.position.close()
                return

            # Time-based exit
            if bars_held >= self.time_exit_bars:
                # Only exit if no favorable move
                if is_long and price <= entry_price:
                    print(f"🌙 Time Exit LONG @ {price:.2f}")
                    self.position.close()
                    return
                if not is_long and price >= entry_price:
                    print(f"🌙 Time Exit SHORT @ {price:.2f}")
                    self.position.close()
                    return

            return

        # ---------------- ENTRY LOGIC ----------------
        # Volume confirmation
        vol_ok = self.data.Volume[-1] > self.volume_mult * self.vol_ma[-1]

        # RSI rising/falling over last 3 bars
        rsi_rising = self.rsi[-1] > self.rsi[-2] > self.rsi[-3]
        rsi_falling = self.rsi[-1] < self.rsi[-2] < self.rsi[-3]

        # MACD histogram expanding
        hist_expanding = self.macd_hist[-1] > self.macd_hist[-2]
        hist_contracting = self.macd_hist[-1] < self.macd_hist[-2]

        # Minimum ATR filter (avoid choppy markets) - ATR must be > 0.1% of price
        atr_ok = self.atr[-1] > price * 0.001

        # LONG conditions
        breakout_long = price > self.hh[-2]  # Breakout above prior 20-bar high
        rsi_long_ok = self.rsi[-1] > self.rsi_long_thresh and rsi_rising
        macd_long_ok = self.macd[-1] > self.macd_signal_line[-1] and self.macd_hist[-1] > 0 and hist_expanding
        not_overbought = self.rsi[-1] < self.rsi_overbought

        if breakout_long and rsi_long_ok and macd_long_ok and vol_ok and not_overbought and atr_ok:
            sl = price - self.atr_sl_mult * self.atr[-1]
            tp = price + self.atr_tp_mult * self.atr[-1]
            risk_per_unit = price - sl
            if risk_per_unit > 0:
                risk_amount = self.equity * self.risk_pct
                size = int(round(risk_amount / risk_per_unit))
                size = max(1, min(size, int(self.equity / price)))
                print(f"🚀🌙 LONG BREAKOUT! Price={price:.2f} SL={sl:.2f} TP={tp:.2f} Size={size}")
                self.buy(size=size, sl=sl, tp=tp)
                return

        # SHORT conditions
        breakout_short = price < self.ll[-2]
        rsi_short_ok = self.rsi[-1] < self.rsi_short_thresh and rsi_falling
        macd_short_ok = self.macd[-1] < self.macd_signal_line[-1] and self.macd_hist[-1] < 0 and hist_contracting
        not_oversold = self.rsi[-1] > self.rsi_oversold

        if breakout_short and rsi_short_ok and macd_short_ok and vol_ok and not_oversold and atr_ok:
            sl = price + self.atr_sl_mult * self.atr[-1]
            tp = price - self.atr_tp_mult * self.atr[-1]
            risk_per_unit = sl - price
            if risk_per_unit > 0:
                risk_amount = self.equity * self.risk_pct
                size = int(round(risk_amount / risk_per_unit))
                size = max(1, min(size, int(self.equity / price)))
                print(f"🚀🌙 SHORT BREAKDOWN! Price={price:.2f} SL={sl:.2f} TP={tp:.2f} Size={size}")
                self.sell(size=size, sl=sl, tp=tp)
                return


# ============================================================
# 🌙 RUN THE BACKTEST 🚀
# ============================================================

bt = Backtest(
    data,
    MomentumConvergence,
    cash=1_000_000,
    commission=0.001,
    exclusive_orders=True
)

print("🌙✨ Launching Moon Dev backtest sequence... 🚀")
stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙✨ Moon Dev backtest complete! To the moon! 🚀🌙")