import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# ── Moon Dev Data Loader ──────────────────────────────────────────
print("🌙 Moon Dev: Loading cosmic data from the vault...")
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper column mapping
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume',
    'datetime': 'Date'
})

# Parse datetime
if 'Date' in data.columns:
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.set_index('Date')

data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
print(f"🌙 Moon Dev: Data loaded! {len(data)} rows ready for launch 🚀")


# ── CorrelationPulse Strategy ─────────────────────────────────────
class CorrelationPulse(Strategy):
    """
    CorrelationPulse 🌙
    - MACD + RSI rolling correlation
    - Entry: corr crosses above +0.7 AND (MACD hist > 0 OR RSI > 50)
    - Exit: corr crosses below +0.7
    - Position sizing: 20% compounding after wins, reset on loss, cap at 3x base
    """

    # ── Tunable Parameters ────────────────────────────────────────
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    rsi_period = 14
    corr_window = 14
    corr_threshold = 0.7
    atr_period = 14
    atr_min_filter = 0.0  # minimum ATR (absolute) to trade

    base_size = 1_000_000
    size_multiplier = 1.20
    max_size_mult = 3.0

    stop_loss_pct = 0.03   # 3% safety stop
    dd_guard = 0.20        # halt if drawdown > 20%

    def init(self):
        print("🌙 Moon Dev: Initializing CorrelationPulse indicators...")

        # Use numpy arrays for talib — clean and fast
        close = np.asarray(self.data.Close, dtype=float)
        high = np.asarray(self.data.High, dtype=float)
        low = np.asarray(self.data.Low, dtype=float)

        # MACD — returns (macd, signal, hist) as numpy arrays
        self.macd, self.macd_signal, self.macd_hist = self.I(
            talib.MACD, close,
            fastperiod=self.macd_fast,
            slowperiod=self.macd_slow,
            signalperiod=self.macd_signal
        )

        # RSI
        self.rsi = self.I(talib.RSI, close, timeperiod=self.rsi_period)

        # ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        # Rolling correlation between MACD and RSI — compute as a full array
        # and wrap with a passthrough I() so backtesting.py indexes it correctly.
        macd_arr = np.asarray(self.macd, dtype=float)
        rsi_arr = np.asarray(self.rsi, dtype=float)

        # Use pandas rolling corr on numpy-backed series, then convert to numpy
        macd_s = pd.Series(macd_arr)
        rsi_s = pd.Series(rsi_arr)
        corr_arr = macd_s.rolling(self.corr_window).corr(rsi_s).fillna(0).to_numpy(dtype=float)

        self.corr = self.I(lambda: corr_arr, name='Corr')

        # State
        self.current_size = self.base_size
        self.last_entry_price = None
        self.peak_equity = None
        self.trading_halted = False

        print("🌙 Moon Dev: Indicators online! ✨")

    def next(self):
        # Need enough bars
        if len(self.data) < max(self.corr_window, self.macd_slow + self.macd_signal) + 5:
            return

        price = self.data.Close[-1]
        corr_now = self.corr[-1]
        corr_prev = self.corr[-2]
        macd_hist_now = self.macd_hist[-1]
        rsi_now = self.rsi[-1]
        atr_now = self.atr[-1]

        # ── Drawdown Guard ────────────────────────────────────────
        equity = self.equity
        if self.peak_equity is None or equity > self.peak_equity:
            self.peak_equity = equity
        if self.peak_equity and self.peak_equity > 0:
            dd = (self.peak_equity - equity) / self.peak_equity
            if dd > self.dd_guard and not self.trading_halted:
                print(f"🚨 Moon Dev: DRAWDOWN GUARD triggered ({dd:.2%}) — halting trading!")
                self.trading_halted = True
                if self.position:
                    self.position.close()
                return

        if self.trading_halted:
            return

        # ── Entry ─────────────────────────────────────────────────
        if not self.position:
            crossed_up = corr_prev <= self.corr_threshold and corr_now > self.corr_threshold
            momentum_ok = (macd_hist_now > 0) or (rsi_now > 50)
            vol_ok = atr_now > self.atr_min_filter

            if crossed_up and momentum_ok and vol_ok:
                size = int(round(self.current_size))
                if size < 1:
                    size = 1
                sl_price = price * (1 - self.stop_loss_pct)
                print(f"🚀 Moon Dev: ENTRY LONG @ {price:.2f} | corr={corr_now:.3f} "
                      f"| RSI={rsi_now:.1f} | MACDhist={macd_hist_now:.4f} | size={size}")
                self.buy(size=size, sl=sl_price)
                self.last_entry_price = price

        # ── Exit ──────────────────────────────────────────────────
        else:
            crossed_down = corr_prev >= self.corr_threshold and corr_now < self.corr_threshold
            if crossed_down:
                trade = self.trades[-1] if self.trades else None
                pnl = trade.pl if trade else 0
                print(f"🌙 Moon Dev: EXIT @ {price:.2f} | corr={corr_now:.3f} | PnL={pnl:.2f}")
                self.position.close()

                # Adjust size for next trade
                if pnl > 0:
                    new_size = self.current_size * self.size_multiplier
                    cap = self.base_size * self.max_size_mult
                    self.current_size = min(new_size, cap)
                    print(f"✨ Moon Dev: WIN! Next size boosted → {int(round(self.current_size))}")
                else:
                    self.current_size = self.base_size
                    print(f"💧 Moon Dev: LOSS. Size reset → {self.current_size}")


# ── Run Backtest ──────────────────────────────────────────────────
print("🌙 Moon Dev: Launching CorrelationPulse backtest... 🚀")
bt = Backtest(data, CorrelationPulse, cash=10_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)