import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# ── Moon Dev Data Loading 🌙 ────────────────────────────────────────────────
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'datetime': 'Date',
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume',
})
data['Date'] = pd.to_datetime(data['Date'])
data = data.set_index('Date')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
print(f"🌙 Moon Dev data loaded: {len(data)} bars ✨")
print(f"🚀 Date range: {data.index[0]} → {data.index[-1]}")


class VolatilityWeave(Strategy):
    # Strategy parameters
    base_period = 20
    min_period = 5
    max_period = 50
    bb_period = 20
    bb_std_mult = 2.0
    atr_period = 14
    atr_sma_period = 50
    risk_pct = 0.02          # 2% equity risk per trade
    atr_stop_mult = 1.5
    swing_lookback = 20
    vol_spike_mult = 3.0
    size = 0.99              # fraction of equity (0 < size < 1)

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low

        # ── ATR for volatility measurement ──
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)
        self.atr_sma = self.I(talib.SMA, self.atr, timeperiod=self.atr_sma_period)

        # ── Bollinger Bands ──
        self.bb_mid = self.I(talib.SMA, close, timeperiod=self.bb_period)
        self.bb_stddev = self.I(talib.STDDEV, close, timeperiod=self.bb_period, nbdev=1.0)
        self.bb_upper = self.I(lambda m, s: m + self.bb_std_mult * s, self.bb_mid, self.bb_stddev)
        self.bb_lower = self.I(lambda m, s: m - self.bb_std_mult * s, self.bb_mid, self.bb_stddev)

        # ── Adaptive EMA (computed bar-by-bar) ──
        self.adaptive_ema = self.I(self._adaptive_ema, close, self.atr, self.atr_sma)

        # ── Swing high/low for stop placement ──
        self.swing_high = self.I(talib.MAX, high, timeperiod=self.swing_lookback)
        self.swing_low = self.I(talib.MIN, low, timeperiod=self.swing_lookback)

        # State
        self.stop_price = None
        self.trade_dir = 0

        print("🌙✨ VolatilityWeave indicators initialized 🚀")

    @staticmethod
    def _adaptive_ema(close, atr, atr_sma):
        """Compute adaptive EMA bar-by-bar using volatility ratio."""
        n = len(close)
        out = np.full(n, np.nan)
        base = 20
        min_p = 5
        max_p = 50
        alpha_arr = np.full(n, np.nan)

        for i in range(n):
            if np.isnan(atr[i]) or np.isnan(atr_sma[i]) or atr_sma[i] == 0:
                continue
            vol_ratio = atr[i] / atr_sma[i]
            if vol_ratio <= 0:
                continue
            period = base * (1.0 / vol_ratio)
            period = max(min_p, min(max_p, period))
            alpha_arr[i] = 2.0 / (period + 1.0)

        # EMA pass
        prev = np.nan
        for i in range(n):
            if np.isnan(alpha_arr[i]):
                continue
            if np.isnan(prev):
                prev = close[i]
            else:
                prev = alpha_arr[i] * close[i] + (1 - alpha_arr[i]) * prev
            out[i] = prev
        return out

    def next(self):
        i = len(self.data) - 1
        if i < 2:
            return

        price = self.data.Close[-1]
        ema = self.adaptive_ema[-1]
        ema_prev = self.adaptive_ema[-2]
        lower = self.bb_lower[-1]
        lower_prev = self.bb_lower[-2]
        upper = self.bb_upper[-1]
        upper_prev = self.bb_upper[-2]
        mid = self.bb_mid[-1]
        mid_prev = self.bb_mid[-2]
        atr = self.atr[-1]
        atr_sma = self.atr_sma[-1]

        if (np.isnan(ema) or np.isnan(ema_prev) or np.isnan(lower) or
                np.isnan(lower_prev) or np.isnan(upper) or np.isnan(upper_prev) or
                np.isnan(mid) or np.isnan(mid_prev) or np.isnan(atr) or
                np.isnan(atr_sma) or atr_sma == 0):
            return

        # ── Volatility spike filter ──
        if atr > self.vol_spike_mult * atr_sma:
            return

        # ── Regime detection ──
        green = price > mid and mid > mid_prev
        red = price < mid and mid < mid_prev

        # ── Manage open position ──
        if self.position:
            if self.position.is_long:
                # Stop hit
                if self.stop_price and self.data.Low[-1] <= self.stop_price:
                    print(f"🛑🌙 Long STOP hit @ {self.stop_price:.2f}")
                    self.position.close()
                    self.stop_price = None
                    return
                # Exit: EMA crosses back below lower band, or regime flips red
                if (ema_prev >= lower_prev and ema < lower) or red:
                    print(f"🚀💚 Long EXIT @ {price:.2f} | ema={ema:.2f} lower={lower:.2f} red={red}")
                    self.position.close()
                    self.stop_price = None
                    return
            elif self.position.is_short:
                if self.stop_price and self.data.High[-1] >= self.stop_price:
                    print(f"🛑🌙 Short STOP hit @ {self.stop_price:.2f}")
                    self.position.close()
                    self.stop_price = None
                    return
                if (ema_prev <= upper_prev and ema > upper) or green:
                    print(f"🚀❤️ Short EXIT @ {price:.2f} | ema={ema:.2f} upper={upper:.2f} green={green}")
                    self.position.close()
                    self.stop_price = None
                    return

        # ── Entry logic ──
        if not self.position:
            # Long: green regime, EMA crosses above lower band
            if green and ema_prev <= lower_prev and ema > lower:
                stop = max(self.swing_low[-1], price - self.atr_stop_mult * atr)
                risk = price - stop
                if risk > 0:
                    # Ensure size is a valid fraction (0 < size < 1)
                    size = float(self.size)
                    if size >= 1.0:
                        size = 0.99
                    if size <= 0:
                        size = 0.99
                    self.buy(size=size, sl=stop)
                    self.stop_price = stop
                    print(f"🌙✨ LONG ENTRY @ {price:.2f} | ema={ema:.2f} lower={lower:.2f} "
                          f"stop={stop:.2f} size={size}")

            # Short: red regime, EMA crosses below upper band
            elif red and ema_prev >= upper_prev and ema < upper:
                stop = min(self.swing_high[-1], price + self.atr_stop_mult * atr)
                risk = stop - price
                if risk > 0:
                    size = float(self.size)
                    if size >= 1.0:
                        size = 0.99
                    if size <= 0:
                        size = 0.99
                    self.sell(size=size, sl=stop)
                    self.stop_price = stop
                    print(f"🌙✨ SHORT ENTRY @ {price:.2f} | ema={ema:.2f} upper={upper:.2f} "
                          f"stop={stop:.2f} size={size}")


# ── Run Backtest ────────────────────────────────────────────────────────────
bt = Backtest(data, VolatilityWeave, cash=1_000_000, commission=0.001, exclusive_orders=True)
stats = bt.run()
print(stats)
print(stats._strategy)