import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Load data
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper column mapping
data = data.rename(columns={
    'datetime': 'Date',
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

data['Date'] = pd.to_datetime(data['Date'])
data = data.set_index('Date')

print("🌙 Moon Dev data loaded! Shape:", data.shape)
print("✨ Columns:", list(data.columns))
print("🚀 Head:\n", data.head())


class VolatilityDivergence(Strategy):
    # Strategy parameters
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    atr_period = 14
    rsi_period = 14
    vix_sma_period = 5
    swing_lookback = 5
    vix_spike_pct = 0.08  # 8% spike over N bars
    vix_spike_bars = 3
    vol_lookback = 20
    vol_decline_bars = 3
    risk_pct = 0.01
    atr_stop_mult = 1.0
    atr_target_mult = 2.0
    time_stop_bars = 20
    position_size = 1_000_000

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        # MACD
        self.macd, self.macd_sig, self.macd_hist = self.I(
            talib.MACD, close,
            fastperiod=self.macd_fast,
            slowperiod=self.macd_slow,
            signalperiod=self.macd_signal
        )

        # ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        # RSI
        self.rsi = self.I(talib.RSI, close, timeperiod=self.rsi_period)

        # Swing highs (fractal-like using rolling max)
        self.swing_high = self.I(talib.MAX, high, timeperiod=self.swing_lookback)
        self.swing_low = self.I(talib.MIN, low, timeperiod=self.swing_lookback)

        # Volume SMA
        self.vol_sma = self.I(talib.SMA, volume, timeperiod=self.vol_lookback)

        # VIX proxy: use ATR-based volatility proxy since no VIX column exists
        # We'll use ATR/Close as a volatility proxy, and its SMA
        vol_proxy = self.atr / close
        self.vol_proxy = self.I(lambda: vol_proxy)
        self.vol_proxy_sma = self.I(talib.SMA, vol_proxy, timeperiod=self.vix_sma_period)

        # Track swing highs and MACD values at those highs for divergence
        self.swing_highs = []  # list of (index, price, macd_val)
        self.entry_bar = None
        self.entry_price = None
        self.entry_atr = None
        self.entry_vol_proxy = None
        self.stop_price = None
        self.target_price = None

        print("🌙✨ VolatilityDivergence indicators initialized! 🚀")

    def _detect_bearish_divergence(self):
        """Detect bearish MACD divergence using recent swing highs."""
        i = len(self.data) - 1
        if i < self.swing_lookback * 2 + 5:
            return False

        # Check if current bar is a swing high (local max)
        window = self.data.High[i - self.swing_lookback + 1:i + 1]
        if len(window) < self.swing_lookback:
            return False
        if self.data.High[i] < np.max(window):
            return False

        current_price = self.data.High[i]
        current_macd = self.macd[i]

        if np.isnan(current_macd):
            return False

        # Look for a previous swing high within a reasonable window
        for j in range(i - self.swing_lookback * 2, i - self.swing_lookback + 1):
            if j < self.swing_lookback:
                continue
            prev_window = self.data.High[j - self.swing_lookback + 1:j + 1]
            if len(prev_window) < self.swing_lookback:
                continue
            if self.data.High[j] < np.max(prev_window):
                continue

            prev_price = self.data.High[j]
            prev_macd = self.macd[j]

            if np.isnan(prev_macd):
                continue

            # Bearish divergence: price higher high, MACD lower high
            if current_price > prev_price and current_macd < prev_macd:
                print(f"🌙 Bearish divergence detected! Price: {prev_price:.2f} -> {current_price:.2f}, MACD: {prev_macd:.4f} -> {current_macd:.4f}")
                return True

        return False

    def _vix_spike_confirmed(self):
        """Check if volatility proxy spiked recently."""
        i = len(self.data) - 1
        if i < self.vix_spike_bars + 1:
            return False

        current_vol = self.vol_proxy[i]
        past_vol = self.vol_proxy[i - self.vix_spike_bars]

        if np.isnan(current_vol) or np.isnan(past_vol) or past_vol == 0:
            return False

        pct_change = (current_vol - past_vol) / past_vol
        spike = pct_change >= self.vix_spike_pct

        # Also check crossing above SMA
        cross_above = False
        if not np.isnan(self.vol_proxy_sma[i]) and not np.isnan(self.vol_proxy_sma[i - 1]):
            cross_above = (self.vol_proxy[i] > self.vol_proxy_sma[i] and
                           self.vol_proxy[i - 1] <= self.vol_proxy_sma[i - 1])

        if spike or cross_above:
            print(f"🚀 VIX/Vol spike confirmed! pct_change={pct_change:.4f}, cross_above={cross_above}")
            return True
        return False

    def _volume_declining(self):
        """Check if volume declining on down bars for M consecutive bars."""
        i = len(self.data) - 1
        if i < self.vol_decline_bars:
            return False

        declines = 0
        for k in range(i - self.vol_decline_bars + 1, i + 1):
            if k < 1:
                continue
            down_bar = self.data.Close[k] < self.data.Close[k - 1]
            vol_lower = self.data.Volume[k] < self.data.Volume[k - 1]
            if down_bar and vol_lower:
                declines += 1

        return declines >= self.vol_decline_bars

    def next(self):
        i = len(self.data) - 1
        price = self.data.Close[i]

        if np.isnan(self.atr[i]) or self.atr[i] <= 0:
            return

        # ===== ENTRY LOGIC =====
        if not self.position:
            # Avoid extreme volatility
            vol_proxy_val = self.vol_proxy[i]
            if not np.isnan(vol_proxy_val) and vol_proxy_val > 0.05:
                return

            divergence = self._detect_bearish_divergence()
            vix_confirm = self._vix_spike_confirmed()

            # Optional RSI filter
            rsi_ok = True
            if not np.isnan(self.rsi[i]):
                rsi_ok = self.rsi[i] < 75

            if divergence and vix_confirm and rsi_ok:
                # Position sizing based on risk
                stop_price = self.data.High[i] + self.atr_stop_mult * self.atr[i]
                risk_per_unit = stop_price - price

                if risk_per_unit <= 0:
                    return

                equity = self.equity
                risk_amount = equity * self.risk_pct
                size = int(round(risk_amount / risk_per_unit))
                size = max(1, min(size, int(self.position_size)))

                print(f"🌙🚀 SHORT ENTRY! Price={price:.2f}, Stop={stop_price:.2f}, Size={size}, ATR={self.atr[i]:.2f}")

                self.sell(size=size)
                self.entry_bar = i
                self.entry_price = price
                self.entry_atr = self.atr[i]
                self.entry_vol_proxy = vol_proxy_val
                self.stop_price = stop_price
                self.target_price = price - self.atr_target_mult * self.atr[i]

        # ===== EXIT LOGIC =====
        else:
            # Stop loss
            if self.data.High[i] >= self.stop_price:
                print(f"🌙❌ STOP LOSS hit! High={self.data.High[i]:.2f} >= Stop={self.stop_price:.2f}")
                self.position.close()
                self._reset_trade()
                return

            # Profit target
            if self.data.Low[i] <= self.target_price:
                print(f"🌙✅ PROFIT TARGET hit! Low={self.data.Low[i]:.2f} <= Target={self.target_price:.2f}")
                self.position.close()
                self._reset_trade()
                return

            # IV-rise exit: vol proxy rises sharply but price doesn't make new lows
            vol_proxy_val = self.vol_proxy[i]
            if (not np.isnan(vol_proxy_val) and
                not np.isnan(self.entry_vol_proxy) and
                self.entry_vol_proxy > 0):
                vol_rise = (vol_proxy_val - self.entry_vol_proxy) / self.entry_vol_proxy
                made_new_low = self.data.Low[i] < self.data.Low[self.entry_bar] if self.entry_bar else False
                if vol_rise > 0.15 and not made_new_low:
                    print(f"🌙⚠️ IV-RISE EXIT! Vol rose {vol_rise:.2%} without new lows. Closing.")
                    self.position.close()
                    self._reset_trade()
                    return

            # Volume-decline exit
            if self._volume_declining():
                print(f"🌙📉 VOLUME-DECLINE EXIT! Selling pressure fading. Closing.")
                self.position.close()
                self._reset_trade()
                return

            # Time stop
            if self.entry_bar is not None and (i - self.entry_bar) >= self.time_stop_bars:
                print(f"🌙⏰ TIME STOP! {i - self.entry_bar} bars elapsed. Closing.")
                self.position.close()
                self._reset_trade()
                return

    def _reset_trade(self):
        self.entry_bar = None
        self.entry_price = None
        self.entry_atr = None
        self.entry_vol_proxy = None
        self.stop_price = None
        self.target_price = None


# Run backtest
bt = Backtest(data, VolatilityDivergence, cash=1_000_000, commission=0.002)

stats = bt.run()
print(stats)
print(stats._strategy)