import pandas as pd
import numpy as np
import talib
from backtesting import Backtest, Strategy

print("🌙 Moon Dev's LiquidationReversal Backtest loading... ✨")

# Load data
data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Map to proper case
data.columns = [col.capitalize() for col in data.columns]

# Ensure datetime index
if 'Datetime' in data.columns:
    data['Datetime'] = pd.to_datetime(data['Datetime'])
    data = data.set_index('Datetime')

# Keep required columns
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
data = data.dropna()

print(f"🚀 Data loaded: {len(data)} candles 🌙")


class LiquidationReversal(Strategy):
    # Bollinger Bands
    bb_period = 20
    bb_std_mult = 2.0
    # RSI
    rsi_period = 14
    # ATR
    atr_period = 14
    atr_avg_period = 20
    atr_spike_mult = 3.0
    # Liquidation cluster detection (via wick + volume spike)
    wick_threshold = 0.003  # 0.3% wick
    vol_spike_mult = 1.8
    vol_avg_period = 20
    # Risk
    risk_pct = 0.01
    stop_buffer = 0.0025  # 0.25%
    time_stop_bars = 4  # 15m candles ~ 1 hour
    # Divergence lookback
    div_lookback = 10

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        # Bollinger Bands
        self.bb_mid = self.I(talib.SMA, close, timeperiod=self.bb_period)
        self.bb_stddev = self.I(talib.STDDEV, close, timeperiod=self.bb_period, nbdev=1.0)
        self.bb_upper = self.I(lambda m, s: m + self.bb_std_mult * s,
                              self.bb_mid, self.bb_stddev)
        self.bb_lower = self.I(lambda m, s: m - self.bb_std_mult * s,
                              self.bb_mid, self.bb_stddev)

        # RSI
        self.rsi = self.I(talib.RSI, close, timeperiod=self.rsi_period)

        # ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)
        self.atr_avg = self.I(talib.SMA, self.atr, timeperiod=self.atr_avg_period)

        # Volume average
        self.vol_avg = self.I(talib.SMA, volume, timeperiod=self.vol_avg_period)

        print("🌙 Indicators initialized ✨")

    def next(self):
        i = len(self.data) - 1
        if i < self.bb_period + 5:
            return

        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        open_ = self.data.Open[-1]
        vol = self.data.Volume[-1]

        upper = self.bb_upper[-1]
        lower = self.bb_lower[-1]
        mid = self.bb_mid[-1]
        rsi = self.rsi[-1]
        atr = self.atr[-1]
        atr_avg = self.atr_avg[-1]
        vol_avg = self.vol_avg[-1]

        if np.isnan(upper) or np.isnan(lower) or np.isnan(rsi) or np.isnan(atr_avg):
            return

        # Volatility filter: skip extreme volatility spikes
        if atr_avg > 0 and atr > self.atr_spike_mult * atr_avg:
            return

        # Manage existing position
        if self.position:
            bars_held = len(self.data) - self._entry_bar
            if self.position.is_long:
                # Target: upper band
                if price >= upper:
                    print(f"🌙✨ LONG target hit at upper band {price:.2f} 🚀")
                    self.position.close()
                    return
                # Time stop
                if bars_held >= self.time_stop_bars:
                    print(f"⏰ LONG time stop exit at {price:.2f} 🌙")
                    self.position.close()
                    return
            elif self.position.is_short:
                if price <= lower:
                    print(f"🌙✨ SHORT target hit at lower band {price:.2f} 🚀")
                    self.position.close()
                    return
                if bars_held >= self.time_stop_bars:
                    print(f"⏰ SHORT time stop exit at {price:.2f} 🌙")
                    self.position.close()
                    return
            return

        # Detect liquidation cluster via wick + volume spike
        body = abs(price - open_)
        upper_wick = high - max(price, open_)
        lower_wick = min(price, open_) - low

        vol_spike = vol_avg > 0 and vol > self.vol_spike_mult * vol_avg

        # Long reversal setup
        long_setup = False
        if price <= lower and vol_spike:
            # Big lower wick (forced sells)
            if lower_wick > self.wick_threshold * price:
                # Bullish RSI divergence: price lower low, RSI higher low
                lookback = min(self.div_lookback, i - 1)
                if lookback >= 3:
                    price_lows = self.data.Low[-lookback:]
                    rsi_vals = self.rsi[-lookback:]
                    # current is lowest price, but RSI higher than previous low
                    min_price_idx = int(np.argmin(price_lows))
                    if min_price_idx >= len(price_lows) - 2:
                        # find prior low
                        prior_lows = price_lows[:min_price_idx] if min_price_idx > 0 else price_lows[:-1]
                        if len(prior_lows) > 0:
                            prior_min = np.min(prior_lows)
                            prior_idx = int(np.argmin(prior_lows))
                            if price_lows[min_price_idx] < prior_min and rsi_vals[min_price_idx] > rsi_vals[prior_idx]:
                                long_setup = True

        # Short reversal setup
        short_setup = False
        if price >= upper and vol_spike:
            if upper_wick > self.wick_threshold * price:
                lookback = min(self.div_lookback, i - 1)
                if lookback >= 3:
                    price_highs = self.data.High[-lookback:]
                    rsi_vals = self.rsi[-lookback:]
                    max_price_idx = int(np.argmax(price_highs))
                    if max_price_idx >= len(price_highs) - 2:
                        prior_highs = price_highs[:max_price_idx] if max_price_idx > 0 else price_highs[:-1]
                        if len(prior_highs) > 0:
                            prior_max = np.max(prior_highs)
                            prior_idx = int(np.argmax(prior_highs))
                            if price_highs[max_price_idx] > prior_max and rsi_vals[max_price_idx] < rsi_vals[prior_idx]:
                                short_setup = True

        # Execute long
        if long_setup:
            stop = low * (1 - self.stop_buffer)
            risk = price - stop
            if risk > 0:
                size = int(round(1_000_000 / price))
                if size > 0:
                    print(f"🚀🌙 LONG ENTRY: price={price:.2f} lower_bb={lower:.2f} "
                          f"rsi={rsi:.1f} stop={stop:.2f} size={size}")
                    self.buy(size=size, sl=stop)
                    self._entry_bar = i

        # Execute short
        elif short_setup:
            stop = high * (1 + self.stop_buffer)
            risk = stop - price
            if risk > 0:
                size = int(round(1_000_000 / price))
                if size > 0:
                    print(f"🚀🌙 SHORT ENTRY: price={price:.2f} upper_bb={upper:.2f} "
                          f"rsi={rsi:.1f} stop={stop:.2f} size={size}")
                    self.sell(size=size, sl=stop)
                    self._entry_bar = i


print("🌙 Running backtest... ✨🚀")
bt = Backtest(data, LiquidationReversal, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙✨ Backtest complete! 🚀")