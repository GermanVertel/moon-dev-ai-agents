import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# 🌙 Moon Dev's Ichimoku Volume Reversion Strategy ✨
print("🌙 Moon Dev: Loading data and initializing backtest...")

data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Map to proper case
data.columns = [col.capitalize() for col in data.columns]

# Ensure datetime index
if 'Datetime' in data.columns:
    data['Datetime'] = pd.to_datetime(data['Datetime'])
    data = data.set_index('Datetime')

# Ensure required columns
required = ['Open', 'High', 'Low', 'Close', 'Volume']
data = data[required].dropna()

print(f"🌙 Moon Dev: Data loaded with {len(data)} bars ✨")
print(f"🌙 Moon Dev: Date range: {data.index[0]} to {data.index[-1]} 🚀")


class IchimokuVolumeReversion(Strategy):
    # Strategy parameters
    chikou_shift = 26
    sma_mean_period = 50
    sma_trend_period = 200
    vol_sma_period = 20
    vol_spike_mult = 1.5
    atr_period = 14
    atr_stop_mult = 2.0
    atr_trail_mult = 2.0
    max_hold_bars = 30
    risk_pct = 0.01

    def init(self):
        print("🌙 Moon Dev: Initializing indicators... ✨")
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        # Chikou Span = Close shifted back 26 (i.e., close plotted 26 bars back)
        # We compare current close against SMA shifted forward by 26 to align
        # Actually: Chikou Span at time t = Close[t+26] when plotted at t-26.
        # For real-time, Chikou Span value = Close (current), compared to SMA 26 bars ago.
        # Standard interpretation: compare current close to SMA 26 periods back.
        # We'll compute: chikou = current close; sma_mean_shifted = SMA(50) shifted +26 (so at t we look at SMA value 26 bars ago)
        self.sma_mean = self.I(talib.SMA, close, timeperiod=self.sma_mean_period)
        self.sma_mean_shifted = self.I(
            lambda x: pd.Series(x).shift(self.chikou_shift).values,
            self.sma_mean,
            name="SMA50_Shifted"
        )
        self.sma_trend = self.I(talib.SMA, close, timeperiod=self.sma_trend_period)
        self.vol_sma = self.I(talib.SMA, volume, timeperiod=self.vol_sma_period)
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        # ATR percentile for regime filter
        self.atr_pct_low = self.I(
            lambda x: pd.Series(x).rolling(200).quantile(0.20).values,
            self.atr,
            name="ATR_20pct"
        )
        self.atr_pct_high = self.I(
            lambda x: pd.Series(x).rolling(200).quantile(0.95).values,
            self.atr,
            name="ATR_95pct"
        )

        # Track entry info
        self.entry_bar = None
        self.entry_price = None
        self.stop_price = None
        self.target_hit = False

        print("🌙 Moon Dev: Indicators ready! 🚀")

    def next(self):
        # Need enough bars
        if len(self.data) < self.sma_trend_period + self.chikou_shift + 5:
            return

        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        volume = self.data.Volume[-1]
        atr = self.atr[-1]

        if np.isnan(atr) or atr <= 0:
            return

        sma_mean_shifted = self.sma_mean_shifted[-1]
        sma_trend = self.sma_trend[-1]
        vol_sma = self.vol_sma[-1]

        if np.isnan(sma_mean_shifted) or np.isnan(sma_trend) or np.isnan(vol_sma):
            return

        # Chikou Span = current close, compared to SMA 26 bars ago
        chikou = price

        # Volume spike
        vol_spike = volume > (self.vol_spike_mult * vol_sma)

        # Divergence: price move small relative to ATR (exhaustion)
        body = abs(self.data.Close[-1] - self.data.Open[-1])
        divergence = body < (0.5 * atr)

        # Volatility regime filter
        atr_low = self.atr_pct_low[-1]
        atr_high = self.atr_pct_high[-1]
        regime_ok = True
        if not np.isnan(atr_low) and atr < atr_low:
            regime_ok = False
        if not np.isnan(atr_high) and atr > atr_high:
            regime_ok = False

        # Manage open position
        if self.position:
            bars_held = len(self.data) - self.entry_bar
            is_long = self.position.is_long

            # Trailing stop
            if is_long:
                new_stop = price - (self.atr_trail_mult * atr)
                if new_stop > self.stop_price:
                    self.stop_price = new_stop
                # Check stop
                if low <= self.stop_price:
                    print(f"🌙 Moon Dev: LONG trailing stop hit at {self.stop_price:.2f} 💥")
                    self.position.close()
                    return
                # Mean target partial exit handled via simple close at mean
                if not self.target_hit and price >= sma_mean_shifted:
                    # Check trend - if trend opposite (bearish), hold longer
                    trend_bullish = price > sma_trend
                    if not trend_bullish:
                        # Hold longer, only trail
                        pass
                    else:
                        print(f"🌙 Moon Dev: LONG mean target reached at {price:.2f} 🎯")
                        self.position.close()
                        return
            else:
                new_stop = price + (self.atr_trail_mult * atr)
                if new_stop < self.stop_price or self.stop_price == 0:
                    self.stop_price = new_stop
                if high >= self.stop_price:
                    print(f"🌙 Moon Dev: SHORT trailing stop hit at {self.stop_price:.2f} 💥")
                    self.position.close()
                    return
                if not self.target_hit and price <= sma_mean_shifted:
                    trend_bearish = price < sma_trend
                    if not trend_bearish:
                        pass
                    else:
                        print(f"🌙 Moon Dev: SHORT mean target reached at {price:.2f} 🎯")
                        self.position.close()
                        return

            # Time stop
            if bars_held >= self.max_hold_bars:
                print(f"🌙 Moon Dev: Time stop after {bars_held} bars ⏰")
                self.position.close()
                return

            return

        # Entry logic
        if not regime_ok:
            return

        if not vol_spike or not divergence:
            return

        # Chikou cross detection (manual crossover - no backtesting.lib!)
        prev_chikou = self.data.Close[-2] if len(self.data) > 1 else chikou
        prev_sma_shifted = self.sma_mean_shifted[-2] if len(self.data) > 1 else sma_mean_shifted

        if np.isnan(prev_sma_shifted):
            return

        cross_above = prev_chikou <= prev_sma_shifted and chikou > sma_mean_shifted
        cross_below = prev_chikou >= prev_sma_shifted and chikou < sma_mean_shifted

        # Short entry: Chikou crosses above 50-SMA (overextended)
        if cross_above:
            # Risk-based position sizing
            stop_dist = self.atr_stop_mult * atr
            if stop_dist <= 0:
                return
            risk_amount = self.equity * self.risk_pct
            position_size = risk_amount / stop_dist
            position_size = int(round(position_size))
            if position_size <= 0:
                position_size = 1

            self.stop_price = price + stop_dist
            self.entry_bar = len(self.data)
            self.entry_price = price
            self.target_hit = False
            print(f"🌙 Moon Dev: SHORT entry at {price:.2f} | ATR={atr:.2f} | Size={position_size} 🚀")
            self.sell(size=position_size)

        # Long entry: Chikou crosses below 50-SMA (oversold)
        elif cross_below:
            stop_dist = self.atr_stop_mult * atr
            if stop_dist <= 0:
                return
            risk_amount = self.equity * self.risk_pct
            position_size = risk_amount / stop_dist
            position_size = int(round(position_size))
            if position_size <= 0:
                position_size = 1

            self.stop_price = price - stop_dist
            self.entry_bar = len(self.data)
            self.entry_price = price
            self.target_hit = False
            print(f"🌙 Moon Dev: LONG entry at {price:.2f} | ATR={atr:.2f} | Size={position_size} 🚀")
            self.buy(size=position_size)


print("🌙 Moon Dev: Starting backtest... ✨")
bt = Backtest(
    data,
    IchimokuVolumeReversion,
    cash=1_000_000,
    commission=0.002,
    exclusive_orders=True
)

stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙 Moon Dev: Backtest complete! 🚀✨")