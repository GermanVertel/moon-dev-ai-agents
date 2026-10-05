import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

print("🌙 FlowPulse Strategy Initializing... ✨")

# Load data
data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper column mapping
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

print(f"🚀 Data loaded: {len(data)} rows")
print(f"📊 Columns: {list(data.columns)}")


class FlowPulse(Strategy):
    # Parameters
    spike_lookback = 20          # rolling window for order flow std
    spike_mult = 2.0             # std multiplier for spike detection
    entry_delay_min = 3          # min days after spike
    entry_delay_max = 5          # max days after spike
    ema_period = 20
    atr_period = 14
    atr_mult_stop = 1.5
    profit_target_pct = 0.03     # 3% profit target
    max_hold_bars = 7            # time stop
    risk_pct = 0.02              # 2% risk per trade

    def init(self):
        print("🌙 Initializing FlowPulse indicators... ✨")
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume
        open_ = self.data.Open

        # Proxy for order flow delta: use close-vs-open direction weighted by volume
        # Positive when close > open (buy-side pressure), negative otherwise
        close_arr = np.asarray(close)
        open_arr = np.asarray(open_)
        volume_arr = np.asarray(volume)

        direction = np.where(close_arr > open_arr, 1.0,
                             np.where(close_arr < open_arr, -1.0, 0.0))
        delta = direction * volume_arr

        self.delta = self.I(lambda: delta, name="OrderFlowDelta")

        # Cumulative volume delta
        cvd = np.cumsum(delta)
        self.cvd = self.I(lambda: cvd, name="CVD")

        # Rolling mean and std of delta for spike detection
        delta_series = pd.Series(delta)
        self.delta_mean = self.I(
            lambda s: s.rolling(self.spike_lookback).mean().values,
            delta_series, name="DeltaMean"
        )
        self.delta_std = self.I(
            lambda s: s.rolling(self.spike_lookback).std().values,
            delta_series, name="DeltaStd"
        )

        # 20 EMA
        self.ema = self.I(talib.EMA, close, timeperiod=self.ema_period, name="EMA20")

        # ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period, name="ATR")

        # VWAP (rolling anchored proxy: 20-bar VWAP)
        typical = (high + low + close) / 3
        typical_arr = np.asarray(typical)
        vwap = (pd.Series(typical_arr * volume_arr).rolling(20).sum() /
                pd.Series(volume_arr).rolling(20).sum()).values
        self.vwap = self.I(lambda: vwap, name="VWAP")

        # Track spike events
        self.spike_day = -1
        self.spike_low = 0.0
        self.spike_mid = 0.0
        self.spike_cvd = 0.0
        self.bars_since_spike = 0
        self.entry_bar = 0
        self.entry_price = 0.0
        self.stop_price = 0.0
        self.target_price = 0.0

        print("🌙 FlowPulse indicators ready! 🚀")

    def next(self):
        i = len(self.data) - 1

        if i < self.spike_lookback + 5:
            return

        # Detect spike: delta exceeds mean + spike_mult * std
        delta_val = self.delta[i]
        mean_val = self.delta_mean[i]
        std_val = self.delta_std[i]

        if np.isnan(mean_val) or np.isnan(std_val) or std_val == 0:
            return

        spike_threshold = mean_val + self.spike_mult * std_val

        # Check if today is a spike day
        if delta_val > spike_threshold and self.spike_day == -1 and not self.position:
            self.spike_day = i
            self.spike_low = self.data.Low[i]
            self.spike_mid = (self.data.High[i] + self.data.Low[i]) / 2
            self.spike_cvd = self.cvd[i]
            self.bars_since_spike = 0
            print(f"🌙✨ Spike detected at bar {i}! Delta={delta_val:.2f} > "
                  f"Thresh={spike_threshold:.2f} 🚀")
            return

        # Track bars since spike
        if self.spike_day != -1 and not self.position:
            self.bars_since_spike = i - self.spike_day

            # Entry window: 3-5 bars after spike
            if self.entry_delay_min <= self.bars_since_spike <= self.entry_delay_max:
                price = self.data.Close[i]
                price_above_mid = price > self.spike_mid
                price_above_vwap = price > self.vwap[i]
                cvd_positive = self.cvd[i] > self.spike_cvd
                above_ema = price > self.ema[i]

                if price_above_mid and price_above_vwap and cvd_positive and above_ema:
                    # Position sizing based on risk
                    atr_val = self.atr[i]
                    if np.isnan(atr_val) or atr_val <= 0:
                        return

                    stop_price = price - self.atr_mult_stop * atr_val
                    risk_per_unit = price - stop_price
                    if risk_per_unit <= 0:
                        return

                    equity = self.equity
                    risk_amount = equity * self.risk_pct
                    size = int(round(risk_amount / risk_per_unit))
                    if size < 1:
                        size = 1

                    self.buy(size=size)
                    self.entry_price = price
                    self.stop_price = stop_price
                    self.target_price = price * (1 + self.profit_target_pct)
                    self.entry_bar = i
                    print(f"🌙🚀 ENTRY at bar {i} | Price={price:.2f} | "
                          f"Size={size} | Stop={stop_price:.2f} | "
                          f"Target={self.target_price:.2f} | "
                          f"Delay={self.bars_since_spike} bars ✨")

                # Reset spike if window expired
                if self.bars_since_spike > self.entry_delay_max:
                    print(f"🌙 Spike window expired at bar {i}, resetting...")
                    self.spike_day = -1
                    self.bars_since_spike = 0

        # Manage open position
        if self.position:
            price = self.data.Close[i]
            bars_held = i - self.entry_bar

            # Stop loss
            if price <= self.stop_price:
                self.position.close()
                print(f"🌙🛑 STOP LOSS at bar {i} | Price={price:.2f} 💔")
                self.spike_day = -1
                return

            # Profit target
            if price >= self.target_price:
                self.position.close()
                print(f"🌙💰 PROFIT TARGET at bar {i} | Price={price:.2f} ✨")
                self.spike_day = -1
                return

            # Time stop / order flow fade
            if bars_held >= self.max_hold_bars:
                self.position.close()
                print(f"🌙⏰ TIME STOP at bar {i} | Price={price:.2f} | "
                      f"Bars held={bars_held} ✨")
                self.spike_day = -1
                return

            # Order flow turned negative
            if self.delta[i] < 0 and self.cvd[i] < self.cvd[self.entry_bar]:
                self.position.close()
                print(f"🌙📉 FLOW FADE EXIT at bar {i} | Price={price:.2f} ✨")
                self.spike_day = -1
                return


print("🌙✨ Launching FlowPulse Backtest... 🚀")
bt = Backtest(data, FlowPulse, cash=1_000_000, commission=0.002)
stats = bt.run()
print(stats)
print(stats._strategy)