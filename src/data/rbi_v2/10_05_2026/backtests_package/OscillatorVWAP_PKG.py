import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# 🌙 Moon Dev Data Loading & Cleaning
data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper column mapping
data.columns = [col.capitalize() for col in data.columns]
if 'Datetime' in data.columns:
    data = data.drop(columns=['Datetime'])

print("🌙✨ Moon Dev Data Loaded! Shape:", data.shape)
print("🚀 Columns:", list(data.columns))


class OscillatorVWAP(Strategy):
    # Strategy parameters
    so_period = 14
    so_smooth = 3
    rsi_period = 14
    divergence_lookback = 20
    pivot_window = 5
    atr_period = 14
    atr_stop_mult = 2.0
    risk_pct = 0.02
    rsi_exit_level = 30
    rsi_exit_band = 28
    time_stop_bars = 50

    def init(self):
        print("🌙 Moon Dev initializing indicators... ✨")
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        # Standard Stochastic (0-100) then shift to zero-based (-50 to +50)
        # Zero-based SO: %K - 50, so oversold < -20 means standard %K < 30
        def zero_based_stoch(high, low, close):
            k, d = talib.STOCH(high, low, close,
                               fastk_period=self.so_period,
                               slowk_period=self.so_smooth,
                               slowk_matype=0,
                               slowd_period=self.so_smooth,
                               slowd_matype=0)
            return k - 50.0

        self.so = self.I(zero_based_stoch, high, low, close, name="SO_zero")

        # RSI
        self.rsi = self.I(talib.RSI, close, timeperiod=self.rsi_period, name="RSI")

        # ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period, name="ATR")

        # VWAP (cumulative session proxy: cumulative typical*vol / cumulative vol)
        def vwap_calc(high, low, close, volume):
            tp = (high + low + close) / 3.0
            cum_v = np.cumsum(volume)
            cum_tpv = np.cumsum(tp * volume)
            with np.errstate(divide='ignore', invalid='ignore'):
                vwap = np.where(cum_v > 0, cum_tpv / cum_v, np.nan)
            return vwap

        self.vwap = self.I(vwap_calc, high, low, close, volume, name="VWAP")

        # Rolling min low for divergence (price lower low)
        self.low_min = self.I(talib.MIN, low, timeperiod=self.divergence_lookback, name="LowMin")
        # Rolling min SO for divergence (SO higher low)
        self.so_min = self.I(talib.MIN, self.so, timeperiod=self.divergence_lookback, name="SOMin")

        # Track stop loss and entry bar
        self.stop_price = None
        self.entry_bar = None

        print("🌙 Moon Dev indicators ready! 🚀")

    def next(self):
        price = self.data.Close[-1]
        so = self.so[-1]
        rsi = self.rsi[-1]
        vwap = self.vwap[-1]
        atr = self.atr[-1]

        # Skip if indicators not ready
        if np.isnan(so) or np.isnan(rsi) or np.isnan(vwap) or np.isnan(atr):
            return

        # ==================== EXIT LOGIC ====================
        if self.position:
            # Primary exit: RSI recovers toward 30 from below
            if rsi >= self.rsi_exit_band and rsi <= self.rsi_exit_level + 5:
                print(f"🌙✨ RSI exit! RSI={rsi:.2f} -> Closing long at {price:.2f}")
                self.position.close()
                self.stop_price = None
                self.entry_bar = None
                return

            # Protective stop loss
            if self.stop_price is not None and price <= self.stop_price:
                print(f"🛑 Stop loss hit at {price:.2f} (stop={self.stop_price:.2f})")
                self.position.close()
                self.stop_price = None
                self.entry_bar = None
                return

            # Time stop
            if self.entry_bar is not None and (len(self.data) - self.entry_bar) >= self.time_stop_bars:
                print(f"⏰ Time stop! Closing at {price:.2f}")
                self.position.close()
                self.stop_price = None
                self.entry_bar = None
                return

        # ==================== ENTRY LOGIC ====================
        if not self.position:
            # Condition 1: SO < -20 (zero-based oversold)
            so_oversold = so < -20

            # Condition 2: Bullish divergence
            # Price makes lower low: current low is near the rolling min (new low)
            # SO makes higher low: current SO is above the rolling min of SO
            if len(self.data) >= self.divergence_lookback + 2:
                cur_low = self.data.Low[-1]
                prev_low = self.low_min[-2]
                cur_so = self.so[-1]
                prev_so_min = self.so_min[-2]

                price_lower_low = cur_low <= prev_low
                so_higher_low = cur_so > prev_so_min
                divergence = price_lower_low and so_higher_low
            else:
                divergence = False

            # Condition 3: VWAP confirmation - price above VWAP
            vwap_confirm = price > vwap

            if so_oversold and divergence and vwap_confirm:
                # Risk-based position sizing
                equity = self.equity
                risk_amount = equity * self.risk_pct
                stop_distance = atr * self.atr_stop_mult
                if stop_distance <= 0:
                    return

                position_size = risk_amount / stop_distance
                position_size = int(round(position_size))
                if position_size < 1:
                    position_size = 1

                # Cap by equity (approx)
                max_size = int(equity / price) if price > 0 else 1
                position_size = min(position_size, max_size)
                if position_size < 1:
                    return

                self.stop_price = price - stop_distance
                self.entry_bar = len(self.data)

                print(f"🚀🌙 LONG ENTRY! price={price:.2f} SO={so:.2f} RSI={rsi:.2f} "
                      f"VWAP={vwap:.2f} size={position_size} stop={self.stop_price:.2f}")
                self.buy(size=position_size)


# 🌙 Run the backtest
print("🌙 Moon Dev launching OscillatorVWAP backtest... 🚀")
bt = Backtest(data, OscillatorVWAP, cash=1_000_000, commission=0.002)
stats = bt.run()
print(stats)
print(stats._strategy)