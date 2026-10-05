import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Load data 🌙
data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean column names ✨
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper column mapping 🚀
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')

# Ensure numeric dtypes for talib 🚀
for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    data[col] = pd.to_numeric(data[col], errors='coerce').astype('float64')

data = data.dropna()

print(f"🌙 Moon Dev data loaded! Shape: {data.shape}")
print(f"✨ Columns: {list(data.columns)}")
print(f"🚀 Date range: {data.index[0]} to {data.index[-1]}")


class LiquidationCluster(Strategy):
    # Strategy parameters
    kc_ema_period = 20
    kc_atr_mult = 2.0
    atr_period = 14
    vwap_period = 20
    atr_avg_period = 20
    atr_vol_threshold = 1.2
    cluster_proximity_pct = 0.005  # 0.5%
    cluster_max_distance_pct = 0.05  # 5%
    stop_buffer_pct = 0.003  # 0.3%
    swing_lookback = 20
    time_exit_bars = 10
    risk_pct = 0.015  # 1.5% risk

    def init(self):
        print("🌙 Initializing LiquidationCluster indicators...")
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        # Keltner Channel
        self.ema = self.I(talib.EMA, close, timeperiod=self.kc_ema_period)
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)
        self.upper_kc = self.I(lambda e, a: e + self.kc_atr_mult * a, self.ema, self.atr)
        self.lower_kc = self.I(lambda e, a: e - self.kc_atr_mult * a, self.ema, self.atr)

        # ATR average for volatility trigger
        self.atr_avg = self.I(talib.SMA, self.atr, timeperiod=self.atr_avg_period)

        # VWAP (rolling 20-period) - cast arrays to float64 for talib
        typical = np.asarray((high + low + close) / 3, dtype=np.float64)
        volume_arr = np.asarray(volume, dtype=np.float64)

        vol_sum = self.I(talib.SUM, volume_arr, timeperiod=self.vwap_period)
        tp_vol = self.I(lambda t, v: np.asarray(t, dtype=np.float64) * np.asarray(v, dtype=np.float64),
                        typical, volume_arr)
        tp_vol_sum = self.I(talib.SUM, tp_vol, timeperiod=self.vwap_period)
        self.vwap = self.I(lambda a, b: np.asarray(a, dtype=np.float64) / np.where(np.asarray(b) == 0, 1, np.asarray(b)),
                           tp_vol_sum, vol_sum)

        # Swing high for stop placement
        self.swing_high = self.I(talib.MAX, high, timeperiod=self.swing_lookback)

        # Proxy for liquidation clusters: recent swing highs (where long liquidations cluster)
        self.recent_high = self.I(talib.MAX, high, timeperiod=50)

        # RSI for optional divergence confirmation
        self.rsi = self.I(talib.RSI, close, timeperiod=14)

        # Volume average
        self.vol_avg = self.I(talib.SMA, volume_arr, timeperiod=20)

        print("✨ All indicators initialized!")

    def next(self):
        price = self.data.Close[-1]

        # Skip if indicators not ready
        if len(self.data) < max(self.kc_ema_period, self.atr_avg_period, self.swing_lookback) + 5:
            return
        if np.isnan(self.upper_kc[-1]) or np.isnan(self.atr_avg[-1]) or np.isnan(self.vwap[-1]):
            return

        # Manage existing position
        if self.position:
            self._manage_position()
            return

        # Check entry conditions
        self._check_entry(price)

    def _manage_position(self):
        price = self.data.Close[-1]
        vwap = self.vwap[-1]
        mid_kc = self.ema[-1]

        # Count bars since entry
        bars_in_trade = len(self.data) - self.trades[-1].entry_bar if self.trades else 0

        # Partial profit at VWAP
        if self.position.is_short and not getattr(self, '_partial_taken', False):
            if price <= vwap:
                print(f"🌙✨ Price hit VWAP at {price:.2f}! Taking 50% profit!")
                self.position.close(portion=0.5)
                self._partial_taken = True

        # Trail remaining with middle KC
        if self.position.is_short and getattr(self, '_partial_taken', False):
            if price >= mid_kc:
                print(f"🚀 Trail stop hit at middle KC {mid_kc:.2f}! Exiting!")
                self.position.close()
                self._partial_taken = False
                return

        # Time-based exit
        if bars_in_trade >= self.time_exit_bars:
            print(f"⏰ Time exit after {bars_in_trade} bars!")
            self.position.close()
            self._partial_taken = False
            return

        # Invalidation: price reclaims upper KC with strong volume
        if self.position.is_short:
            if price > self.upper_kc[-1] and self.data.Volume[-1] > self.vol_avg[-1] * 1.5:
                print(f"🚨 INVALIDATION: Price reclaimed upper KC with strong volume! Exiting!")
                self.position.close()
                self._partial_taken = False

    def _check_entry(self, price):
        upper_kc = self.upper_kc[-1]
        lower_kc = self.lower_kc[-1]
        atr = self.atr[-1]
        atr_avg = self.atr_avg[-1]
        vwap = self.vwap[-1]

        # Volatility trigger
        if atr < self.atr_vol_threshold * atr_avg:
            return

        # Check recent KC breakout (previous candle closed above upper KC)
        prev_close = self.data.Close[-2]
        prev_upper = self.upper_kc[-2]

        if prev_close > prev_upper and price < upper_kc:
            # Failed breakout / rejection - short setup
            print(f"🌙 LiquidationCluster SHORT signal! Price={price:.2f}, UpperKC={upper_kc:.2f}, VWAP={vwap:.2f}")

            # Stop loss: above recent swing high + buffer
            swing = self.swing_high[-1]
            stop = max(swing, price + 1.5 * atr) * (1 + self.stop_buffer_pct)

            # Risk/Reward check
            risk = stop - price
            reward = price - vwap
            if risk <= 0 or reward <= 0:
                return
            if reward / risk < 1.5:
                print(f"⚠️ R/R too low: {reward/risk:.2f}. Skipping.")
                return

            # Position sizing: 1.5% risk, fraction of equity
            equity = self.equity
            risk_amount = equity * self.risk_pct
            position_size = risk_amount / risk  # fraction of equity

            # Cap fraction at 0.95 to avoid insufficient cash
            if position_size > 0.95:
                position_size = 0.95
            if position_size < 0.01:
                position_size = 0.01

            print(f"🚀 Entering SHORT size={position_size:.4f}, Stop={stop:.2f}, Target VWAP={vwap:.2f}, R/R={reward/risk:.2f}")

            self.sell(size=position_size, sl=stop, tp=vwap)
            self._partial_taken = False


print("🌙✨🚀 Starting Moon Dev LiquidationCluster Backtest...")
bt = Backtest(data, LiquidationCluster, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)