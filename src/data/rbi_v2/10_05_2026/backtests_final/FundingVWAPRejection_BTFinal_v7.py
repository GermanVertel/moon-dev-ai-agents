import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# ============================================================
# 🌙 Moon Dev's FundingVWAPRejection Backtest 🌙
# ============================================================

print("🌙 Moon Dev: Loading data from the lunar archives...")

data_path = '/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv'
data = pd.read_csv(data_path)

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Ensure proper column mapping
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

# Set datetime index
if 'datetime' in data.columns:
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')

data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

print(f"🌙 Moon Dev: Data loaded! {len(data)} candles ready for launch 🚀")


class FundingVWAPRejection(Strategy):
    # Strategy parameters
    bb_period = 20
    bb_std = 2.0
    atr_period = 14
    rsi_period = 14
    vwap_cluster_pct = 0.01
    stop_atr_mult = 1.5
    risk_pct = 0.02
    rsi_threshold = 55

    def init(self):
        print("🌙 Moon Dev: Initializing lunar indicators...")

        # Bollinger Bands
        self.bb_upper, self.bb_mid, self.bb_lower = self.I(
            talib.BBANDS, self.data.Close,
            timeperiod=self.bb_period, nbdevup=self.bb_std, nbdevdn=self.bb_std
        )

        # ATR(14)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close,
                          timeperiod=self.atr_period)

        # RSI(14)
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=self.rsi_period)

        # Weekly VWAP proxies via rolling windows
        close_arr = np.asarray(self.data.Close, dtype=float)
        high_arr = np.asarray(self.data.High, dtype=float)
        low_arr = np.asarray(self.data.Low, dtype=float)
        vol_arr = np.asarray(self.data.Volume, dtype=float)

        typical = (high_arr + low_arr + close_arr) / 3.0
        pv = typical * vol_arr

        pv_series = pd.Series(pv)
        vol_series = pd.Series(vol_arr)

        def rolling_vwap_672():
            num = pv_series.rolling(672).sum()
            den = vol_series.rolling(672).sum()
            return (num / den).values

        def rolling_vwap_1344():
            num = pv_series.rolling(1344).sum()
            den = vol_series.rolling(1344).sum()
            return (num / den).values

        def rolling_vwap_2016():
            num = pv_series.rolling(2016).sum()
            den = vol_series.rolling(2016).sum()
            return (num / den).values

        self.vwap_1 = self.I(rolling_vwap_672)
        self.vwap_2 = self.I(rolling_vwap_1344)
        self.vwap_3 = self.I(rolling_vwap_2016)

        print("🌙 Moon Dev: Indicators online! ✨")

    def _count_vwap_cluster(self, price):
        """Count how many weekly VWAPs are within cluster_pct of price"""
        count = 0
        for v in [self.vwap_1[-1], self.vwap_2[-1], self.vwap_3[-1]]:
            if v is not None and not np.isnan(v) and v > 0:
                if abs(price - v) / price <= self.vwap_cluster_pct:
                    count += 1
        return count

    def next(self):
        # Need enough data for the longest rolling VWAP (2016) plus buffer
        if len(self.data) < 2100:
            return

        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        open_ = self.data.Open[-1]

        atr = self.atr[-1]
        if np.isnan(atr) or atr <= 0:
            return

        # ---------------- Entry Logic ----------------
        if not self.position:
            rsi_val = self.rsi[-1]
            cluster_count = self._count_vwap_cluster(price)

            candle_range = high - low
            upper_wick = high - max(open_, price)
            bearish = price < open_

            rejection = False
            if candle_range > 0:
                upper_wick_ratio = upper_wick / candle_range
                if bearish and upper_wick_ratio >= 0.4:
                    rejection = True

            if (cluster_count >= 1 and rejection and
                    not np.isnan(rsi_val) and rsi_val > self.rsi_threshold):

                stop_price = price + self.stop_atr_mult * atr
                risk_per_unit = stop_price - price

                if risk_per_unit > 0:
                    # Use fraction of equity for position sizing (valid for backtesting.py)
                    position_size = self.risk_pct

                    if position_size > 0:
                        self.sell(size=position_size, sl=stop_price)
                        print(f"🌙 Moon Dev: SHORT ENTRY! Price={price:.2f} "
                              f"RSI={rsi_val:.1f} Cluster={cluster_count} "
                              f"Size={position_size} SL={stop_price:.2f} 🚀")

        # ---------------- Exit Logic ----------------
        else:
            bb_low = self.bb_lower[-1]
            if not np.isnan(bb_low) and low <= bb_low:
                self.position.close()
                print(f"🌙 Moon Dev: BB LOWER SWEEP EXIT! Price={price:.2f} "
                      f"BB_Lower={bb_low:.2f} 💰")
                return

            rsi_val = self.rsi[-1]
            if not np.isnan(rsi_val) and rsi_val <= 50:
                self.position.close()
                print(f"🌙 Moon Dev: FUNDING NORMALIZED EXIT! RSI={rsi_val:.1f} 💰")
                return


print("🌙 Moon Dev: Launching backtest sequence... 🚀")

bt = Backtest(
    data,
    FundingVWAPRejection,
    cash=1_000_000,
    commission=0.001
)

stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙 Moon Dev: Backtest complete! To the moon! 🚀✨")