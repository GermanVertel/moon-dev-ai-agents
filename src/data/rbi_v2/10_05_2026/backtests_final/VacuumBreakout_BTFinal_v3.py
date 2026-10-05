import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from datetime import datetime, time

# 🌙 Moon Dev VacuumBreakout Strategy ✨

DATA_PATH = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"

print("🌙 Moon Dev loading data from:", DATA_PATH)

data = pd.read_csv(DATA_PATH)
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

print("🌙 Moon Dev data shape:", data.shape)
print("✨ Moon Dev data head:\n", data.head())


class VacuumBreakout(Strategy):
    # Strategy parameters
    bb_period = 20
    bb_std = 2.0
    atr_period = 14
    squeeze_lookback = 100
    squeeze_pct_threshold = 0.20
    vwap_gap_threshold = 1.0
    void_volume_pct = 0.10
    void_bins = 30
    risk_pct = 0.01
    trail_atr_mult = 1.5
    time_exit_bars = 30
    max_consecutive_losses = 2

    def init(self):
        close = pd.Series(self.data.Close, index=self.data.index)
        high = pd.Series(self.data.High, index=self.data.index)
        low = pd.Series(self.data.Low, index=self.data.index)
        volume = pd.Series(self.data.Volume, index=self.data.index)

        # ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        # Bollinger Bands
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS, close, timeperiod=self.bb_period,
            nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0
        )

        # BandWidth = (upper - lower) / middle
        def _bw():
            up = np.array(self.bb_upper)
            lo = np.array(self.bb_lower)
            mid = np.array(self.bb_middle)
            with np.errstate(divide='ignore', invalid='ignore'):
                return np.where(mid != 0, (up - lo) / mid, np.nan)
        self.bandwidth = self.I(_bw)

        # BandWidth percentile rank over squeeze_lookback
        def _bw_rank():
            bw_arr = np.array(self.bandwidth)
            out = np.full(len(bw_arr), np.nan)
            for i in range(self.squeeze_lookback, len(bw_arr)):
                window = bw_arr[i - self.squeeze_lookback:i + 1]
                if np.any(np.isnan(window)):
                    continue
                out[i] = (window[-1] <= window).mean()
            return out
        self.bw_pct_rank = self.I(_bw_rank)

        # Session VWAP (anchored to daily reset)
        self.session_vwap = self.I(self._compute_vwap)

        # VWAP std bands
        def _vwap_std():
            return pd.Series(self.data.Close).rolling(50).std().values
        self.vwap_std = self.I(_vwap_std)

        # Overnight liquidity void boundaries (rolling proxy)
        self.void_high = self.I(talib.MAX, high, timeperiod=32)
        self.void_low = self.I(talib.MIN, low, timeperiod=32)

        self.consecutive_losses = 0
        self.last_entry_bar = -1
        self.entry_price = None
        self.stop_price = None
        self.target_price = None

    def _compute_vwap(self):
        pv = np.array(self.data.Close) * np.array(self.data.Volume)
        v = np.array(self.data.Volume)
        vwap_arr = np.zeros(len(pv))
        cum_pv = 0.0
        cum_v = 0.0
        prev_date = None
        for i in range(len(pv)):
            cur_date = self.data.index[i].date() if hasattr(self.data.index[i], 'date') else None
            if prev_date is not None and cur_date != prev_date:
                cum_pv = 0.0
                cum_v = 0.0
            cum_pv += pv[i]
            cum_v += v[i]
            vwap_arr[i] = cum_pv / cum_v if cum_v > 0 else pv[i]
            prev_date = cur_date
        return vwap_arr

    def next(self):
        i = len(self.data) - 1
        if i < self.squeeze_lookback + 5:
            return

        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        atr = self.atr[-1]
        vwap = self.session_vwap[-1]
        vwap_std = self.vwap_std[-1]
        bw_rank = self.bw_pct_rank[-1]
        void_high = self.void_high[-1]
        void_low = self.void_low[-1]

        if (np.isnan(atr) or np.isnan(vwap) or np.isnan(bw_rank)
                or np.isnan(vwap_std) or vwap_std == 0
                or np.isnan(void_high) or np.isnan(void_low)):
            return

        # VWAP gap normalized
        vwap_gap = (price - vwap) / vwap_std

        # Squeeze release detection: was in bottom 20%, now expanding
        prev_bw = self.bandwidth[-2] if len(self.bandwidth) > 1 else np.nan
        cur_bw = self.bandwidth[-1]
        squeeze_releasing = (
            not np.isnan(prev_bw) and not np.isnan(cur_bw) and
            cur_bw > prev_bw and
            bw_rank < self.squeeze_pct_threshold * 2
        )

        # Manage open position
        if self.position:
            bars_held = i - self.last_entry_bar
            if self.position.is_long:
                # Trailing stop
                new_stop = price - self.trail_atr_mult * atr
                if self.stop_price is None or new_stop > self.stop_price:
                    self.stop_price = new_stop
                if low <= self.stop_price:
                    self.position.close()
                    self.consecutive_losses += 1
                    print(f"🌙 Moon Dev LONG stopped out at {self.stop_price:.2f} | consecutive losses: {self.consecutive_losses}")
                    return
                # Take profit at target
                if self.target_price and high >= self.target_price:
                    self.position.close()
                    self.consecutive_losses = 0
                    print(f"✨ Moon Dev LONG target hit at {self.target_price:.2f} 🚀")
                    return
            else:
                new_stop = price + self.trail_atr_mult * atr
                if self.stop_price is None or new_stop < self.stop_price:
                    self.stop_price = new_stop
                if high >= self.stop_price:
                    self.position.close()
                    self.consecutive_losses += 1
                    print(f"🌙 Moon Dev SHORT stopped out at {self.stop_price:.2f} | consecutive losses: {self.consecutive_losses}")
                    return
                if self.target_price and low <= self.target_price:
                    self.position.close()
                    self.consecutive_losses = 0
                    print(f"✨ Moon Dev SHORT target hit at {self.target_price:.2f} 🚀")
                    return
            # Time-based exit
            if bars_held >= self.time_exit_bars:
                self.position.close()
                print(f"⏰ Moon Dev time exit after {bars_held} bars")
                return
            return

        # Skip if too many consecutive losses
        if self.consecutive_losses >= self.max_consecutive_losses:
            return

        if not squeeze_releasing:
            return

        # LONG entry
        if price > void_high and vwap_gap > self.vwap_gap_threshold:
            stop = max(void_high - 0.1 * atr, price - atr)
            risk = price - stop
            if risk <= 0:
                return
            size = 0.99  # 🌙 fraction of equity (valid position sizing)
            self.buy(size=size)
            self.entry_price = price
            self.stop_price = stop
            self.target_price = price + 2 * risk
            self.last_entry_bar = i
            print(f"🚀🌙 Moon Dev LONG VacuumBreakout | price={price:.2f} void_high={void_high:.2f} vwap_gap={vwap_gap:.2f} stop={stop:.2f} target={self.target_price:.2f}")

        # SHORT entry
        elif price < void_low and vwap_gap < -self.vwap_gap_threshold:
            stop = min(void_low + 0.1 * atr, price + atr)
            risk = stop - price
            if risk <= 0:
                return
            size = 0.99  # 🌙 fraction of equity (valid position sizing)
            self.sell(size=size)
            self.entry_price = price
            self.stop_price = stop
            self.target_price = price - 2 * risk
            self.last_entry_bar = i
            print(f"🚀🌙 Moon Dev SHORT VacuumBreakout | price={price:.2f} void_low={void_low:.2f} vwap_gap={vwap_gap:.2f} stop={stop:.2f} target={self.target_price:.2f}")


bt = Backtest(data, VacuumBreakout, cash=1_000_000, commission=0.0002)
stats = bt.run()
print(stats)
print(stats._strategy)