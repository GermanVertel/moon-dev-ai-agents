import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# 🌙 Moon Dev's AnchoredVolatility Backtest 🚀

print("🌙 Moon Dev: Loading data...")

data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper column mapping
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

print(f"🌙 Moon Dev: Data loaded with {len(data)} bars ✨")


class AnchoredVolatility(Strategy):
    # Strategy parameters
    bb_k = 2.0
    bb_lookback = 20
    adx_period = 14
    adx_threshold = 25
    vol_lookback = 100
    vol_percentile = 90
    atr_period = 14
    atr_stop_mult = 1.5
    risk_pct = 0.01
    time_stop_bars = 10

    def init(self):
        print("🌙 Moon Dev: Initializing AnchoredVolatility strategy...")

        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        # ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        # ADX and DI
        self.adx = self.I(talib.ADX, high, low, close, timeperiod=self.adx_period)
        self.plus_di = self.I(talib.PLUS_DI, high, low, close, timeperiod=self.adx_period)
        self.minus_di = self.I(talib.MINUS_DI, high, low, close, timeperiod=self.adx_period)

        # Anchored VWAP (session anchor - reset each day)
        # Compute typical price * volume cumulative within each session
        tp = (high + low + close) / 3.0
        pv = tp * volume

        # Session identification (daily anchor)
        idx = self.data.index
        if hasattr(idx, 'date'):
            dates = pd.Series(idx.date, index=idx)
        else:
            dates = pd.Series(idx, index=idx)

        # Build cumulative sums within each day
        pv_series = pd.Series(np.asarray(pv), index=idx)
        vol_series = pd.Series(np.asarray(volume), index=idx)

        # Group by date for session anchoring
        date_arr = np.array([d for d in dates])
        # Find session boundaries
        session_id = np.zeros(len(date_arr), dtype=int)
        sid = 0
        for i in range(1, len(date_arr)):
            if date_arr[i] != date_arr[i - 1]:
                sid += 1
            session_id[i] = sid

        cum_pv = np.zeros(len(pv_series))
        cum_v = np.zeros(len(vol_series))
        pv_arr = np.asarray(pv_series)
        v_arr = np.asarray(vol_series)

        for i in range(len(pv_arr)):
            if i == 0 or session_id[i] != session_id[i - 1]:
                cum_pv[i] = pv_arr[i]
                cum_v[i] = v_arr[i]
            else:
                cum_pv[i] = cum_pv[i - 1] + pv_arr[i]
                cum_v[i] = cum_v[i - 1] + v_arr[i]

        avwap = np.where(cum_v > 0, cum_pv / cum_v, np.nan)
        self.avwap = self.I(lambda x: x, avwap, name="AVWAP")

        # Deviation from AVWAP (rolling std over lookback)
        deviation = close - avwap
        dev_series = pd.Series(deviation, index=idx)
        std_dev = dev_series.rolling(self.bb_lookback).std().values
        self.std_dev = self.I(lambda x: x, std_dev, name="AVWAP_STD")

        # Bands
        upper = avwap + self.bb_k * std_dev
        lower = avwap - self.bb_k * std_dev
        self.upper_band = self.I(lambda x: x, upper, name="UpperBand")
        self.lower_band = self.I(lambda x: x, lower, name="LowerBand")

        # Volume percentile (rolling)
        vol_series_full = pd.Series(np.asarray(volume), index=idx)
        vol_pct = vol_series_full.rolling(self.vol_lookback).quantile(self.vol_percentile / 100.0).values
        self.vol_thresh = self.I(lambda x: x, vol_pct, name="VolPct90")

        # Bandwidth
        bandwidth = (upper - lower) / np.where(avwap != 0, avwap, np.nan)
        self.bandwidth = self.I(lambda x: x, bandwidth, name="Bandwidth")

        # Track entry state
        self.entry_price = None
        self.entry_bar = None
        self.stop_price = None
        self.target1 = None
        self.target2 = None
        self.scaled_out = False
        self.highest_since_entry = None
        self.lowest_since_entry = None
        self.trade_direction = 0

        print("🌙 Moon Dev: Indicators ready! ✨")

    def next(self):
        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        vol = self.data.Volume[-1]

        # Check indicator readiness
        if len(self.data) < max(self.vol_lookback, self.bb_lookback, self.adx_period) + 5:
            return

        avwap = self.avwap[-1]
        upper = self.upper_band[-1]
        lower = self.lower_band[-1]
        adx = self.adx[-1]
        adx_prev = self.adx[-2]
        pdi = self.plus_di[-1]
        mdi = self.minus_di[-1]
        atr = self.atr[-1]
        vol_thresh = self.vol_thresh[-1]
        bw = self.bandwidth[-1]
        bw_prev = self.bandwidth[-2]

        if any(np.isnan(x) for x in [avwap, upper, lower, adx, adx_prev, pdi, mdi, atr, vol_thresh, bw, bw_prev]):
            return

        # ================= MANAGE OPEN POSITION =================
        if self.position:
            if self.trade_direction == 1:  # Long
                self.highest_since_entry = max(self.highest_since_entry, high)

                # Regime exit
                if adx < 20 or mdi > pdi:
                    print(f"🌙 Moon Dev: Regime exit LONG @ {price:.2f} (ADX={adx:.1f}) 🚪")
                    self.position.close()
                    self._reset_trade_state()
                    return

                # Trailing stop using Chandelier Exit (3x ATR from highest high)
                chandelier = self.highest_since_entry - 3.0 * atr
                new_stop = max(self.stop_price, chandelier, avwap)
                if new_stop > self.stop_price:
                    self.stop_price = new_stop

                # Scale out at 1.5R
                if not self.scaled_out and price >= self.target1:
                    print(f"🌙 Moon Dev: Target 1 hit LONG @ {price:.2f}, scaling out 50% 💰")
                    self.position.close(0.5)
                    self.scaled_out = True

                # Stop loss
                if low <= self.stop_price:
                    print(f"🌙 Moon Dev: Stop hit LONG @ {self.stop_price:.2f} 🛑")
                    self.position.close()
                    self._reset_trade_state()
                    return

                # Target 2
                if price >= self.target2:
                    print(f"🌙 Moon Dev: Target 2 hit LONG @ {price:.2f} 🎯")
                    self.position.close()
                    self._reset_trade_state()
                    return

                # Time stop
                if not self.scaled_out and (len(self.data) - self.entry_bar) >= self.time_stop_bars:
                    if price < self.entry_price + (self.entry_price - self.stop_price):
                        print(f"🌙 Moon Dev: Time stop LONG @ {price:.2f} ⏰")
                        self.position.close()
                        self._reset_trade_state()
                        return

            elif self.trade_direction == -1:  # Short
                self.lowest_since_entry = min(self.lowest_since_entry, low)

                if adx < 20 or pdi > mdi:
                    print(f"🌙 Moon Dev: Regime exit SHORT @ {price:.2f} (ADX={adx:.1f}) 🚪")
                    self.position.close()
                    self._reset_trade_state()
                    return

                chandelier = self.lowest_since_entry + 3.0 * atr
                new_stop = min(self.stop_price, chandelier, avwap)
                if new_stop < self.stop_price:
                    self.stop_price = new_stop

                if not self.scaled_out and price <= self.target1:
                    print(f"🌙 Moon Dev: Target 1 hit SHORT @ {price:.2f}, scaling out 50% 💰")
                    self.position.close(0.5)
                    self.scaled_out = True

                if high >= self.stop_price:
                    print(f"🌙 Moon Dev: Stop hit SHORT @ {self.stop_price:.2f} 🛑")
                    self.position.close()
                    self._reset_trade_state()
                    return

                if price <= self.target2:
                    print(f"🌙 Moon Dev: Target 2 hit SHORT @ {price:.2f} 🎯")
                    self.position.close()
                    self._reset_trade_state()
                    return

                if not self.scaled_out and (len(self.data) - self.entry_bar) >= self.time_stop_bars:
                    if price > self.entry_price - (self.stop_price - self.entry_price):
                        print(f"🌙 Moon Dev: Time stop SHORT @ {price:.2f} ⏰")
                        self.position.close()
                        self._reset_trade_state()
                        return

            return  # Don't open new position while in one

        # ================= ENTRY LOGIC =================
        adx_rising = adx > adx_prev
        vol_surge = vol > vol_thresh
        bw_expanding = bw > bw_prev

        long_cond = (
            price > upper and
            adx >= self.adx_threshold and adx_rising and
            pdi > mdi and
            vol_surge and
            bw_expanding
        )

        short_cond = (
            price < lower and
            adx >= self.adx_threshold and adx_rising and
            mdi > pdi and
            vol_surge and
            bw_expanding
        )

        if long_cond:
            stop = min(avwap, price - self.atr_stop_mult * atr)
            risk = price - stop
            if risk <= 0:
                return

            # Position sizing: risk_pct of equity
            equity = self.equity
            risk_amount = equity * self.risk_pct
            position_size = int(round(risk_amount / risk))
            if position_size < 1:
                position_size = 1

            self.buy(size=position_size)
            self.entry_price = price
            self.entry_bar = len(self.data)
            self.stop_price = stop
            self.target1 = price + 1.5 * risk
            self.target2 = price + 2.5 * risk
            self.scaled_out = False
            self.highest_since_entry = high
            self.trade_direction = 1

            print(f"🚀 Moon Dev LONG ENTRY @ {price:.2f} | Stop={stop:.2f} | T1={self.target1:.2f} | T2={self.target2:.2f} | Size={position_size} | ADX={adx:.1f} 🌙")

        elif short_cond:
            stop = max(avwap, price + self.atr_stop_mult * atr)
            risk = stop - price
            if risk <= 0:
                return

            equity = self.equity
            risk_amount = equity * self.risk_pct
            position_size = int(round(risk_amount / risk))
            if position_size < 1:
                position_size = 1

            self.sell(size=position_size)
            self.entry_price = price
            self.entry_bar = len(self.data)
            self.stop_price = stop
            self.target1 = price - 1.5 * risk
            self.target2 = price - 2.5 * risk
            self.scaled_out = False
            self.lowest_since_entry = low
            self.trade_direction = -1

            print(f"🔻 Moon Dev SHORT ENTRY @ {price:.2f} | Stop={stop:.2f} | T1={self.target1:.2f} | T2={self.target2:.2f} | Size={position_size} | ADX={adx:.1f} 🌙")

    def _reset_trade_state(self):
        self.entry_price = None
        self.entry_bar = None
        self.stop_price = None
        self.target1 = None
        self.target2 = None
        self.scaled_out = False
        self.highest_since_entry = None
        self.lowest_since_entry = None
        self.trade_direction = 0


print("🌙 Moon Dev: Starting backtest... 🚀")

bt = Backtest(
    data,
    AnchoredVolatility,
    cash=1_000_000,
    commission=0.001,
    exclusive_orders=True
)

stats = bt.run()
print(stats)
print(stats._strategy)

print("🌙 Moon Dev: Backtest complete! ✨")