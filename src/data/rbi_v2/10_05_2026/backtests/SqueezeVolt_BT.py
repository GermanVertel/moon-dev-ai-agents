import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# 🌙 Moon Dev SqueezeVolt Backtest 🌙
print("🌙✨ Starting SqueezeVolt backtest... Loading cosmic data 🚀")

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

if 'Date' in data.columns:
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.set_index('Date')

data = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
print(f"🌙 Data loaded: {len(data)} bars ✨")


class SqueezeVolt(Strategy):
    # Parameters
    kc_ema_period = 20
    kc_atr_period = 20
    kc_mult = 1.5
    bb_period = 20
    bb_std = 2.0
    bbw_sma_period = 20
    vol_sma_period = 20
    atr_period = 14
    squeeze_lookback = 100
    squeeze_pct = 20
    squeeze_recent_bars = 5
    risk_pct = 0.01
    adx_period = 14
    adx_threshold = 20

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        # 🌙 Keltner Channel
        self.kc_ema = self.I(talib.EMA, close, timeperiod=self.kc_ema_period)
        self.kc_atr = self.I(talib.ATR, high, low, close, timeperiod=self.kc_atr_period)
        self.kc_upper = self.I(lambda e, a: e + self.kc_mult * a, self.kc_ema, self.kc_atr)
        self.kc_lower = self.I(lambda e, a: e - self.kc_mult * a, self.kc_ema, self.kc_atr)
        self.kc_width = self.I(lambda u, l: u - l, self.kc_upper, self.kc_lower)

        # 🌙 Bollinger Bands
        self.bb_upper, self.bb_mid, self.bb_lower = self.I(
            talib.BBANDS, close, timeperiod=self.bb_period,
            nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0
        )

        # 🌙 Bollinger Bandwidth
        self.bbw = self.I(lambda u, l, m: (u - l) / m, self.bb_upper, self.bb_lower, self.bb_mid)
        self.bbw_sma = self.I(talib.SMA, self.bbw, timeperiod=self.bbw_sma_period)

        # 🌙 Volume SMA
        self.vol_sma = self.I(talib.SMA, volume, timeperiod=self.vol_sma_period)

        # 🌙 ATR for stops
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        # 🌙 ADX trend filter
        self.adx = self.I(talib.ADX, high, low, close, timeperiod=self.adx_period)

        # 🌙 KC width percentile tracking
        self.kc_width_series = pd.Series(self.kc_width)
        self.kc_width_pct = self.I(
            lambda: self.kc_width_series.rolling(self.squeeze_lookback).quantile(self.squeeze_pct / 100.0)
        )

        # State tracking
        self.squeeze_active = False
        self.bars_since_squeeze = 999
        self.entry_price = None
        self.stop_price = None
        self.target1 = None
        self.partial_taken = False
        self.bars_in_trade = 0

    def next(self):
        i = len(self.data) - 1
        if i < max(self.squeeze_lookback, self.bb_period, self.kc_ema_period) + 5:
            return

        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        vol = self.data.Volume[-1]

        # 🌙 Squeeze detection
        kc_width_now = self.kc_width[-1]
        kc_width_pct = self.kc_width_pct[-1]
        bb_inside_kc = (self.bb_upper[-1] < self.kc_upper[-1]) and (self.bb_lower[-1] > self.kc_lower[-1])
        bbw_low = self.bbw[-1] < self.bbw_sma[-1]

        squeeze_condition = (
            not np.isnan(kc_width_pct) and
            kc_width_now <= kc_width_pct and
            bb_inside_kc and
            bbw_low
        )

        if squeeze_condition:
            self.squeeze_active = True
            self.bars_since_squeeze = 0
        else:
            self.bars_since_squeeze += 1
            if self.bars_since_squeeze > self.squeeze_recent_bars:
                self.squeeze_active = False

        # 🌙 BBW expansion check
        bbw_expanding = (not np.isnan(self.bbw[-2])) and (self.bbw[-1] > self.bbw[-2]) and (self.bbw[-1] > self.bbw_sma[-1])

        # 🌙 Volume surge
        vol_surge = (not np.isnan(self.vol_sma[-1])) and (vol > 1.5 * self.vol_sma[-1])

        # 🌙 Trend filter
        adx_ok = (not np.isnan(self.adx[-1])) and (self.adx[-1] > self.adx_threshold)

        # 🌙 Recent squeeze
        squeeze_recent = self.bars_since_squeeze <= self.squeeze_recent_bars

        # ============ EXIT LOGIC ============
        if self.position:
            self.bars_in_trade += 1
            entry = self.entry_price
            is_long = self.position.is_long

            # Time stop
            if self.bars_in_trade >= 10 and squeeze_condition:
                print(f"🌙⏰ Time stop exit at {price:.2f}")
                self.position.close()
                self._reset_trade()
                return

            # Volatility reversal exit
            if (not np.isnan(self.bbw[-2])) and self.bbw[-1] < self.bbw[-2] * 0.7:
                print(f"🌙📉 Volatility reversal exit at {price:.2f}")
                self.position.close()
                self._reset_trade()
                return

            if is_long:
                # Stop loss
                if low <= self.stop_price:
                    print(f"🌙🛑 Long stop hit at {self.stop_price:.2f}")
                    self.position.close()
                    self._reset_trade()
                    return
                # Partial at 1.5R
                if not self.partial_taken and high >= self.target1:
                    print(f"🌙💰 Long partial TP at {self.target1:.2f}")
                    self.position.close(0.5)
                    self.partial_taken = True
                # Trail using KC midline
                if price < self.kc_ema[-1] and self.partial_taken:
                    print(f"🌙🎯 Long trail exit at {price:.2f}")
                    self.position.close()
                    self._reset_trade()
                    return
            else:
                if high >= self.stop_price:
                    print(f"🌙🛑 Short stop hit at {self.stop_price:.2f}")
                    self.position.close()
                    self._reset_trade()
                    return
                if not self.partial_taken and low <= self.target1:
                    print(f"🌙💰 Short partial TP at {self.target1:.2f}")
                    self.position.close(0.5)
                    self.partial_taken = True
                if price > self.kc_ema[-1] and self.partial_taken:
                    print(f"🌙🎯 Short trail exit at {price:.2f}")
                    self.position.close()
                    self._reset_trade()
                    return
            return

        # ============ ENTRY LOGIC ============
        if not squeeze_recent:
            return
        if not bbw_expanding:
            return
        if not vol_surge:
            return
        if not adx_ok:
            return

        atr_val = self.atr[-1]
        if np.isnan(atr_val) or atr_val <= 0:
            return

        # Long entry
        if price > self.kc_upper[-1]:
            stop = max(price - 1.5 * atr_val, self.kc_ema[-1])
            risk = price - stop
            if risk <= 0:
                return
            equity = self.equity
            size = int(round((self.risk_pct * equity) / risk))
            if size < 1:
                return
            self.entry_price = price
            self.stop_price = stop
            self.target1 = price + 1.5 * risk
            self.partial_taken = False
            self.bars_in_trade = 0
            print(f"🌙🚀 LONG SqueezeVolt entry at {price:.2f} | stop {stop:.2f} | size {size}")
            self.buy(size=size)
            return

        # Short entry
        if price < self.kc_lower[-1]:
            stop = min(price + 1.5 * atr_val, self.kc_ema[-1])
            risk = stop - price
            if risk <= 0:
                return
            equity = self.equity
            size = int(round((self.risk_pct * equity) / risk))
            if size < 1:
                return
            self.entry_price = price
            self.stop_price = stop
            self.target1 = price - 1.5 * risk
            self.partial_taken = False
            self.bars_in_trade = 0
            print(f"🌙🔻 SHORT SqueezeVolt entry at {price:.2f} | stop {stop:.2f} | size {size}")
            self.sell(size=size)
            return

    def _reset_trade(self):
        self.entry_price = None
        self.stop_price = None
        self.target1 = None
        self.partial_taken = False
        self.bars_in_trade = 0


bt = Backtest(
    data, SqueezeVolt,
    cash=1_000_000,
    commission=0.001,
    exclusive_orders=True
)

stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙✨ SqueezeVolt backtest complete! 🚀")