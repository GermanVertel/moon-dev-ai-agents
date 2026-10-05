import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# 🌙 Moon Dev's VolatilityPulse Backtest 🌙

DATA_PATH = '/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv'

print("🌙✨ Moon Dev VolatilityPulse Backtest Initializing... 🚀")

# Load data
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

# Ensure datetime index
if 'datetime' in data.columns:
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')

print(f"🌙 Data loaded: {len(data)} bars ✨")
print(f"🚀 Columns: {list(data.columns)}")


class VolatilityPulse(Strategy):
    # Strategy parameters
    bb_period = 20
    bb_std = 2.0
    bbw_avg_period = 20
    bbw_expansion_mult = 1.5
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    vwma_period = 200
    atr_period = 14
    vol_sma_period = 20
    risk_pct = 0.01
    rr_ratio = 2.0
    flip_window = 2
    time_stop_bars = 10

    def init(self):
        print("🌙 Initializing indicators... ✨")
        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        volume = pd.Series(self.data.Volume)

        # Bollinger Bands
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS, close, timeperiod=self.bb_period,
            nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0
        )

        # Bollinger Bandwidth
        def calc_bbw(upper, lower, middle):
            return (upper - lower) / middle

        self.bbw = self.I(calc_bbw, self.bb_upper, self.bb_lower, self.bb_middle)
        self.bbw_avg = self.I(talib.SMA, self.bbw, timeperiod=self.bbw_avg_period)

        # MACD
        self.macd_line, self.macd_signal_line, self.macd_hist = self.I(
            talib.MACD, close, fastperiod=self.macd_fast,
            slowperiod=self.macd_slow, signalperiod=self.macd_signal
        )

        # VWMA (200)
        def calc_vwma(price, vol, period):
            price = pd.Series(price)
            vol = pd.Series(vol)
            return (price * vol).rolling(period).sum() / vol.rolling(period).sum()

        self.vwma = self.I(calc_vwma, close, volume, self.vwma_period)

        # ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        # Volume SMA
        self.vol_sma = self.I(talib.SMA, volume, timeperiod=self.vol_sma_period)

        # Track entry info
        self.entry_price = None
        self.entry_bar = None
        self.stop_price = None
        self.tp_price = None

        print("🌙✨ Indicators ready! 🚀")

    def next(self):
        # Need enough data
        if len(self.data) < self.vwma_period + 5:
            return

        price = self.data.Close[-1]
        bbw_now = self.bbw[-1]
        bbw_avg = self.bbw_avg[-1]
        hist_now = self.macd_hist[-1]
        vwma_now = self.vwma[-1]
        atr_now = self.atr[-1]
        vol_now = self.data.Volume[-1]
        vol_avg = self.vol_sma[-1]

        if np.isnan(bbw_now) or np.isnan(bbw_avg) or np.isnan(hist_now) or np.isnan(vwma_now):
            return

        # BBW expansion condition
        bbw_expansion = bbw_now > (self.bbw_expansion_mult * bbw_avg)

        # MACD histogram flip within window
        long_flip = False
        short_flip = False
        for i in range(1, self.flip_window + 2):
            if len(self.macd_hist) > i + 1:
                h_prev = self.macd_hist[-i - 1]
                h_curr = self.macd_hist[-i]
                if not np.isnan(h_prev) and not np.isnan(h_curr):
                    if h_prev < 0 and h_curr > 0:
                        long_flip = True
                    if h_prev > 0 and h_curr < 0:
                        short_flip = True

        # Volume confirmation
        vol_confirm = vol_now > vol_avg

        # Manage existing position
        if self.position:
            bars_held = len(self.data) - self.entry_bar
            is_long = self.position.is_long

            # Stop loss hit
            if self.stop_price is not None:
                if is_long and self.data.Low[-1] <= self.stop_price:
                    print(f"🌙💥 LONG STOP HIT at {self.stop_price:.2f} 🚀")
                    self.position.close()
                    self._reset()
                    return
                if not is_long and self.data.High[-1] >= self.stop_price:
                    print(f"🌙💥 SHORT STOP HIT at {self.stop_price:.2f} 🚀")
                    self.position.close()
                    self._reset()
                    return

            # Take profit
            if self.tp_price is not None:
                if is_long and self.data.High[-1] >= self.tp_price:
                    print(f"🌙🎯 LONG TP HIT at {self.tp_price:.2f} ✨")
                    self.position.close()
                    self._reset()
                    return
                if not is_long and self.data.Low[-1] <= self.tp_price:
                    print(f"🌙🎯 SHORT TP HIT at {self.tp_price:.2f} ✨")
                    self.position.close()
                    self._reset()
                    return

            # Exit: MACD histogram flip back
            if is_long and hist_now < 0:
                print(f"🌙🔄 LONG EXIT: MACD hist flipped negative 🚀")
                self.position.close()
                self._reset()
                return
            if not is_long and hist_now > 0:
                print(f"🌙🔄 SHORT EXIT: MACD hist flipped positive 🚀")
                self.position.close()
                self._reset()
                return

            # Exit: price crossed VWMA
            if is_long and price > vwma_now:
                print(f"🌙🔄 LONG EXIT: price above VWMA ✨")
                self.position.close()
                self._reset()
                return
            if not is_long and price < vwma_now:
                print(f"🌙🔄 SHORT EXIT: price below VWMA ✨")
                self.position.close()
                self._reset()
                return

            # Exit: BBW contraction
            if bbw_now < bbw_avg:
                print(f"🌙🔄 EXIT: BBW contracted below avg 🚀")
                self.position.close()
                self._reset()
                return

            # Time stop
            if bars_held >= self.time_stop_bars:
                # Check if 1R reached
                if is_long:
                    r_reached = self.data.High[-1] >= self.entry_price + (self.entry_price - self.stop_price)
                else:
                    r_reached = self.data.Low[-1] <= self.entry_price - (self.stop_price - self.entry_price)
                if not r_reached:
                    print(f"🌙⏰ TIME STOP: {bars_held} bars, no 1R 🚀")
                    self.position.close()
                    self._reset()
                    return

        # Entry logic
        if not self.position:
            # Long entry
            if (bbw_expansion and long_flip and price < vwma_now and vol_confirm):
                stop_dist = 1.5 * atr_now
                if stop_dist <= 0:
                    return
                # Position size: risk 1% of equity
                equity = self.equity
                risk_amount = equity * self.risk_pct
                size = int(round(risk_amount / stop_dist))
                if size < 1:
                    size = 1
                self.stop_price = price - stop_dist
                self.tp_price = price + (stop_dist * self.rr_ratio)
                self.entry_price = price
                self.entry_bar = len(self.data)
                print(f"🌙🚀 LONG ENTRY @ {price:.2f} | SL: {self.stop_price:.2f} | TP: {self.tp_price:.2f} | Size: {size} ✨")
                self.buy(size=size)

            # Short entry
            elif (bbw_expansion and short_flip and price > vwma_now and vol_confirm):
                stop_dist = 1.5 * atr_now
                if stop_dist <= 0:
                    return
                equity = self.equity
                risk_amount = equity * self.risk_pct
                size = int(round(risk_amount / stop_dist))
                if size < 1:
                    size = 1
                self.stop_price = price + stop_dist
                self.tp_price = price - (stop_dist * self.rr_ratio)
                self.entry_price = price
                self.entry_bar = len(self.data)
                print(f"🌙🚀 SHORT ENTRY @ {price:.2f} | SL: {self.stop_price:.2f} | TP: {self.tp_price:.2f} | Size: {size} ✨")
                self.sell(size=size)

    def _reset(self):
        self.entry_price = None
        self.entry_bar = None
        self.stop_price = None
        self.tp_price = None


print("🌙✨ Running VolatilityPulse Backtest... 🚀")
bt = Backtest(data, VolatilityPulse, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙✨ Backtest complete! 🚀")