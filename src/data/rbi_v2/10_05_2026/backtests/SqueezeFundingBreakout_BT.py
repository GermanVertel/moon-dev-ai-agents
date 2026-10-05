import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# 🌙 Moon Dev's SqueezeFundingBreakout Backtest ✨🚀

DATA_PATH = '/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv'


def load_and_clean_data(path):
    print("🌙 Loading data from the Moon base...", path)
    data = pd.read_csv(path)
    data.columns = data.columns.str.strip().str.lower()
    data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
    # Proper column mapping
    data = data.rename(columns={
        'datetime': 'Datetime',
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    })
    data['Datetime'] = pd.to_datetime(data['Datetime'])
    data = data.set_index('Datetime')
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    print(f"✨ Data loaded with {len(data)} rows of pure Moon energy! 🌙")
    return data


class SqueezeFundingBreakout(Strategy):
    # Strategy parameters
    bb_period = 20
    bb_std = 2.0
    bbw_lookback = 100
    squeeze_pct = 0.20
    squeeze_min_bars = 3
    vol_sma_period = 20
    vol_mult = 1.5
    atr_period = 14
    atr_stop_mult = 1.8
    atr_tp1_mult = 2.0
    atr_tp2_mult = 3.5
    ema_period = 20
    time_stop_bars = 12

    def init(self):
        print("🌙 Initializing Moon Dev SqueezeFundingBreakout indicators... ✨")
        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        volume = pd.Series(self.data.Volume)

        # Bollinger Bands
        self.bb_mid = self.I(talib.SMA, close, timeperiod=self.bb_period, name='BB_MID')
        self.bb_stddev = self.I(talib.STDDEV, close, timeperiod=self.bb_period, nbdev=1, name='BB_STD')
        # Upper/lower computed in next
        self.bb_upper = self.I(lambda c, m, s: m + self.bb_std * s,
                               close, self.bb_mid, self.bb_stddev, name='BB_UPPER')
        self.bb_lower = self.I(lambda c, m, s: m - self.bb_std * s,
                               close, self.bb_mid, self.bb_stddev, name='BB_LOWER')

        # Bandwidth
        self.bbw = self.I(lambda u, l, m: (u - l) / np.where(m == 0, np.nan, m),
                          self.bb_upper, self.bb_lower, self.bb_mid, name='BBW')

        # Volume SMA
        self.vol_sma = self.I(talib.SMA, volume, timeperiod=self.vol_sma_period, name='VOL_SMA')

        # ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period, name='ATR')

        # EMA for trailing
        self.ema = self.I(talib.EMA, close, timeperiod=self.ema_period, name='EMA')

        # Squeeze detection: rolling percentile of bandwidth
        bbw_arr = np.array(self.bbw)
        squeeze_thresh = np.full(len(bbw_arr), np.nan)
        for i in range(self.bbw_lookback, len(bbw_arr)):
            window = bbw_arr[i - self.bbw_lookback:i]
            window = window[~np.isnan(window)]
            if len(window) > 10:
                squeeze_thresh[i] = np.percentile(window, self.squeeze_pct * 100)
        self.squeeze_thresh = self.I(lambda: squeeze_thresh, name='SQUEEZE_THRESH')

        # In-squeeze boolean series
        is_sq = np.zeros(len(bbw_arr))
        for i in range(len(bbw_arr)):
            if not np.isnan(bbw_arr[i]) and not np.isnan(squeeze_thresh[i]):
                if bbw_arr[i] < squeeze_thresh[i]:
                    is_sq[i] = 1
        self.is_squeeze = self.I(lambda: is_sq, name='IS_SQUEEZE')

        # Track consecutive squeeze bars
        sq_streak = np.zeros(len(is_sq))
        for i in range(1, len(is_sq)):
            if is_sq[i] == 1:
                sq_streak[i] = sq_streak[i - 1] + 1
            else:
                sq_streak[i] = 0
        self.sq_streak = self.I(lambda: sq_streak, name='SQ_STREAK')

        # Trade management state
        self.entry_price = None
        self.entry_bar = None
        self.initial_stop = None
        self.tp1 = None
        self.tp2 = None
        self.tp1_hit = False
        self.side = None

        print("🚀 All indicators armed and ready for launch! 🌙")

    def next(self):
        i = len(self.data) - 1
        if i < self.bbw_lookback + 5:
            return

        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        vol = self.data.Volume[-1]

        upper = self.bb_upper[-1]
        lower = self.bb_lower[-1]
        mid = self.bb_mid[-1]
        atr = self.atr[-1]
        vol_sma = self.vol_sma[-1]
        ema = self.ema[-1]
        sq_streak = self.sq_streak[-1]

        if np.isnan(atr) or np.isnan(upper) or np.isnan(vol_sma):
            return

        # ================= MANAGE OPEN POSITION =================
        if self.position:
            bars_held = i - self.entry_bar
            pnl_atr = (price - self.entry_price) / atr if self.side == 'long' else (self.entry_price - price) / atr

            # Stop loss
            if self.side == 'long' and price <= self.initial_stop:
                print(f"🛑 Moon Dev STOP HIT (long) @ {price:.2f} 🌙")
                self.position.close()
                self._reset()
                return
            if self.side == 'short' and price >= self.initial_stop:
                print(f"🛑 Moon Dev STOP HIT (short) @ {price:.2f} 🌙")
                self.position.close()
                self._reset()
                return

            # TP1 - scale out 50%
            if not self.tp1_hit:
                if self.side == 'long' and price >= self.tp1:
                    print(f"✨ TP1 hit (long) - scaling 50% @ {price:.2f} 🚀")
                    self.position.close(0.5)
                    self.tp1_hit = True
                elif self.side == 'short' and price <= self.tp1:
                    print(f"✨ TP1 hit (short) - scaling 50% @ {price:.2f} 🚀")
                    self.position.close(0.5)
                    self.tp1_hit = True

            # TP2 - full exit
            if self.side == 'long' and price >= self.tp2:
                print(f"🎯 TP2 hit (long) @ {price:.2f} - full exit! 🌙💰")
                self.position.close()
                self._reset()
                return
            if self.side == 'short' and price <= self.tp2:
                print(f"🎯 TP2 hit (short) @ {price:.2f} - full exit! 🌙💰")
                self.position.close()
                self._reset()
                return

            # EMA trail
            if self.side == 'long' and price < ema and pnl_atr > 1.0:
                print(f"📉 EMA trail exit (long) @ {price:.2f} 🌙")
                self.position.close()
                self._reset()
                return
            if self.side == 'short' and price > ema and pnl_atr > 1.0:
                print(f"📈 EMA trail exit (short) @ {price:.2f} 🌙")
                self.position.close()
                self._reset()
                return

            # Time stop
            if bars_held >= self.time_stop_bars and pnl_atr < 1.0:
                print(f"⏰ Time stop hit after {bars_held} bars @ {price:.2f} 🌙")
                self.position.close()
                self._reset()
                return

            # Close back inside bands after TP1
            if self.tp1_hit:
                if self.side == 'long' and price < mid:
                    print(f"🌙 Price closed back inside bands (long exit) @ {price:.2f}")
                    self.position.close()
                    self._reset()
                    return
                if self.side == 'short' and price > mid:
                    print(f"🌙 Price closed back inside bands (short exit) @ {price:.2f}")
                    self.position.close()
                    self._reset()
                    return

            return

        # ================= ENTRY LOGIC =================
        # Need squeeze prior to breakout
        if sq_streak < self.squeeze_min_bars:
            return

        # Volume surge
        vol_ok = vol >= self.vol_mult * vol_sma
        if not vol_ok:
            return

        # Long breakout
        if price > upper:
            # funding filter placeholder (no funding data) - treat as sentiment proxy via mid slope
            # Use structure: only long if EMA is rising
            ema_rising = self.ema[-1] > self.ema[-2]
            if not ema_rising:
                return
            print(f"🌙🚀 LONG SQUEEZE BREAKOUT @ {price:.2f} | BBW squeeze streak={sq_streak} | Vol={vol:.2f} vs SMA={vol_sma:.2f}")
            stop = min(lower, low) - 0.1 * atr
            size = int(round(1_000_000 / price))
            if size < 1:
                return
            self.buy(size=size)
            self.entry_price = price
            self.entry_bar = i
            self.initial_stop = stop
            self.tp1 = price + self.atr_tp1_mult * atr
            self.tp2 = price + self.atr_tp2_mult * atr
            self.tp1_hit = False
            self.side = 'long'

        # Short breakout
        elif price < lower:
            ema_falling = self.ema[-1] < self.ema[-2]
            if not ema_falling:
                return
            print(f"🌙🔻 SHORT SQUEEZE BREAKDOWN @ {price:.2f} | BBW squeeze streak={sq_streak} | Vol={vol:.2f} vs SMA={vol_sma:.2f}")
            stop = max(upper, high) + 0.1 * atr
            # Half size for short (lower conviction)
            size = int(round((1_000_000 / price) * 0.5))
            if size < 1:
                return
            self.sell(size=size)
            self.entry_price = price
            self.entry_bar = i
            self.initial_stop = stop
            self.tp1 = price - self.atr_tp1_mult * atr
            self.tp2 = price - self.atr_tp2_mult * atr
            self.tp1_hit = False
            self.side = 'short'

    def _reset(self):
        self.entry_price = None
        self.entry_bar = None
        self.initial_stop = None
        self.tp1 = None
        self.tp2 = None
        self.tp1_hit = False
        self.side = None


print("🌙✨ Moon Dev Backtest Engine warming up... 🚀")
data = load_and_clean_data(DATA_PATH)

bt = Backtest(
    data,
    SqueezeFundingBreakout,
    cash=1_000_000,
    commission=0.002,
    exclusive=False,
)

stats = bt.run()
print(stats)
print(stats._strategy)