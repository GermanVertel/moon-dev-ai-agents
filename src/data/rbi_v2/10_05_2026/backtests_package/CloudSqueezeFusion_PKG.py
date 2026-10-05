import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# ─── Moon Dev Data Loading ───────────────────────────────────────────────
print("🌙 Moon Dev: Loading BTC-USD 15m data...")
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')

# Clean columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper column mapping
data.rename(columns={
    'datetime': 'Date',
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
print(f"✨ Moon Dev: Data loaded with {len(data)} rows 🚀")


def heikin_ashi(open_, high, low, close):
    """Compute Heikin-Ashi candles."""
    ha_close = (open_ + high + low + close) / 4.0
    ha_open = np.zeros_like(ha_close)
    ha_open[0] = (open_[0] + close[0]) / 2.0
    for i in range(1, len(ha_close)):
        ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2.0
    ha_high = np.maximum.reduce([high, ha_open, ha_close])
    ha_low = np.minimum.reduce([low, ha_open, ha_close])
    return ha_open, ha_high, ha_low, ha_close


class CloudSqueezeFusion(Strategy):
    # Ichimoku
    tenkan_period = 9
    kijun_period = 26
    senkou_b_period = 52
    displacement = 26

    # TTM Squeeze
    bb_period = 20
    bb_mult = 2.0
    kc_period = 20
    kc_mult = 1.5

    # ATR
    atr_period = 14
    atr_mult = 1.5

    # Risk
    risk_pct = 0.02

    def init(self):
        print("🌙 Moon Dev: Initializing CloudSqueeze Fusion indicators...")
        o = self.data.Open
        h = self.data.High
        l = self.data.Low
        c = self.data.Close

        # ─── Heikin-Ashi ─────────────────────────────
        ha_open, ha_high, ha_low, ha_close = heikin_ashi(
            np.asarray(o, dtype=float),
            np.asarray(h, dtype=float),
            np.asarray(l, dtype=float),
            np.asarray(c, dtype=float),
        )
        self.ha_open = self.I(lambda: ha_open, name='HA_Open')
        self.ha_high = self.I(lambda: ha_high, name='HA_High')
        self.ha_low = self.I(lambda: ha_low, name='HA_Low')
        self.ha_close = self.I(lambda: ha_close, name='HA_Close')

        # ─── Ichimoku ────────────────────────────────
        tenkan = (talib.MAX(h, self.tenkan_period) + talib.MIN(l, self.tenkan_period)) / 2.0
        kijun = (talib.MAX(h, self.kijun_period) + talib.MIN(l, self.kijun_period)) / 2.0
        senkou_a = (tenkan + kijun) / 2.0
        senkou_b = (talib.MAX(h, self.senkou_b_period) + talib.MIN(l, self.senkou_b_period)) / 2.0

        # Displacement: shift forward by `displacement` (plot at current bar uses past values)
        senkou_a_shifted = np.full_like(senkou_a, np.nan)
        senkou_b_shifted = np.full_like(senkou_b, np.nan)
        d = self.displacement
        if len(senkou_a) > d:
            senkou_a_shifted[d:] = senkou_a[:-d]
            senkou_b_shifted[d:] = senkou_b[:-d]

        self.tenkan = self.I(lambda: tenkan, name='Tenkan')
        self.kijun = self.I(lambda: kijun, name='Kijun')
        self.senkou_a = self.I(lambda: senkou_a_shifted, name='SenkouA')
        self.senkou_b = self.I(lambda: senkou_b_shifted, name='SenkouB')

        # ─── TTM Squeeze ─────────────────────────────
        bb_basis = talib.SMA(c, self.bb_period)
        bb_dev = self.bb_mult * talib.STDDEV(c, self.bb_period, nbdev=1)
        bb_upper = bb_basis + bb_dev
        bb_lower = bb_basis - bb_dev

        kc_basis = talib.EMA(c, self.kc_period)
        kc_range = talib.EMA(h - l, self.kc_period)
        kc_upper = kc_basis + self.kc_mult * kc_range
        kc_lower = kc_basis - self.kc_mult * kc_range

        squeeze_on = (bb_upper < kc_upper) & (bb_lower > kc_lower)

        # Momentum histogram (LazyBear style approximation)
        highest = talib.MAX(h, self.kc_period)
        lowest = talib.MIN(l, self.kc_period)
        mid = (highest + lowest) / 2.0
        sma_close = talib.SMA(c, self.kc_period)
        delta = c - ((mid + sma_close) / 2.0)

        # Linear regression for momentum value
        mom = talib.LINEARREG(delta, self.kc_period)

        self.bb_upper = self.I(lambda: bb_upper, name='BB_Upper')
        self.bb_lower = self.I(lambda: bb_lower, name='BB_Lower')
        self.kc_upper = self.I(lambda: kc_upper, name='KC_Upper')
        self.kc_lower = self.I(lambda: kc_lower, name='KC_Lower')
        self.squeeze_on = self.I(lambda: squeeze_on.astype(float), name='SqueezeOn')
        self.momentum = self.I(lambda: mom, name='Momentum')

        # ─── ATR ─────────────────────────────────────
        self.atr = self.I(talib.ATR, h, l, c, timeperiod=self.atr_period, name='ATR')

        self.entry_price = None
        self.stop_price = None
        self.tp_price = None

        print("✨ Moon Dev: Indicators ready! 🚀")

    def next(self):
        # Need enough bars
        if len(self.data) < self.senkou_b_period + self.displacement + 5:
            return

        price = self.data.Close[-1]
        cloud_top = max(self.senkou_a[-1], self.senkou_b[-1])
        cloud_bot = min(self.senkou_a[-1], self.senkou_b[-1])

        ha_c = self.ha_close[-1]
        ha_o = self.ha_open[-1]
        ha_l = self.ha_low[-1]

        ha_bull = (ha_c > ha_o) and (ha_l >= min(ha_o, ha_c) - 1e-9)

        above_cloud = price > cloud_top and ha_c > cloud_top
        tk_bull = self.tenkan[-1] > self.kijun[-1]

        mom_now = self.momentum[-1]
        mom_prev = self.momentum[-2]
        mom_flip_pos = (mom_prev <= 0) and (mom_now > 0)

        # ─── Manage open position ────────────────────
        if self.position:
            # Cloud flip exit
            if price < cloud_bot:
                print(f"🌙 Moon Dev EXIT: Cloud flip below Kumo @ {price:.2f} 💫")
                self.position.close()
                self.entry_price = None
                return

            # Momentum exit
            if mom_now < 0:
                print(f"🌙 Moon Dev EXIT: Momentum turned negative @ {price:.2f} 🔻")
                self.position.close()
                self.entry_price = None
                return

            # HA reversal: two consecutive bearish HA candles
            ha_bear_1 = self.ha_close[-1] < self.ha_open[-1]
            ha_bear_2 = self.ha_close[-2] < self.ha_open[-2]
            if ha_bear_1 and ha_bear_2:
                print(f"🌙 Moon Dev EXIT: Two bearish HA candles @ {price:.2f} 🐻")
                self.position.close()
                self.entry_price = None
                return

            # Stop loss / take profit
            if self.stop_price and price <= self.stop_price:
                print(f"🌙 Moon Dev STOP HIT @ {price:.2f} 🛑")
                self.position.close()
                self.entry_price = None
                return

            if self.tp_price and price >= self.tp_price:
                print(f"🌙 Moon Dev TP HIT @ {price:.2f} 🎯")
                self.position.close()
                self.entry_price = None
                return
            return

        # ─── Entry logic ─────────────────────────────
        if above_cloud and tk_bull and mom_flip_pos:
            atr_val = self.atr[-1]
            if np.isnan(atr_val) or atr_val <= 0:
                return

            entry = price
            stop = entry - self.atr_mult * atr_val
            # Tighter of Kijun or ATR stop
            kijun_stop = self.kijun[-1]
            stop = max(stop, min(kijun_stop, entry - 0.5 * atr_val))
            risk = entry - stop
            if risk <= 0:
                return

            tp = entry + 2 * risk

            # Position sizing: risk_pct of equity
            equity = self.equity
            risk_amount = equity * self.risk_pct
            size = risk_amount / risk
            size = int(round(size))
            if size < 1:
                size = 1

            self.entry_price = entry
            self.stop_price = stop
            self.tp_price = tp

            print(f"🚀 Moon Dev ENTRY LONG @ {entry:.2f} | SL {stop:.2f} | TP {tp:.2f} | size {size} 🌙")
            self.buy(size=size)


# ─── Run Backtest ────────────────────────────────────────────────────────
print("🌙 Moon Dev: Launching CloudSqueeze Fusion backtest... ✨")
bt = Backtest(data, CloudSqueezeFusion, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)