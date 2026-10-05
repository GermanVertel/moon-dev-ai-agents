import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# ============================================================
# 🌙 MOON DEV BACKTEST AI - VortexVolumeSync Strategy 🚀
# ============================================================

DATA_PATH = '/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv'


class VortexVolumeSync(Strategy):
    # --- Strategy Parameters ---
    vi_period = 14
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    vol_sma_period = 20
    kc_ema_period = 20
    kc_atr_period = 20
    kc_mult = 2.0
    kc_width_sma_period = 20
    atr_period = 20
    risk_pct = 0.01
    stop_atr_mult = 1.5
    tp_atr_mult = 2.0
    vol_expansion_mult = 1.5
    time_stop_bars = 15

    def init(self):
        print("🌙✨ Initializing VortexVolumeSync indicators...")

        high = self.data.High
        low = self.data.Low
        close = self.data.Close
        volume = self.data.Volume

        # ---------- Vortex Indicator ----------
        # True range
        self.tr = self.I(talib.TRANGE, high, low, close)
        # Sum of TR over period
        self.tr_sum = self.I(talib.SUM, self.tr, timeperiod=self.vi_period)

        # VM+ and VM- using custom numpy
        h_arr = np.array(high)
        l_arr = np.array(low)
        vm_plus_raw = np.abs(h_arr - np.roll(l_arr, 1))
        vm_minus_raw = np.abs(l_arr - np.roll(h_arr, 1))
        vm_plus_raw[0] = 0
        vm_minus_raw[0] = 0

        self.vm_plus_sum = self.I(talib.SUM, vm_plus_raw, timeperiod=self.vi_period)
        self.vm_minus_sum = self.I(talib.SUM, vm_minus_raw, timeperiod=self.vi_period)

        def vi_plus_fn():
            tr_s = self.tr_sum
            vm_s = self.vm_plus_sum
            out = np.where(tr_s > 0, vm_s / tr_s, 0)
            return out

        def vi_minus_fn():
            tr_s = self.tr_sum
            vm_s = self.vm_minus_sum
            out = np.where(tr_s > 0, vm_s / tr_s, 0)
            return out

        self.vi_plus = self.I(vi_plus_fn)
        self.vi_minus = self.I(vi_minus_fn)

        # ---------- MACD ----------
        self.macd, self.macd_signal, self.macd_hist = self.I(
            talib.MACD, close,
            fastperiod=self.macd_fast,
            slowperiod=self.macd_slow,
            signalperiod=self.macd_signal
        )

        # Volume factor
        self.vol_sma = self.I(talib.SMA, volume, timeperiod=self.vol_sma_period)

        def vol_factor_fn():
            vs = self.vol_sma
            v = np.array(volume)
            out = np.where(vs > 0, v / vs, 0)
            return out

        self.vol_factor = self.I(vol_factor_fn)

        def vw_hist_fn():
            return self.macd_hist * self.vol_factor

        self.vw_hist = self.I(vw_hist_fn)

        # ---------- Keltner Channels ----------
        self.kc_ema = self.I(talib.EMA, close, timeperiod=self.kc_ema_period)
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.kc_atr_period)

        def kc_width_fn():
            return 2 * self.kc_mult * self.atr

        self.kc_width = self.I(kc_width_fn)
        self.kc_width_sma = self.I(talib.SMA, self.kc_width, timeperiod=self.kc_width_sma_period)

        # Track entry bar & entry price for time stop
        self.entry_bar = None
        self.entry_price = None

        print("🌙✨ Indicators ready! Ready to hunt trends 🚀")

    def next(self):
        # Wait for enough data
        if len(self.data) < max(self.macd_slow, self.kc_width_sma_period, self.vi_period) + 5:
            return

        price = self.data.Close[-1]

        # ---------- Current indicator values ----------
        vi_p = self.vi_plus[-1]
        vi_m = self.vi_minus[-1]
        vw_h = self.vw_hist[-1]
        vw_h_prev = self.vw_hist[-2]
        macd_now = self.macd[-1]
        macd_sig_now = self.macd_signal[-1]
        macd_prev = self.macd[-2]
        macd_sig_prev = self.macd_signal[-2]
        volf = self.vol_factor[-1]
        kc_w = self.kc_width[-1]
        kc_w_sma = self.kc_width_sma[-1]
        atr_now = self.atr[-1]

        if np.isnan(vi_p) or np.isnan(vi_m) or np.isnan(vw_h) or np.isnan(kc_w_sma) or np.isnan(atr_now):
            return

        # Crossover detection (replaced backtesting.lib.crossover)
        bull_cross = macd_prev < macd_sig_prev and macd_now > macd_sig_now
        bear_cross = macd_prev > macd_sig_prev and macd_now < macd_sig_now

        vol_compressed = kc_w < kc_w_sma
        vol_expanded = kc_w > self.vol_expansion_mult * kc_w_sma

        # ---------- Manage open position ----------
        if self.position:
            is_long = self.position.is_long
            bars_held = len(self.data) - self.entry_bar if self.entry_bar else 0

            # Primary exit: opposite VW-MACD crossover
            if is_long and bear_cross:
                print(f"🌙 Exit LONG: VW-MACD bearish crossover @ {price:.2f}")
                self.position.close()
                self.entry_bar = None
                return
            if not is_long and bull_cross:
                print(f"🌙 Exit SHORT: VW-MACD bullish crossover @ {price:.2f}")
                self.position.close()
                self.entry_bar = None
                return

            # Trend invalidation: Vortex flip
            if is_long and vi_m > vi_p:
                print(f"🌙 Exit LONG: Vortex flip (VI- > VI+) @ {price:.2f}")
                self.position.close()
                self.entry_bar = None
                return
            if not is_long and vi_p > vi_m:
                print(f"🌙 Exit SHORT: Vortex flip (VI+ > VI-) @ {price:.2f}")
                self.position.close()
                self.entry_bar = None
                return

            # Volatility expansion exit
            if vol_expanded:
                print(f"🌙 Exit: KC width expanded > 1.5x avg @ {price:.2f}")
                self.position.close()
                self.entry_bar = None
                return

            # Time stop
            if bars_held >= self.time_stop_bars:
                favorable = (price > self.entry_price) if is_long else (price < self.entry_price)
                if not favorable:
                    print(f"🌙 Exit: Time stop ({bars_held} bars) @ {price:.2f}")
                    self.position.close()
                    self.entry_bar = None
                    return

            return  # already in a position, no new entries

        # ---------- Entry Logic ----------
        long_signal = (
            vi_p > vi_m and
            bull_cross and
            vw_h > 0 and
            vol_compressed and
            volf > 1.0
        )

        short_signal = (
            vi_m > vi_p and
            bear_cross and
            vw_h < 0 and
            vol_compressed and
            volf > 1.0
        )

        if long_signal:
            stop_price = price - self.stop_atr_mult * atr_now
            risk_per_unit = price - stop_price
            if risk_per_unit <= 0:
                return
            equity = self.equity
            risk_amount = equity * self.risk_pct
            size = int(round(risk_amount / risk_per_unit))
            if size < 1:
                return
            print(f"🚀🌙 LONG ENTRY @ {price:.2f} | VI+={vi_p:.3f} VI-={vi_m:.3f} | VWh={vw_h:.4f} | VolF={volf:.2f} | size={size}")
            self.buy(size=size, sl=stop_price)
            self.entry_bar = len(self.data)
            self.entry_price = price

        elif short_signal:
            stop_price = price + self.stop_atr_mult * atr_now
            risk_per_unit = stop_price - price
            if risk_per_unit <= 0:
                return
            equity = self.equity
            risk_amount = equity * self.risk_pct
            size = int(round(risk_amount / risk_per_unit))
            if size < 1:
                return
            print(f"🚀🌙 SHORT ENTRY @ {price:.2f} | VI+={vi_p:.3f} VI-={vi_m:.3f} | VWh={vw_h:.4f} | VolF={volf:.2f} | size={size}")
            self.sell(size=size, sl=stop_price)
            self.entry_bar = len(self.data)
            self.entry_price = price


# ============================================================
# 🌙 DATA LOADING & BACKTEST EXECUTION 🚀
# ============================================================
print("🌙 Loading Moon Dev data...")
data = pd.read_csv(DATA_PATH)

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Ensure proper casing
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
print(f"🌙 Data loaded: {len(data)} bars ✨")

bt = Backtest(
    data,
    VortexVolumeSync,
    cash=1_000_000,
    commission=0.001,
    exclusive_orders=True
)

stats = bt.run()
print(stats)
print(stats._strategy)