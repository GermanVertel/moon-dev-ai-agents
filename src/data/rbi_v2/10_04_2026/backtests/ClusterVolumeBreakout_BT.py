import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# ============================================================
# 🌙 MOON DEV'S CLUSTER VOLUME BREAKOUT STRATEGY 🌙
# ============================================================

def load_and_clean_data(path):
    print("🌙 Loading cosmic data from the Moon Dev vault...")
    data = pd.read_csv(path)
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
    if 'datetime' in data.columns:
        data['datetime'] = pd.to_datetime(data['datetime'])
        data = data.set_index('datetime')
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    print(f"✨ Cleaned data: {len(data)} bars ready for launch 🚀")
    return data


class ClusterVolumeBreakout(Strategy):
    # Strategy parameters
    lookback = 50            # volume profile lookback
    bins = 40                # number of price bins
    value_area_pct = 0.70    # 70% value area
    consolidation_bars = 12  # min bars in consolidation
    squeeze_lookback = 20    # BB width lookback
    vol_ma_period = 20
    vol_mult = 1.5
    atr_period = 14
    bb_period = 20
    bb_std = 2.0
    risk_pct = 0.01          # 1% risk per trade
    htf_ema_period = 200     # higher timeframe trend proxy on same series
    time_stop_bars = 10

    def init(self):
        print("🌙 Initializing Moon Dev indicators...")
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        vol = self.data.Volume

        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)
        self.vol_ma = self.I(talib.SMA, vol, timeperiod=self.vol_ma_period)
        self.ema200 = self.I(talib.EMA, close, timeperiod=self.htf_ema_period)

        self.bb_upper, self.bb_mid, self.bb_lower = self.I(
            talib.BBANDS, close, timeperiod=self.bb_period,
            nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0
        )

        # swing highs/lows for trailing
        self.swing_high = self.I(talib.MAX, high, timeperiod=10)
        self.swing_low = self.I(talib.MIN, low, timeperiod=10)

        # state
        self.trade_state = {}  # track per-position info
        self.last_entry_bar = None

    # ---------- Volume Profile helpers ----------
    def build_volume_profile(self, start_idx, end_idx):
        """Build a volume-at-price profile over [start_idx, end_idx)."""
        highs = self.data.High[start_idx:end_idx]
        lows = self.data.Low[start_idx:end_idx]
        closes = self.data.Close[start_idx:end_idx]
        vols = self.data.Volume[start_idx:end_idx]

        if len(highs) < 5:
            return None

        pmin = float(np.min(lows))
        pmax = float(np.max(highs))
        if pmax <= pmin:
            return None

        edges = np.linspace(pmin, pmax, self.bins + 1)
        bin_vol = np.zeros(self.bins)

        for i in range(len(highs)):
            lo = float(lows[i]); hi = float(highs[i]); v = float(vols[i])
            if hi <= lo:
                # single price bar
                idx = int(np.clip(np.searchsorted(edges, lo) - 1, 0, self.bins - 1))
                bin_vol[idx] += v
                continue
            lo_bin = int(np.clip(np.searchsorted(edges, lo) - 1, 0, self.bins - 1))
            hi_bin = int(np.clip(np.searchsorted(edges, hi) - 1, 0, self.bins - 1))
            span = hi_bin - lo_bin + 1
            bin_vol[lo_bin:hi_bin + 1] += v / span

        centers = (edges[:-1] + edges[1:]) / 2.0
        total = bin_vol.sum()
        if total <= 0:
            return None

        # POC
        poc_idx = int(np.argmax(bin_vol))
        poc = centers[poc_idx]

        # Value area (expand around POC until pct covered)
        order = np.argsort(bin_vol)[::-1]
        target = total * self.value_area_pct
        acc = 0.0
        va_idx = []
        for i in order:
            acc += bin_vol[i]
            va_idx.append(i)
            if acc >= target:
                break
        va_prices = centers[va_idx]
        val = float(np.min(va_prices))
        vah = float(np.max(va_prices))

        # HVN / LVN: top/bottom 20% bins by volume
        sorted_idx = np.argsort(bin_vol)
        n = max(1, self.bins // 5)
        lvn_idx = sorted_idx[:n]
        hvn_idx = sorted_idx[-n:]
        lvn_prices = centers[lvn_idx]
        hvn_prices = centers[hvn_idx]

        return {
            'poc': poc,
            'val': val,
            'vah': vah,
            'hvn': hvn_prices,
            'lvn': lvn_prices,
            'pmin': pmin,
            'pmax': pmax,
        }

    def next(self):
        i = len(self.data) - 1
        if i < max(self.lookback, self.ema200, self.bb_period) + 5:
            return

        # ---------- Manage existing trades ----------
        if self.position:
            self._manage_position(i)
            return

        # ---------- Build volume profile ----------
        start = i - self.lookback
        vp = self.build_volume_profile(start, i)
        if vp is None:
            return

        # ---------- Consolidation detection ----------
        cons_start = i - self.consolidation_bars
        cons_high = float(np.max(self.data.High[cons_start:i]))
        cons_low = float(np.min(self.data.Low[cons_start:i]))
        cons_range = cons_high - cons_low
        if cons_range <= 0:
            return

        # squeeze: BB width contracting
        bw_now = (self.bb_upper[i] - self.bb_lower[i]) / max(self.bb_mid[i], 1e-9)
        bw_prev = (self.bb_upper[i - self.squeeze_lookback] - self.bb_lower[i - self.squeeze_lookback]) / max(self.bb_mid[i - self.squeeze_lookback], 1e-9)
        squeeze = bw_now < bw_prev
        if not squeeze:
            return

        # ---------- Breakout trigger ----------
        close = float(self.data.Close[i])
        vol = float(self.data.Volume[i])
        vol_ma = float(self.vol_ma[i])
        if vol_ma <= 0:
            return
        vol_confirm = vol > self.vol_mult * vol_ma

        long_trigger = close > cons_high and close > vp['vah'] and vol_confirm
        short_trigger = close < cons_low and close < vp['val'] and vol_confirm

        # HTF trend filter
        uptrend = close > float(self.ema200[i])
        downtrend = close < float(self.ema200[i])

        if long_trigger and uptrend:
            self._enter_long(i, cons_high, cons_low, vp)
        elif short_trigger and downtrend:
            self._enter_short(i, cons_high, cons_low, vp)

    # ---------- Entry helpers ----------
    def _enter_long(self, i, cons_high, cons_low, vp):
        entry = float(self.data.Close[i])
        mid = (cons_high + cons_low) / 2.0
        # stop: below POC or consolidation midpoint (whichever is tighter = higher for long)
        stop = max(vp['poc'], mid)
        if stop >= entry:
            stop = min(vp['poc'], mid)
        if stop >= entry:
            print("🌙⚠️ Long stop invalid, skipping")
            return

        risk = entry - stop
        if risk <= 0:
            return

        # nearest HVN above entry for TP1
        hvns_above = [h for h in vp['hvn'] if h > entry]
        tp1 = min(hvns_above) if hvns_above else entry + 1.5 * risk
        # TP2: 2x consolidation range projected
        cons_range = cons_high - cons_low
        tp2 = entry + 2.0 * cons_range

        # R:R filter
        if (tp1 - entry) / risk < 1.0:
            print(f"🌙⚠️ Poor R:R long ({((tp1-entry)/risk):.2f}), skipping")
            return

        equity = self.equity
        risk_amount = equity * self.risk_pct
        size = int(round(risk_amount / risk))
        if size < 1:
            size = 1

        print(f"🌙🚀 MOON DEV LONG BREAKOUT! entry={entry:.2f} stop={stop:.2f} tp1={tp1:.2f} tp2={tp2:.2f} size={size}")
        self.buy(size=size)
        self.trade_state = {
            'side': 'long',
            'entry': entry,
            'stop': stop,
            'tp1': tp1,
            'tp2': tp2,
            'tp1_hit': False,
            'cons_high': cons_high,
            'cons_low': cons_low,
            'bar': i,
            'initial_size': size,
        }
        self.last_entry_bar = i

    def _enter_short(self, i, cons_high, cons_low, vp):
        entry = float(self.data.Close[i])
        mid = (cons_high + cons_low) / 2.0
        # stop: above POC or consolidation midpoint (whichever is tighter = lower for short)
        stop = min(vp['poc'], mid)
        if stop <= entry:
            stop = max(vp['poc'], mid)
        if stop <= entry:
            print("🌙⚠️ Short stop invalid, skipping")
            return

        risk = stop - entry
        if risk <= 0:
            return

        hvns_below = [h for h in vp['hvn'] if h < entry]
        tp1 = max(hvns_below) if hvns_below else entry - 1.5 * risk
        cons_range = cons_high - cons_low
        tp2 = entry - 2.0 * cons_range

        if (entry - tp1) / risk < 1.0:
            print(f"🌙⚠️ Poor R:R short ({((entry-tp1)/risk):.2f}), skipping")
            return

        equity = self.equity
        risk_amount = equity * self.risk_pct
        size = int(round(risk_amount / risk))
        if size < 1:
            size = 1

        print(f"🌙🚀 MOON DEV SHORT BREAKOUT! entry={entry:.2f} stop={stop:.2f} tp1={tp1:.2f} tp2={tp2:.2f} size={size}")
        self.sell(size=size)
        self.trade_state = {
            'side': 'short',
            'entry': entry,
            'stop': stop,
            'tp1': tp1,
            'tp2': tp2,
            'tp1_hit': False,
            'cons_high': cons_high,
            'cons_low': cons_low,
            'bar': i,
            'initial_size': size,
        }
        self.last_entry_bar = i

    # ---------- Position management ----------
    def _manage_position(self, i):
        st = self.trade_state
        if not st:
            return
        price = float(self.data.Close[i])
        side = st['side']

        # Time stop
        if i - st['bar'] >= self.time_stop_bars and not st['tp1_hit']:
            if side == 'long' and price < st['cons_high']:
                print(f"🌙⏰ Time stop long @ {price:.2f}")
                self.position.close()
                self.trade_state = {}
                return
            if side == 'short' and price > st['cons_low']:
                print(f"🌙⏰ Time stop short @ {price:.2f}")
                self.position.close()
                self.trade_state = {}
                return

        if side == 'long':
            # stop loss
            if price <= st['stop']:
                print(f"🌙🛑 Long stop hit @ {price:.2f}")
                self.position.close()
                self.trade_state = {}
                return
            # tp2 - full exit
            if price >= st['tp2']:
                print(f"🌙🎯 TP2 long hit @ {price:.2f}")
                self.position.close()
                self.trade_state = {}
                return
            # tp1 - scale out 50% & trail
            if not st['tp1_hit'] and price >= st['tp1']:
                half = max(1, int(round(st['initial_size'] * 0.5)))
                print(f"🌙✨ TP1 long hit @ {price:.2f} — scaling out {half}")
                try:
                    self.position.close(portion=0.5)
                except Exception:
                    pass
                st['tp1_hit'] = True
                # move stop to breakeven
                st['stop'] = max(st['stop'], st['entry'])
            # trailing stop below swing low
            if st['tp1_hit']:
                trail = float(self.swing_low[i])
                if trail > st['stop']:
                    st['stop'] = trail

        else:  # short
            if price >= st['stop']:
                print(f"🌙🛑 Short stop hit @ {price:.2f}")
                self.position.close()
                self.trade_state = {}
                return
            if price <= st['tp2']:
                print(f"🌙🎯 TP2 short hit @ {price:.2f}")
                self.position.close()
                self.trade_state = {}
                return
            if not st['tp1_hit'] and price <= st['tp1']:
                half = max(1, int(round(st['initial_size'] * 0.5)))
                print(f"🌙✨ TP1 short hit @ {price:.2f} — scaling out {half}")
                try:
                    self.position.close(portion=0.5)
                except Exception:
                    pass
                st['tp1_hit'] = True
                st['stop'] = min(st['stop'], st['entry'])
            if st['tp1_hit']:
                trail = float(self.swing_high[i])
                if trail < st['stop']:
                    st['stop'] = trail


# ============================================================
# 🌙 RUN THE BACKTEST 🚀
# ============================================================
if __name__ == "__main__":
    data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
    data = load_and_clean_data(data_path)

    bt = Backtest(
        data,
        ClusterVolumeBreakout,
        cash=1_000_000,
        commission=0.0002,
        exclusive_orders=True,
    )

    print("🌙✨ Launching Moon Dev Cluster Volume Breakout backtest... 🚀")
    stats = bt.run()
    print(stats)
    print(stats._strategy)