import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from scipy.stats import norm

DATA_PATH = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"

print("🌙 Moon Dev's StochasticMertonBands backtest booting up... ✨🚀")

data = pd.read_csv(DATA_PATH)
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
print(f"🌙 Loaded {len(data)} bars of lunar data ✨")


def merton_p_hit(S, P, sigma, T):
    if sigma <= 0 or T <= 0 or S <= 0 or P <= 0:
        return 0.0
    z = -abs(np.log(P / S)) / (sigma * np.sqrt(T))
    return float(2.0 * norm.cdf(z))


class StochasticMertonBands(Strategy):
    bb_period = 20
    bb_std_dev = 2.0
    bbw_rank_window = 100
    rsi_period = 14
    sma_regime_period = 200
    target_vol = 0.02
    base_size = 1_000_000
    risk_per_trade = 0.01
    max_concurrent = 3
    daily_loss_limit = 0.03
    T_bars = 10
    bars_per_year = 35040  # 15m bars

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low

        self.bb_mid = self.I(talib.SMA, close, timeperiod=self.bb_period)
        self.bb_std = self.I(talib.STDDEV, close, timeperiod=self.bb_period, nbdev=1.0)
        self.bb_upper = self.I(lambda m, s: m + self.bb_std_dev * s, self.bb_mid, self.bb_std)
        self.bb_lower = self.I(lambda m, s: m - self.bb_std_dev * s, self.bb_mid, self.bb_std)
        self.bandwidth = self.I(lambda u, l, m: (u - l) / (m + 1e-12), self.bb_upper, self.bb_lower, self.bb_mid)
        self.pct_b = self.I(lambda c, l, u: (c - l) / (u - l + 1e-12), close, self.bb_lower, self.bb_upper)
        self.rsi = self.I(talib.RSI, close, timeperiod=self.rsi_period)
        self.sma_regime = self.I(talib.SMA, close, timeperiod=self.sma_regime_period)

        # Compute log returns safely with pandas
        close_series = pd.Series(np.asarray(close, dtype=float))
        log_ret_series = np.log(close_series / close_series.shift(1)).fillna(0.0).values
        self.log_ret = self.I(lambda: log_ret_series)
        self.realized_vol = self.I(talib.STDDEV, self.log_ret, timeperiod=self.bb_period, nbdev=1.0)

        self.bbw_rank = self.I(self._rolling_pct_rank, self.bandwidth, self.bbw_rank_window)

        self.entry_price = None
        self.stop_price = None
        self.take_price = None
        self.hard_stop = None
        self.prev_pct_b = None
        self.prev_rsi = None
        self.day_start_equity = None
        self.current_day = None
        self.trading_halted_today = False

    def _rolling_pct_rank(self, series, window):
        series = np.asarray(series, dtype=float)
        out = np.full(len(series), np.nan)
        for i in range(window, len(series)):
            w = series[i - window:i]
            v = series[i]
            if np.isnan(v) or np.any(np.isnan(w)):
                continue
            out[i] = np.sum(w <= v) / window
        return out

    def _compute_merton_levels(self, price, sigma_bar):
        sigma_ann = sigma_bar * np.sqrt(self.bars_per_year)
        if sigma_ann <= 0 or np.isnan(sigma_ann):
            return None, None, None, None
        T = self.T_bars / self.bars_per_year
        sigma_T = sigma_ann * np.sqrt(T)
        if sigma_T <= 0:
            return None, None, None, None

        stop_mults = [0.5, 1.0, 1.5, 2.0]
        stop_levels = [price * (1 - m * sigma_T) for m in stop_mults]
        stop_probs = [merton_p_hit(price, lv, sigma_ann, T) for lv in stop_levels]

        lim_mults = [0.5, 1.0, 1.5, 2.0]
        lim_levels = [price * (1 + m * sigma_T) for m in lim_mults]
        lim_probs = [merton_p_hit(price, lv, sigma_ann, T) for lv in lim_levels]

        sp = np.array(stop_probs)
        if sp.sum() > 0:
            weighted_stop = float(np.sum(np.array(stop_levels) * sp) / sp.sum())
        else:
            weighted_stop = stop_levels[1]

        lp = 1.0 - np.array(lim_probs)
        if lp.sum() > 0:
            weighted_limit = float(np.sum(np.array(lim_levels) * lp) / lp.sum())
        else:
            weighted_limit = lim_levels[1]

        # clamp stop within [0.5σ, 2.5σ]
        min_stop = price * (1 - 0.5 * sigma_T)
        max_stop = price * (1 - 2.5 * sigma_T)
        weighted_stop = float(np.clip(weighted_stop, max_stop, min_stop))

        return weighted_stop, weighted_limit, sigma_ann, sigma_T

    def next(self):
        i = len(self.data) - 1
        if i < max(self.bbw_rank_window, self.sma_regime_period) + 5:
            return

        price = self.data.Close[-1]
        current_day = self.data.index[-1].date()

        if self.current_day != current_day:
            self.current_day = current_day
            self.day_start_equity = self.equity
            self.trading_halted_today = False

        if self.day_start_equity is not None and self.day_start_equity > 0:
            day_pnl = (self.equity - self.day_start_equity) / self.day_start_equity
            if day_pnl <= -self.daily_loss_limit:
                self.trading_halted_today = True

        # Manage open position
        if self.position:
            # divergence exit
            if self.prev_pct_b is not None and self.prev_rsi is not None:
                pb_now = self.pct_b[-1]
                rsi_now = self.rsi[-1]
                if (not np.isnan(pb_now) and not np.isnan(rsi_now)
                        and not np.isnan(self.prev_pct_b) and not np.isnan(self.prev_rsi)):
                    if pb_now > self.prev_pct_b and rsi_now < self.prev_rsi and price > self.bb_mid[-1]:
                        print(f"🌙✨ Divergence exit triggered! %B up, RSI down. Closing long. 🚀")
                        self.position.close()
                        self.entry_price = None
                        self.prev_pct_b = self.pct_b[-1]
                        self.prev_rsi = self.rsi[-1]
                        return

            # volatility expansion exit
            if not np.isnan(self.bbw_rank[-1]) and self.bbw_rank[-1] > 0.75:
                print(f"🌙✨ Volatility expansion exit! BBW rank={self.bbw_rank[-1]:.2f}. Closing. 🚀")
                self.position.close()
                self.entry_price = None
                self.prev_pct_b = self.pct_b[-1]
                self.prev_rsi = self.rsi[-1]
                return

            # stop / target
            if self.stop_price is not None and price <= self.stop_price:
                print(f"🌙💥 Probability-weighted stop hit at {price:.2f} (stop={self.stop_price:.2f})")
                self.position.close()
                self.entry_price = None
                self.prev_pct_b = self.pct_b[-1]
                self.prev_rsi = self.rsi[-1]
                return
            if self.hard_stop is not None and price <= self.hard_stop:
                print(f"🌙💥 HARD stop hit at {price:.2f} (hard={self.hard_stop:.2f})")
                self.position.close()
                self.entry_price = None
                self.prev_pct_b = self.pct_b[-1]
                self.prev_rsi = self.rsi[-1]
                return
            if self.take_price is not None and price >= self.take_price:
                print(f"🌙🎯 Take-profit hit at {price:.2f} (target={self.take_price:.2f}) 🚀")
                self.position.close()
                self.entry_price = None
                self.prev_pct_b = self.pct_b[-1]
                self.prev_rsi = self.rsi[-1]
                return

        # Entry
        if not self.position and not self.trading_halted_today:
            bw_rank = self.bbw_rank[-1]
            lower = self.bb_lower[-1]
            mid = self.bb_mid[-1]
            std = self.bb_std[-1]
            rvol = self.realized_vol[-1]
            regime_ok = True
            if not np.isnan(self.sma_regime[-1]) and not np.isnan(self.sma_regime[-2]):
                if price < self.sma_regime[-1] and self.sma_regime[-1] < self.sma_regime[-2]:
                    regime_ok = False

            if (not np.isnan(bw_rank) and bw_rank <= 0.25
                    and not np.isnan(lower) and not np.isnan(std)
                    and price <= lower + 0.25 * std
                    and regime_ok
                    and not np.isnan(rvol) and rvol > 0):

                sigma_ann = rvol * np.sqrt(self.bars_per_year)
                T = self.T_bars / self.bars_per_year
                sigma_T = sigma_ann * np.sqrt(T)
                if sigma_T > 0:
                    p_break_lower = merton_p_hit(price, lower, sigma_ann, T)
                    p_stay = 1.0 - p_break_lower
                    if p_stay > 0.60:
                        stop_p, limit_p, sigma_ann2, sigma_T2 = self._compute_merton_levels(price, rvol)
                        if stop_p is not None:
                            stop_dist = price - stop_p
                            if stop_dist <= 0:
                                stop_dist = 0.5 * sigma_T2 * price
                                stop_p = price - stop_dist

                            risk_amount = self.equity * self.risk_per_trade
                            size_units = risk_amount / stop_dist
                            vol_scalar = min(2.0, self.target_vol / max(rvol, 1e-6))
                            size_units *= vol_scalar
                            size_units = min(size_units, self.base_size)
                            # Fraction of equity sizing (0 < size < 1)
                            cash_per_unit = price
                            equity_val = self.equity
                            if equity_val > 0 and cash_per_unit > 0:
                                size_fraction = (size_units * cash_per_unit) / equity_val
                                # Cap at 0.95 to avoid margin issues
                                size_fraction = min(size_fraction, 0.95)
                                if size_fraction > 0.001:
                                    print(f"🌙🚀 LONG entry! price={price:.2f} stop={stop_p:.2f} "
                                          f"target={limit_p:.2f} size_frac={size_fraction:.4f} "
                                          f"bw_rank={bw_rank:.2f} p_stay={p_stay:.2f} ✨")
                                    self.buy(size=size_fraction)
                                    self.entry_price = price
                                    self.stop_price = stop_p
                                    self.take_price = limit_p
                                    self.hard_stop = price * (1 - 2.0 * sigma_T2)

        self.prev_pct_b = self.pct_b[-1]
        self.prev_rsi = self.rsi[-1]


bt = Backtest(data, StochasticMertonBands, cash=1_000_000, commission=0.0002)
stats = bt.run()
print(stats)
print(stats._strategy)