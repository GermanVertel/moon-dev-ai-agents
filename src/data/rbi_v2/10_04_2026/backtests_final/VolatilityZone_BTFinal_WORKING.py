import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy


def load_data(path):
    data = pd.read_csv(path)
    data.columns = data.columns.str.strip().str.lower()
    data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
    data = data.rename(columns={
        'open': 'Open', 'high': 'High', 'low': 'Low',
        'close': 'Close', 'volume': 'Volume'
    })
    if 'datetime' in data.columns:
        data['datetime'] = pd.to_datetime(data['datetime'])
        data = data.set_index('datetime')
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    return data


class VolatilityZone(Strategy):
    atr_period = 14
    atr_sma_period = 50
    bb_period = 20
    bb_dev = 2.0
    ema_fast = 20
    ema_mid = 50
    ema_slow = 200
    rsi_period = 14
    vol_sma_period = 20
    swing_lookback = 5
    zone_tolerance_atr = 0.5
    risk_pct = 0.01

    def init(self):
        h = self.data.High
        l = self.data.Low
        c = self.data.Close
        v = self.data.Volume

        self.atr = self.I(talib.ATR, h, l, c, timeperiod=self.atr_period)
        self.atr_sma = self.I(talib.SMA, self.atr, timeperiod=self.atr_sma_period)

        def _bb_upper(close):
            u, m, lo = talib.BBANDS(
                close, timeperiod=self.bb_period,
                nbdevup=self.bb_dev, nbdevdn=self.bb_dev, matype=0
            )
            return u

        def _bb_middle(close):
            u, m, lo = talib.BBANDS(
                close, timeperiod=self.bb_period,
                nbdevup=self.bb_dev, nbdevdn=self.bb_dev, matype=0
            )
            return m

        def _bb_lower(close):
            u, m, lo = talib.BBANDS(
                close, timeperiod=self.bb_period,
                nbdevup=self.bb_dev, nbdevdn=self.bb_dev, matype=0
            )
            return lo

        self.bb_upper = self.I(_bb_upper, c)
        self.bb_middle = self.I(_bb_middle, c)
        self.bb_lower = self.I(_bb_lower, c)

        self.ema20 = self.I(talib.EMA, c, timeperiod=self.ema_fast)
        self.ema50 = self.I(talib.EMA, c, timeperiod=self.ema_mid)
        self.ema200 = self.I(talib.EMA, c, timeperiod=self.ema_slow)
        self.rsi = self.I(talib.RSI, c, timeperiod=self.rsi_period)
        self.vol_sma = self.I(talib.SMA, v, timeperiod=self.vol_sma_period)

        self.swing_high = self.I(talib.MAX, h, timeperiod=self.swing_lookback * 2 + 1)
        self.swing_low = self.I(talib.MIN, l, timeperiod=self.swing_lookback * 2 + 1)

        print("🌙✨ VolatilityZone indicators initialized! 🚀")

    def vol_ratio(self):
        a = self.atr[-1]
        s = self.atr_sma[-1]
        if a is None or s is None or s == 0 or np.isnan(a) or np.isnan(s):
            return 1.0
        return a / s

    def bbw(self):
        u = self.bb_upper[-1]
        l = self.bb_lower[-1]
        m = self.bb_middle[-1]
        if m is None or m == 0 or np.isnan(m):
            return 0.0
        return (u - l) / m

    def next(self):
        if len(self.data) < self.ema_slow + 10:
            return

        price = self.data.Close[-1]
        atr = self.atr[-1]
        if atr is None or np.isnan(atr) or atr <= 0:
            return

        ratio = self.vol_ratio()
        bbw = self.bbw()
        rsi = self.rsi[-1]
        vol = self.data.Volume[-1]
        vol_avg = self.vol_sma[-1]

        ema20 = self.ema20[-1]
        ema50 = self.ema50[-1]
        ema200 = self.ema200[-1]

        if (np.isnan(ema20) or np.isnan(ema50) or np.isnan(ema200)
                or np.isnan(rsi) or np.isnan(vol_avg)):
            return

        sh = self.swing_high[-1]
        sl = self.swing_low[-1]
        if np.isnan(sh) or np.isnan(sl):
            return
        tol = self.zone_tolerance_atr * atr

        if ratio < 0.85:
            regime = "COMPRESSION"
        elif ratio > 1.15:
            regime = "EXPANSION"
        else:
            regime = "NORMAL"

        bull_trend = ema50 > ema200
        bear_trend = ema50 < ema200
        momentum_up = ema20 > self.ema20[-2]
        momentum_dn = ema20 < self.ema20[-2]

        o = self.data.Open[-1]
        h = self.data.High[-1]
        l = self.data.Low[-1]
        c = self.data.Close[-1]
        body = abs(c - o)
        rng = h - l if h > l else 1e-9
        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l

        hammer = (lower_wick > 2 * body) and (upper_wick < body) and (body / rng < 0.4)
        shooting = (upper_wick > 2 * body) and (lower_wick < body) and (body / rng < 0.4)

        prev_o = self.data.Open[-2]
        prev_c = self.data.Close[-2]
        bull_engulf = (c > o) and (prev_c < prev_o) and (c > prev_o) and (o < prev_c)
        bear_engulf = (c < o) and (prev_c > prev_o) and (c < prev_o) and (o > prev_c)

        near_support = (price <= sl + tol) and (price >= sl - tol)
        near_resistance = (price >= sh - tol) and (price <= sh + tol)

        vol_confirm = (vol_avg is not None and not np.isnan(vol_avg)
                       and vol_avg > 0 and vol > 1.5 * vol_avg)

        if self.position:
            entry = self.trades[-1].entry_price
            is_long = self.position.is_long
            pnl_dist = (price - entry) if is_long else (entry - price)

            if len(self.trades) > 0 and (len(self.data) - self.trades[-1].entry_bar) > 20:
                if pnl_dist < atr * 0.5:
                    print(f"🌙⏰ Time stop hit! Exiting at {price:.2f}")
                    self.position.close()
                    return

            if is_long and c < sl - tol:
                print(f"🌙❌ Long invalidated, closing at {price:.2f}")
                self.position.close()
                return
            if (not is_long) and c > sh + tol:
                print(f"🌙❌ Short invalidated, closing at {price:.2f}")
                self.position.close()
                return
            return

        # ---- ENTRY LOGIC ----
        if (regime in ("COMPRESSION", "NORMAL")
                and near_support
                and ratio < 1.15
                and (hammer or bull_engulf)
                and rsi < 40
                and bull_trend):
            stop = sl - 0.5 * atr
            risk = price - stop
            if risk > 0:
                risk_amount = self.equity * self.risk_pct
                size = risk_amount / price
                if size > 0:
                    tp = price + 1.5 * atr
                    print(f"🌙🚀 LONG REVERSAL @ {price:.2f} | SL {stop:.2f} | TP {tp:.2f} | regime={regime} | RSI={rsi:.1f}")
                    self.buy(size=size, sl=stop, tp=tp)
                    return

        if (regime in ("COMPRESSION", "NORMAL")
                and near_resistance
                and ratio < 1.15
                and (shooting or bear_engulf)
                and rsi > 60
                and bear_trend):
            stop = sh + 0.5 * atr
            risk = stop - price
            if risk > 0:
                risk_amount = self.equity * self.risk_pct
                size = risk_amount / price
                if size > 0:
                    tp = price - 1.5 * atr
                    print(f"🌙🔻 SHORT REVERSAL @ {price:.2f} | SL {stop:.2f} | TP {tp:.2f} | regime={regime} | RSI={rsi:.1f}")
                    self.sell(size=size, sl=stop, tp=tp)
                    return

        if (regime in ("EXPANSION", "NORMAL")
                and price > sh + 0.25 * atr
                and ratio > 1.15
                and vol_confirm
                and bull_trend
                and momentum_up):
            stop = sh - 0.5 * atr
            risk = price - stop
            if risk > 0:
                risk_amount = self.equity * self.risk_pct
                size = risk_amount / price
                if size > 0:
                    tp = price + 2.0 * atr
                    print(f"🌙💥 LONG BREAKOUT @ {price:.2f} | SL {stop:.2f} | TP {tp:.2f} | regime={regime} | BBW={bbw:.4f}")
                    self.buy(size=size, sl=stop, tp=tp)
                    return

        if (regime in ("EXPANSION", "NORMAL")
                and price < sl - 0.25 * atr
                and ratio > 1.15
                and vol_confirm
                and bear_trend
                and momentum_dn):
            stop = sl + 0.5 * atr
            risk = stop - price
            if risk > 0:
                risk_amount = self.equity * self.risk_pct
                size = risk_amount / price
                if size > 0:
                    tp = price - 2.0 * atr
                    print(f"🌙💥 SHORT BREAKOUT @ {price:.2f} | SL {stop:.2f} | TP {tp:.2f} | regime={regime} | BBW={bbw:.4f}")
                    self.sell(size=size, sl=stop, tp=tp)
                    return


if __name__ == "__main__":
    data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
    print("🌙 Loading Moon Dev data... 🚀")
    df = load_data(data_path)
    print(f"✨ Data loaded: {len(df)} bars")

    bt = Backtest(df, VolatilityZone, cash=1_000_000, commission=0.001, exclusive_orders=True)
    stats = bt.run()
    print(stats)
    print(stats._strategy)