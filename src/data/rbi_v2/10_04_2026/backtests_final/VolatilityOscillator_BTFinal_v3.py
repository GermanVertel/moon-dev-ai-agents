import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

print("🌙✨🚀 Moon Dev Backtest AI Initializing VolatilityOscillator Strategy...")

data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume',
})

if 'datetime' in data.columns:
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')
    data.index.name = 'Datetime'

data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

print(f"🌙 Data loaded: {len(data)} rows")
print(f"🌙 Columns: {list(data.columns)}")


class VolatilityOscillator(Strategy):
    bb_period = 20
    bb_std = 2.0
    stoch_k = 14
    stoch_smooth_k = 3
    stoch_smooth_d = 3
    atr_period = 14
    sma_trend_period = 200
    risk_pct = 0.02
    atr_mult = 1.5
    time_exit_bars = 12

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low

        self.bb_mid = self.I(talib.SMA, close, timeperiod=self.bb_period, name="BB_Mid")
        self.bb_std_arr = self.I(talib.STDDEV, close, timeperiod=self.bb_period, nbdev=1, name="BB_STD")
        self.bb_upper = self.I(lambda c, m, s: m + self.bb_std * s, close, self.bb_mid, self.bb_std_arr, name="BB_Upper")
        self.bb_lower = self.I(lambda c, m, s: m - self.bb_std * s, close, self.bb_mid, self.bb_std_arr, name="BB_Lower")

        stoch_k, stoch_d = talib.STOCH(
            high, low, close,
            fastk_period=self.stoch_k,
            slowk_period=self.stoch_smooth_k,
            slowk_matype=0,
            slowd_period=self.stoch_smooth_d,
            slowd_matype=0,
        )
        self.stoch_k_arr = self.I(lambda x: x, stoch_k, name="Stoch_K")
        self.stoch_d_arr = self.I(lambda x: x, stoch_d, name="Stoch_D")

        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period, name="ATR")
        self.sma_trend = self.I(talib.SMA, close, timeperiod=self.sma_trend_period, name="SMA200")

        self.entry_bar = None
        self.stop_price = None
        self.tp_price = None

        print("🌙✨ Indicators initialized successfully! 🚀")

    def next(self):
        if len(self.data) < self.sma_trend_period + 5:
            return

        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]

        bb_up = self.bb_upper[-1]
        bb_dn = self.bb_lower[-1]
        bb_mid = self.bb_mid[-1]
        k = self.stoch_k_arr[-1]
        d = self.stoch_d_arr[-1]
        k_prev = self.stoch_k_arr[-2]
        d_prev = self.stoch_d_arr[-2]
        atr = self.atr[-1]
        trend = self.sma_trend[-1]

        if (np.isnan(bb_up) or np.isnan(bb_dn) or np.isnan(bb_mid)
                or np.isnan(k) or np.isnan(d) or np.isnan(k_prev) or np.isnan(d_prev)
                or np.isnan(atr) or np.isnan(trend)):
            return

        # Manage open position
        if self.position:
            bars_held = len(self.data) - self.entry_bar

            if self.position.is_long:
                if price >= self.tp_price:
                    print(f"🌙✅ LONG TP HIT @ {price:.2f} | TP={self.tp_price:.2f}")
                    self.position.close()
                    self.entry_bar = None
                    return
                if price <= self.stop_price:
                    print(f"🌙❌ LONG SL HIT @ {price:.2f} | SL={self.stop_price:.2f}")
                    self.position.close()
                    self.entry_bar = None
                    return
                if k > 80 and k_prev <= 80:
                    print(f"🌙💫 LONG EXIT: Stoch overbought cross @ {price:.2f}")
                    self.position.close()
                    self.entry_bar = None
                    return
                if bars_held >= self.time_exit_bars:
                    print(f"🌙⏰ LONG TIME EXIT after {bars_held} bars @ {price:.2f}")
                    self.position.close()
                    self.entry_bar = None
                    return

            elif self.position.is_short:
                if price <= self.tp_price:
                    print(f"🌙✅ SHORT TP HIT @ {price:.2f} | TP={self.tp_price:.2f}")
                    self.position.close()
                    self.entry_bar = None
                    return
                if price >= self.stop_price:
                    print(f"🌙❌ SHORT SL HIT @ {price:.2f} | SL={self.stop_price:.2f}")
                    self.position.close()
                    self.entry_bar = None
                    return
                if k < 20 and k_prev >= 20:
                    print(f"🌙💫 SHORT EXIT: Stoch oversold cross @ {price:.2f}")
                    self.position.close()
                    self.entry_bar = None
                    return
                if bars_held >= self.time_exit_bars:
                    print(f"🌙⏰ SHORT TIME EXIT after {bars_held} bars @ {price:.2f}")
                    self.position.close()
                    self.entry_bar = None
                    return
            return

        # Long entry: price touches lower BB, stoch cross up in oversold, trend filter
        long_cond = (
            low <= bb_dn and
            k_prev < d_prev and k > d and
            k < 20 and d < 20 and
            price > trend
        )

        # Short entry: price touches upper BB, stoch cross down in overbought, trend filter
        short_cond = (
            high >= bb_up and
            k_prev > d_prev and k < d and
            k > 80 and d > 80 and
            price < trend
        )

        if long_cond:
            stop = price - self.atr_mult * atr
            risk_per_unit = price - stop
            if risk_per_unit <= 0:
                return
            risk_amount = self.equity * self.risk_pct
            size = max(1, int(round(risk_amount / risk_per_unit)))
            self.stop_price = stop
            self.tp_price = bb_mid
            self.entry_bar = len(self.data)
            print(f"🌙🚀 LONG ENTRY @ {price:.2f} | SL={stop:.2f} | TP=BB_Mid={bb_mid:.2f} | Size={size} | K={k:.1f} D={d:.1f}")
            self.buy(size=size)

        elif short_cond:
            stop = price + self.atr_mult * atr
            risk_per_unit = stop - price
            if risk_per_unit <= 0:
                return
            risk_amount = self.equity * self.risk_pct
            size = max(1, int(round(risk_amount / risk_per_unit)))
            self.stop_price = stop
            self.tp_price = bb_mid
            self.entry_bar = len(self.data)
            print(f"🌙🔻 SHORT ENTRY @ {price:.2f} | SL={stop:.2f} | TP=BB_Mid={bb_mid:.2f} | Size={size} | K={k:.1f} D={d:.1f}")
            self.sell(size=size)


bt = Backtest(data, VolatilityOscillator, cash=1_000_000, commission=0.001, exclusive_orders=True)
stats = bt.run()
print(stats)
print(stats._strategy)