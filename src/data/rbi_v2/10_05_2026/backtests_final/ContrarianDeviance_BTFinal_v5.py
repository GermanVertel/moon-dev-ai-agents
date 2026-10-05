import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# ============================================================
# 🌙 Moon Dev's ContrarianDeviance Backtest 🚀
# ============================================================

data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"

print("🌙 Loading Moon Dev data from:", data_path)
data = pd.read_csv(data_path)

# Clean column names
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

print("✨ Data loaded:", len(data), "rows")
print("🚀 Columns:", list(data.columns))


class ContrarianDeviance(Strategy):
    # Strategy parameters
    rsi_period = 14
    rsi_low = 35
    rsi_overbought = 70
    atr_period = 14
    vol_lookback = 20
    ma_period = 50
    stop_loss_pct = 0.08
    take_profit_pct = 0.20
    max_hold_bars = 500  # ~ 5 days on 15m bars (96 bars/day)

    def init(self):
        print("🌙 Initializing ContrarianDeviance indicators...")

        close = self.data.Close
        high = self.data.High
        low = self.data.Low

        # RSI
        self.rsi = self.I(talib.RSI, close, timeperiod=self.rsi_period, name="RSI")

        # ATR for volatility
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period, name="ATR")

        # Realized volatility (rolling std of returns)
        def realized_vol(c, period):
            s = pd.Series(c)
            r = s.pct_change()
            return r.rolling(period).std().values

        self.rvol = self.I(realized_vol, close, self.vol_lookback, name="RVOL")

        # Moving average for trend filter
        self.ma = self.I(talib.SMA, close, timeperiod=self.ma_period, name="MA")

        # Deviance Index: price trend vs volatility trend divergence
        def deviance_index(c, v, period):
            s_c = pd.Series(c)
            s_v = pd.Series(v)
            pc = s_c.pct_change(period)
            vc = s_v.pct_change(period)
            di = -np.sign(pc) * np.sign(vc) * (np.abs(pc) / (np.abs(vc) + 1e-9))
            di = di.replace([np.inf, -np.inf], 0).fillna(0)
            return di.values

        self.di = self.I(deviance_index, close, self.rvol, 10, name="DI")

        # Expanse period: ATR above its rolling mean
        def atr_mean(a, period):
            return pd.Series(a).rolling(period).mean().values

        self.atr_mean = self.I(atr_mean, self.atr, self.vol_lookback, name="ATR_MEAN")

        # RSI probability model: historical frequency of low RSI given current vol regime
        def rsi_prob_low(rsi_arr, atr_arr, atr_mean_arr, lookback=200):
            rsi_s = pd.Series(rsi_arr)
            atr_s = pd.Series(atr_arr)
            atr_m = pd.Series(atr_mean_arr)
            prob = np.full(len(rsi_s), np.nan)
            for i in range(lookback, len(rsi_s)):
                if np.isnan(atr_s.iloc[i]) or np.isnan(atr_m.iloc[i]):
                    continue
                if atr_s.iloc[i] > atr_m.iloc[i]:
                    window = rsi_s.iloc[i - lookback:i]
                    if len(window.dropna()) > 0:
                        prob[i] = (window < 35).mean()
            return prob

        self.rsi_prob = self.I(rsi_prob_low, self.rsi, self.atr, self.atr_mean,
                               name="RSI_PROB")

        self.entry_bar = None
        self.entry_price = None
        print("✨ Indicators ready! 🌙")

    def next(self):
        price = self.data.Close[-1]

        if len(self.data) < self.ma_period + 5:
            return

        rsi = self.rsi[-1]
        di = self.di[-1]
        atr = self.atr[-1]
        atr_mean = self.atr_mean[-1]
        rsi_prob = self.rsi_prob[-1]
        ma = self.ma[-1]

        # ---- ENTRY LOGIC ----
        if not self.position:
            if (np.isnan(di) or np.isnan(rsi) or np.isnan(atr)
                    or np.isnan(atr_mean) or np.isnan(rsi_prob)):
                return

            # 1. Contrarian DI alignment
            di_signal = di > 0.5

            # 2. Expanse period: ATR > its mean
            expanse = atr > atr_mean

            # 3. RSI probability model
            prob_signal = rsi_prob > 0.4

            # 4. RSI confirms low zone
            rsi_low_confirm = rsi < self.rsi_low

            # 5. Trend filter: price above MA
            trend_ok = price > ma

            if di_signal and expanse and prob_signal and rsi_low_confirm and trend_ok:
                print(f"🌙🚀 CONTRARIAN LONG SIGNAL! price={price:.2f} "
                      f"RSI={rsi:.2f} DI={di:.2f} ATR={atr:.2f} "
                      f"ATRmean={atr_mean:.2f} P={rsi_prob:.2f}")

                # ✅ FIX: use fractional sizing (percentage of equity)
                size = 0.95
                self.buy(size=size)
                self.entry_bar = len(self.data)
                self.entry_price = price

        # ---- EXIT LOGIC ----
        else:
            bars_held = len(self.data) - self.entry_bar
            # ✅ FIX: use actual trade entry price instead of stored value
            try:
                entry_price = self.trades[-1].entry_price
            except Exception:
                entry_price = self.entry_price

            pnl_pct = (price - entry_price) / entry_price

            di_neutral = abs(di) < 0.2 if not np.isnan(di) else False
            rsi_exit = rsi > self.rsi_overbought if not np.isnan(rsi) else False
            time_exit = bars_held >= self.max_hold_bars
            tp_hit = pnl_pct >= self.take_profit_pct
            sl_hit = pnl_pct <= -self.stop_loss_pct

            if di_neutral or rsi_exit or time_exit or tp_hit or sl_hit:
                reason = ("DI_NEUTRAL" if di_neutral else
                          "RSI_OB" if rsi_exit else
                          "TIME" if time_exit else
                          "TP" if tp_hit else "SL")
                print(f"🌙💫 EXIT ({reason}) price={price:.2f} "
                      f"PnL={pnl_pct*100:.2f}% bars={bars_held}")
                self.position.close()
                self.entry_bar = None
                self.entry_price = None


print("🌙✨ Starting Moon Dev Backtest...")
bt = Backtest(data, ContrarianDeviance, cash=1_000_000, commission=0.001)

stats = bt.run()
print(stats)
print(stats._strategy)