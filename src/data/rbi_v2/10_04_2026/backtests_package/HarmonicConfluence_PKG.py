import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Load and clean data
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'datetime': 'Date',
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})
data['Date'] = pd.to_datetime(data['Date'])
data = data.set_index('Date')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

print("🌙✨ Moon Dev HarmonicConfluence Backtest Initializing... 🚀")
print(f"📊 Data loaded: {len(data)} bars from {data.index[0]} to {data.index[-1]}")


class HarmonicConfluence(Strategy):
    ema_period = 200
    rsi_period = 14
    stoch_k = 14
    stoch_d = 3
    stoch_smooth = 3
    bb_period = 20
    bb_std = 2.0
    atr_period = 14
    atr_mult = 2.0
    risk_pct = 0.02
    size = 1_000_000

    def init(self):
        print("🌙 Initializing indicators...")
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        self.ema200 = self.I(talib.EMA, close, timeperiod=self.ema_period, name='EMA200')
        self.rsi = self.I(talib.RSI, close, timeperiod=self.rsi_period, name='RSI')

        # STOCH returns a tuple; wrap properly
        def _stoch(high, low, close):
            k, d = talib.STOCH(
                high, low, close,
                fastk_period=self.stoch_k,
                slowk_period=self.stoch_smooth,
                slowk_matype=0,
                slowd_period=self.stoch_d,
                slowd_matype=0,
            )
            return k, d

        stoch_k, stoch_d = self.I(_stoch, high, low, close, name='STOCH')
        self.stoch_k = stoch_k
        self.stoch_d = stoch_d

        def _bbands(close):
            u, m, l = talib.BBANDS(
                close,
                timeperiod=self.bb_period,
                nbdevup=self.bb_std,
                nbdevdn=self.bb_std,
                matype=0,
            )
            return u, m, l

        bb_u, bb_m, bb_l = self.I(_bbands, close, name='BB')
        self.bb_upper = bb_u
        self.bb_mid = bb_m
        self.bb_lower = bb_l

        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period, name='ATR')
        self.obv = self.I(talib.OBV, close, volume, name='OBV')
        self.obv_ema = self.I(talib.EMA, self.obv, timeperiod=20, name='OBV_EMA')

        # Swing lows/highs for stop placement
        self.swing_low = self.I(talib.MIN, low, timeperiod=10, name='SwingLow')
        self.swing_high = self.I(talib.MAX, high, timeperiod=10, name='SwingHigh')

        print("✅ Indicators ready! 🌙")
        self.trade_count = 0

    def next(self):
        price = self.data.Close[-1]
        ema = self.ema200[-1]
        rsi = self.rsi[-1]
        rsi_prev = self.rsi[-2]
        k = self.stoch_k[-1]
        d = self.stoch_d[-1]
        k_prev = self.stoch_k[-2]
        d_prev = self.stoch_d[-2]
        bb_u = self.bb_upper[-1]
        bb_l = self.bb_lower[-1]
        atr = self.atr[-1]
        obv_now = self.obv[-1]
        obv_prev = self.obv[-2]
        obv_ema = self.obv_ema[-1]

        if np.isnan(ema) or np.isnan(rsi) or np.isnan(k) or np.isnan(bb_u) or np.isnan(atr):
            return

        # ---------- EXIT LOGIC ----------
        if self.position:
            if self.position.is_long:
                # Trend filter exit
                if price < ema:
                    print(f"🌙💥 LONG EXIT - Price below 200 EMA @ {price:.2f}")
                    self.position.close()
                    return
                # Take profit: RSI 70+ AND upper band touch, or Stoch bearish cross overbought
                if (rsi >= 70 and price >= bb_u) or (k_prev > d_prev and k < d and k > 80):
                    print(f"🌙🎯 LONG TP HIT @ {price:.2f} (RSI={rsi:.1f})")
                    self.position.close()
                    return
                # ATR trailing stop
                trail = price - self.atr_mult * atr
                current_sl = self.position.sl if self.position.sl else -np.inf
                if trail > current_sl:
                    self.position.sl = trail
            elif self.position.is_short:
                if price > ema:
                    print(f"🌙💥 SHORT EXIT - Price above 200 EMA @ {price:.2f}")
                    self.position.close()
                    return
                if (rsi <= 30 and price <= bb_l) or (k_prev < d_prev and k > d and k < 20):
                    print(f"🌙🎯 SHORT TP HIT @ {price:.2f} (RSI={rsi:.1f})")
                    self.position.close()
                    return
                trail = price + self.atr_mult * atr
                current_sl = self.position.sl if self.position.sl else np.inf
                if trail < current_sl:
                    self.position.sl = trail
            return

        # ---------- ENTRY LOGIC ----------
        # LONG conditions
        long_trend = price > ema
        long_rsi = (rsi_prev < 30 and rsi > 30) or (40 <= rsi <= 60 and rsi > rsi_prev)
        long_stoch = k_prev < d_prev and k > d and k < 80
        long_bb = self.data.Low[-1] <= bb_l and price > bb_l
        long_obv = obv_now > obv_prev or obv_now > obv_ema

        long_score = sum([long_trend, long_rsi, long_stoch, long_bb, long_obv])
        long_core = sum([long_trend, long_rsi, long_stoch, long_bb])

        # SHORT conditions
        short_trend = price < ema
        short_rsi = (rsi_prev > 70 and rsi < 70) or (40 <= rsi <= 60 and rsi < rsi_prev)
        short_stoch = k_prev > d_prev and k < d and k > 20
        short_bb = self.data.High[-1] >= bb_u and price < bb_u
        short_obv = obv_now < obv_prev or obv_now < obv_ema

        short_score = sum([short_trend, short_rsi, short_stoch, short_bb, short_obv])
        short_core = sum([short_trend, short_rsi, short_stoch, short_bb])

        # Risk-based sizing: use fixed size as required
        size = self.size

        if long_core >= 3 and long_obv:
            sl = min(self.swing_low[-1], price - self.atr_mult * atr)
            tp = price + 2 * (price - sl)
            print(f"🌙🚀 LONG SIGNAL @ {price:.2f} | Score={long_score}/5 | SL={sl:.2f} TP={tp:.2f}")
            self.buy(size=size, sl=sl, tp=tp)
            self.trade_count += 1
        elif short_core >= 3 and short_obv:
            sl = max(self.swing_high[-1], price + self.atr_mult * atr)
            tp = price - 2 * (sl - price)
            print(f"🌙🔻 SHORT SIGNAL @ {price:.2f} | Score={short_score}/5 | SL={sl:.2f} TP={tp:.2f}")
            self.sell(size=size, sl=sl, tp=tp)
            self.trade_count += 1


print("🌙✨ Launching Moon Dev Backtest... 🚀")
bt = Backtest(data, HarmonicConfluence, cash=1_000_000, commission=0.001, exclusive_orders=True)
stats = bt.run()
print("🌙✨ Backtest Complete! ✨🌙")
print(stats)
print(stats._strategy)