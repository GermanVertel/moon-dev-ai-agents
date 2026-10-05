import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# ============================
# 🌙 Moon Dev CascadeLiquidity 🌙
# ============================

class CascadeLiquidity(Strategy):
    # Strategy parameters
    sr_lookback = 100          # lookback for swing high/low S/R mapping
    atr_period = 14
    vol_ma_period = 20
    vol_spike_mult = 2.5       # volume spike threshold
    body_atr_mult = 1.2        # large candle body threshold
    wick_atr_mult = 0.8        # wick rejection threshold
    zone_width_atr = 1.0       # ATR-based zone width
    risk_pct = 0.0075          # 0.75% risk per trade
    time_stop_bars = 10
    tp1_atr_mult = 1.0
    sl_buffer_atr = 0.5

    def init(self):
        print("🌙✨ Moon Dev CascadeLiquidity initializing... ✨🌙")

        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period)
        self.vol_ma = self.I(talib.SMA, self.data.Volume, timeperiod=self.vol_ma_period)
        self.sr_high = self.I(talib.MAX, self.data.High, timeperiod=self.sr_lookback)
        self.sr_low = self.I(talib.MIN, self.data.Low, timeperiod=self.sr_lookback)

        self.entry_bar = 0
        self.tp1_hit = False
        self.trade_dir = 0
        self.entry_price = 0
        self.stop_price = 0
        self.tp1_price = 0
        self.tp2_price = 0

        print("🌙✨ Indicators ready. Let's hunt cascades! 🚀")

    def next(self):
        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        open_ = self.data.Open[-1]
        vol = self.data.Volume[-1]

        atr = self.atr[-1]
        vol_ma = self.vol_ma[-1]

        if np.isnan(atr) or np.isnan(vol_ma) or vol_ma == 0:
            return

        # Manage open trade
        if self.position:
            self._manage_trade(price, high, low)
            return

        # Detect S/R zones
        sup_zone = self.sr_low[-1]
        res_zone = self.sr_high[-1]

        if np.isnan(sup_zone) or np.isnan(res_zone):
            return

        # Volume spike
        vol_spike = vol > (self.vol_spike_mult * vol_ma)
        if not vol_spike:
            return

        # Candle body & wick
        body = abs(price - open_)
        upper_wick = high - max(price, open_)
        lower_wick = min(price, open_) - low

        big_body = body > (self.body_atr_mult * atr)

        # Zone band
        zone_band = self.zone_width_atr * atr

        # ============ LONG SETUP ============
        # Sweep below support then close back inside zone
        swept_support = low < (sup_zone - 0.1 * atr)
        closed_back_in = price > sup_zone
        bullish_reject = (lower_wick > self.wick_atr_mult * atr) or (price > open_ and body > 0.5 * atr)

        if swept_support and closed_back_in and bullish_reject and big_body:
            sl = low - (self.sl_buffer_atr * atr)
            risk = price - sl
            if risk <= 0:
                return
            risk_amount = self.equity * self.risk_pct
            size = int(round(risk_amount / risk))
            if size <= 0:
                return

            tp1 = price + (self.tp1_atr_mult * atr)
            tp2 = res_zone if res_zone > price else price + 2 * atr

            print(f"🌙🚀 LONG CASCADE FADE! Price={price:.2f} SL={sl:.2f} TP1={tp1:.2f} TP2={tp2:.2f} Size={size}")
            self.buy(size=size, sl=sl, tp=tp2)
            self.entry_bar = len(self.data)
            self.trade_dir = 1
            self.entry_price = price
            self.stop_price = sl
            self.tp1_price = tp1
            self.tp2_price = tp2
            self.tp1_hit = False
            return

        # ============ SHORT SETUP ============
        swept_resistance = high > (res_zone + 0.1 * atr)
        closed_back_in_s = price < res_zone
        bearish_reject = (upper_wick > self.wick_atr_mult * atr) or (price < open_ and body > 0.5 * atr)

        if swept_resistance and closed_back_in_s and bearish_reject and big_body:
            sl = high + (self.sl_buffer_atr * atr)
            risk = sl - price
            if risk <= 0:
                return
            risk_amount = self.equity * self.risk_pct
            size = int(round(risk_amount / risk))
            if size <= 0:
                return

            tp1 = price - (self.tp1_atr_mult * atr)
            tp2 = sup_zone if sup_zone < price else price - 2 * atr

            print(f"🌙🚀 SHORT CASCADE FADE! Price={price:.2f} SL={sl:.2f} TP1={tp1:.2f} TP2={tp2:.2f} Size={size}")
            self.sell(size=size, sl=sl, tp=tp2)
            self.entry_bar = len(self.data)
            self.trade_dir = -1
            self.entry_price = price
            self.stop_price = sl
            self.tp1_price = tp1
            self.tp2_price = tp2
            self.tp1_hit = False
            return

    def _manage_trade(self, price, high, low):
        bars_held = len(self.data) - self.entry_bar

        # Time stop
        if bars_held >= self.time_stop_bars and not self.tp1_hit:
            print(f"⏰🌙 Time stop hit after {bars_held} bars. Exiting.")
            self.position.close()
            return

        if self.trade_dir == 1:
            # Stop loss
            if low <= self.stop_price:
                print(f"🛑 Long SL hit at {self.stop_price:.2f}")
                self.position.close()
                return
            # TP1
            if not self.tp1_hit and high >= self.tp1_price:
                print(f"🎯 Long TP1 hit at {self.tp1_price:.2f}")
                self.tp1_hit = True
                # Move stop to breakeven
                self.stop_price = max(self.stop_price, self.entry_price)
            # TP2
            if self.tp1_hit and high >= self.tp2_price:
                print(f"🎯🎯 Long TP2 hit at {self.tp2_price:.2f}")
                self.position.close()
                return

        elif self.trade_dir == -1:
            if high >= self.stop_price:
                print(f"🛑 Short SL hit at {self.stop_price:.2f}")
                self.position.close()
                return
            if not self.tp1_hit and low <= self.tp1_price:
                print(f"🎯 Short TP1 hit at {self.tp1_price:.2f}")
                self.tp1_hit = True
                self.stop_price = min(self.stop_price, self.entry_price)
            if self.tp1_hit and low <= self.tp2_price:
                print(f"🎯🎯 Short TP2 hit at {self.tp2_price:.2f}")
                self.position.close()
                return


# ============================
# 🌙 Data Loading
# ============================
data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
print("🌙 Loading data from:", data_path)
data = pd.read_csv(data_path)

data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Map columns to proper case
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
print(f"🌙 Data loaded: {len(data)} bars ✨")

# ============================
# 🚀 Run Backtest
# ============================
bt = Backtest(data, CascadeLiquidity, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)