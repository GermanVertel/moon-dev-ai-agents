import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# 🌙 Moon Dev Backtest Engine - ChandeStochasticPullback 🚀

DATA_PATH = '/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv'


def load_data(path):
    print("🌙 Loading Moon Dev data from:", path)
    data = pd.read_csv(path)
    data.columns = data.columns.str.strip().str.lower()
    data = data.drop(columns=[c for c in data.columns if 'unnamed' in c.lower()])
    # Map to backtesting.py required case
    data = data.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume',
        'datetime': 'Datetime',
    })
    if 'Datetime' in data.columns:
        data['Datetime'] = pd.to_datetime(data['Datetime'])
        data = data.set_index('Datetime')
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    print("✨ Data loaded. Rows:", len(data))
    return data


class ChandeStochasticPullback(Strategy):
    # Strategy parameters
    cmo_period = 14
    cmo_pullback_threshold = 20      # CMO dips below this = pullback condition
    stoch_k = 14
    stoch_d = 3
    stoch_smooth = 3
    stoch_oversold = 20
    stoch_overbought = 80
    atr_period = 14
    atr_stop_mult = 1.5
    atr_target_mult = 3.0
    vol_ma_period = 20
    ema_trend_period = 200
    risk_pct = 0.01                  # 1% risk per trade
    max_atr_mult = 2.0               # skip if ATR > 2x its 50-period avg
    atr_avg_period = 50
    time_stop_bars = 50              # optional time stop

    def init(self):
        print("🌙 Initializing ChandeStochasticPullback indicators...")
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        # CMO - trend/pullback filter
        self.cmo = self.I(talib.CMO, close, timeperiod=self.cmo_period, name='CMO')

        # Stochastic %K and %D
        stoch_k_full = self.I(
            talib.STOCH,
            high, low, close,
            fastk_period=self.stoch_k,
            slowk_period=self.stoch_smooth,
            slowk_matype=0,
            slowd_period=self.stoch_d,
            slowd_matype=0,
            name='STOCH_K',
        )
        stoch_d_full = self.I(
            talib.STOCH,
            high, low, close,
            fastk_period=self.stoch_k,
            slowk_period=self.stoch_smooth,
            slowk_matype=0,
            slowd_period=self.stoch_d,
            slowd_matype=0,
            name='STOCH_D',
        )
        # talib.STOCH returns a tuple (slowk, slowd); extract each as separate series
        self.stoch_k = self.I(lambda: stoch_k_full[0], name='STOCH_K_VAL')
        self.stoch_d = self.I(lambda: stoch_d_full[1], name='STOCH_D_VAL')

        # ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period, name='ATR')
        self.atr_avg = self.I(talib.SMA, self.atr, timeperiod=self.atr_avg_period, name='ATR_AVG')

        # Volume MA
        self.vol_ma = self.I(talib.SMA, volume, timeperiod=self.vol_ma_period, name='VOL_MA')

        # 200 EMA trend filter
        self.ema200 = self.I(talib.EMA, close, timeperiod=self.ema_trend_period, name='EMA200')

        print("✨ Indicators ready. Moon Dev systems online. 🚀")

    def next(self):
        price = self.data.Close[-1]

        # Skip if in position
        if self.position:
            self._manage_position()
            return

        # Need enough history
        if len(self.data) < max(self.ema_trend_period, self.atr_avg_period) + 5:
            return

        cmo_now = self.cmo[-1]
        cmo_prev = self.cmo[-2]
        k_now = self.stoch_k[-1]
        k_prev = self.stoch_k[-2]
        d_now = self.stoch_d[-1]
        d_prev = self.stoch_d[-2]
        atr_now = self.atr[-1]
        atr_avg_now = self.atr_avg[-1]
        vol_now = self.data.Volume[-1]
        vol_ma_now = self.vol_ma[-1]
        ema_now = self.ema200[-1]

        # --- Filters ---
        # 1. Trend filter: price above 200 EMA
        if not (price > ema_now):
            return

        # 2. Max ATR filter (avoid whipsaw regimes)
        if atr_avg_now > 0 and atr_now > self.max_atr_mult * atr_avg_now:
            print(f"🌙⚡ Skipping: ATR too high ({atr_now:.2f} vs avg {atr_avg_now:.2f})")
            return

        # 3. Pullback condition: CMO dipped below threshold recently
        cmo_pullback = (cmo_prev < self.cmo_pullback_threshold) or (cmo_now < self.cmo_pullback_threshold)

        # 4. Oversold trigger: stoch was oversold recently
        stoch_oversold_recent = (k_prev < self.stoch_oversold) or (k_now < self.stoch_oversold)

        # 5. Entry signal: stoch %K crosses back above %D from oversold
        # Replaced backtesting.lib.crossover with manual array-index comparison
        stoch_cross_up = (self.stoch_k[-2] < self.stoch_d[-2]) and (self.stoch_k[-1] > self.stoch_d[-1])
        stoch_above_oversold = k_now > self.stoch_oversold

        # 6. Volume confirmation: current volume > vol MA
        volume_confirm = vol_now > vol_ma_now

        if (cmo_pullback and stoch_oversold_recent and stoch_cross_up
                and stoch_above_oversold and volume_confirm):
            # Risk-based position sizing
            stop_price = price - (atr_now * self.atr_stop_mult)
            target_price = price + (atr_now * self.atr_target_mult)
            risk_per_unit = price - stop_price
            if risk_per_unit <= 0:
                return

            equity = self.equity
            risk_amount = equity * self.risk_pct
            position_size = risk_amount / risk_per_unit
            position_size = int(round(position_size))
            if position_size <= 0:
                position_size = 1

            print(f"🌙🚀 MOON DEV LONG SIGNAL! Price={price:.2f} | CMO={cmo_now:.2f} "
                  f"| %K={k_now:.2f} %D={d_now:.2f} | ATR={atr_now:.2f} | Size={position_size}")
            print(f"   🎯 Stop={stop_price:.2f} | Target={target_price:.2f}")

            self.buy(size=position_size, sl=stop_price, tp=target_price)
            self._entry_bar = len(self.data)
            self._entry_price = price

    def _manage_position(self):
        # Momentum exit: CMO crosses below 0 or stoch overbought and turning down
        cmo_now = self.cmo[-1]
        cmo_prev = self.cmo[-2]
        k_now = self.stoch_k[-1]
        k_prev = self.stoch_k[-2]

        # Momentum exit: CMO crosses below 0
        if cmo_prev >= 0 and cmo_now < 0:
            print(f"🌙⚠️ Momentum exit: CMO crossed below 0 ({cmo_now:.2f})")
            self.position.close()
            return

        # Stoch overbought and turning down
        if k_prev > self.stoch_overbought and k_now < k_prev:
            print(f"🌙⚠️ Momentum exit: Stoch overbought turning down ({k_now:.2f})")
            self.position.close()
            return

        # Time stop
        if hasattr(self, '_entry_bar') and (len(self.data) - self._entry_bar) >= self.time_stop_bars:
            print("🌙⏰ Time stop hit. Closing position.")
            self.position.close()
            return


if __name__ == '__main__':
    print("🌙✨ Moon Dev Backtest Engine Starting - ChandeStochasticPullback ✨🌙")
    data = load_data(DATA_PATH)
    bt = Backtest(
        data,
        ChandeStochasticPullback,
        cash=1_000_000,
        commission=0.002,
        exclusive_orders=True,
    )
    stats = bt.run()
    print(stats)
    print(stats._strategy)
    print("🚀🌙 Moon Dev Backtest Complete!")