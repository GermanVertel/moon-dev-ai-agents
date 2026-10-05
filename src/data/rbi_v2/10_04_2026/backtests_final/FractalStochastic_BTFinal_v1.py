import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

DATA_PATH = '/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv'

print("🌙 Moon Dev's FractalStochastic Backtest loading... ✨🚀")

data = pd.read_csv(DATA_PATH)
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
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
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
print(f"🌙 Data loaded: {len(data)} bars ✨")


class FractalStochastic(Strategy):
    stoch_period = 14
    stoch_k = 3
    stoch_d = 3
    atr_period = 14
    ema_period = 50
    risk_pct = 0.02
    atr_mult_stop = 1.0
    min_channel_width_atr = 0.5
    max_channel_width_atr = 12.0

    def init(self):
        high = self.data.High
        low = self.data.Low
        close = self.data.Close

        # Stochastic
        self.k, self.d = self.I(talib.STOCH, high, low, close,
                                fastk_period=self.stoch_period,
                                slowk_period=self.stoch_k,
                                slowk_matype=0,
                                slowd_period=self.stoch_d,
                                slowd_matype=0)

        # ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        # EMA trend filter
        self.ema = self.I(talib.EMA, close, timeperiod=self.ema_period)

        # Fractal detection (5-bar Bill Williams)
        h = np.array(high)
        l = np.array(low)
        n = len(h)
        up_fractal = np.full(n, np.nan)
        dn_fractal = np.full(n, np.nan)

        for i in range(2, n - 2):
            if (h[i] > h[i-1] and h[i] > h[i-2] and
                h[i] > h[i+1] and h[i] > h[i+2]):
                up_fractal[i] = h[i]
            if (l[i] < l[i-1] and l[i] < l[i-2] and
                l[i] < l[i+1] and l[i] < l[i+2]):
                dn_fractal[i] = l[i]

        # Confirmed fractals (shifted 2 bars to avoid repaint)
        up_conf = np.full(n, np.nan)
        dn_conf = np.full(n, np.nan)
        for i in range(2, n):
            up_conf[i] = up_fractal[i-2]
            dn_conf[i] = dn_fractal[i-2]

        self.up_fractal = self.I(lambda: pd.Series(up_conf).ffill().values)
        self.dn_fractal = self.I(lambda: pd.Series(dn_conf).ffill().values)

        # Channel width guard
        self.channel_width = self.I(lambda: np.abs(self.up_fractal - self.dn_fractal))

        # Track our own stop-loss state since Position has no .sl attribute
        self.current_sl = None

        print("🌙 FractalStochastic indicators initialized ✨")

    def next(self):
        price = self.data.Close[-1]
        atr_val = self.atr[-1]
        if np.isnan(atr_val) or atr_val <= 0:
            return

        up_f = self.up_fractal[-1]
        dn_f = self.dn_fractal[-1]
        if np.isnan(up_f) or np.isnan(dn_f):
            return

        channel_w = self.channel_width[-1]
        if np.isnan(channel_w):
            return

        # Volatility guard
        if channel_w < self.min_channel_width_atr * atr_val:
            return
        if channel_w > self.max_channel_width_atr * atr_val:
            return

        k = self.k[-1]
        d = self.d[-1]
        k_prev = self.k[-2]
        d_prev = self.d[-2]

        if np.isnan(k) or np.isnan(d) or np.isnan(k_prev) or np.isnan(d_prev):
            return

        # Crossover detection (replaced backtesting.lib.crossover)
        bull_cross = k_prev < d_prev and k > d
        bear_cross = k_prev > d_prev and k < d
        k_rising = k > k_prev
        k_falling = k < k_prev
        ema_val = self.ema[-1]

        # ============ LONG ============
        if not self.position:
            self.current_sl = None
            if price > up_f and bull_cross and k_rising and k < 80 and price > ema_val:
                stop = dn_f - self.atr_mult_stop * atr_val
                risk = price - stop
                if risk > 0:
                    size = int(round((self.equity * self.risk_pct) / risk))
                    if size > 0:
                        print(f"🚀🌙 LONG BREAKOUT | price={price:.2f} up_fractal={up_f:.2f} K={k:.1f} D={d:.1f} size={size}")
                        self.buy(size=size, sl=stop)
                        self.current_sl = stop

            # ============ SHORT ============
            elif price < dn_f and bear_cross and k_falling and k > 20 and price < ema_val:
                stop = up_f + self.atr_mult_stop * atr_val
                risk = stop - price
                if risk > 0:
                    size = int(round((self.equity * self.risk_pct) / risk))
                    if size > 0:
                        print(f"🔻🌙 SHORT BREAKDOWN | price={price:.2f} dn_fractal={dn_f:.2f} K={k:.1f} D={d:.1f} size={size}")
                        self.sell(size=size, sl=stop)
                        self.current_sl = stop

        else:
            # Manage open position
            if self.position.is_long:
                # Trail stop to latest down fractal
                new_sl = dn_f - self.atr_mult_stop * atr_val
                if self.current_sl is None or new_sl > self.current_sl:
                    self.current_sl = new_sl

                # Momentum exit
                if bear_cross and price < up_f:
                    print(f"⚡🌙 LONG momentum exit | K={k:.1f} D={d:.1f}")
                    self.position.close()
                    self.current_sl = None

                # Channel reversal exit
                elif price < dn_f:
                    print(f"⚠️🌙 LONG channel reversal exit | price={price:.2f} dn_f={dn_f:.2f}")
                    self.position.close()
                    self.current_sl = None

            elif self.position.is_short:
                new_sl = up_f + self.atr_mult_stop * atr_val
                if self.current_sl is None or new_sl < self.current_sl:
                    self.current_sl = new_sl

                if bull_cross and price > dn_f:
                    print(f"⚡🌙 SHORT momentum exit | K={k:.1f} D={d:.1f}")
                    self.position.close()
                    self.current_sl = None

                elif price > up_f:
                    print(f"⚠️🌙 SHORT channel reversal exit | price={price:.2f} up_f={up_f:.2f}")
                    self.position.close()
                    self.current_sl = None


bt = Backtest(data, FractalStochastic, cash=1_000_000, commission=0.0002)
stats = bt.run()
print(stats)
print(stats._strategy)