import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Load and clean data
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
print("🌙✨ Moon Dev data loaded:", data.shape)
print(data.head())


class KineticVolatility(Strategy):
    atr_period = 14
    atr_sma_period = 50
    donchian_period = 20
    ema_period = 200
    compression_lookback = 10
    compression_threshold = 0.0
    expansion_threshold = 0.5
    atr_stop_mult = 1.5
    atr_tp_mult = 3.0
    time_stop_bars = 15
    risk_pct = 0.01
    size = 1_000_000

    def init(self):
        print("🌙🚀 Initializing KineticVolatility indicators...")
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period)
        self.atr_sma = self.I(talib.SMA, self.atr, timeperiod=self.atr_sma_period)
        # volatility oscillator = (ATR / SMA(ATR,50)) - 1
        self.vol_osc = self.I(
            lambda a, s: (a / np.where(s == 0, np.nan, s)) - 1.0,
            self.atr, self.atr_sma
        )
        self.donchian_high = self.I(talib.MAX, self.data.High, timeperiod=self.donchian_period)
        self.donchian_low = self.I(talib.MIN, self.data.Low, timeperiod=self.donchian_period)
        self.ema = self.I(talib.EMA, self.data.Close, timeperiod=self.ema_period)
        print("🌙✨ Indicators ready!")

    def next(self):
        if len(self.data) < max(self.atr_sma_period, self.ema_period, self.donchian_period) + 2:
            return

        price = self.data.Close[-1]
        osc = self.vol_osc[-1]
        osc_prev = self.vol_osc[-2]
        atr = self.atr[-1]

        # compression check within last K bars
        compression = False
        for k in range(1, self.compression_lookback + 1):
            if len(self.vol_osc) > k and self.vol_osc[-k] is not None and not np.isnan(self.vol_osc[-k]):
                if self.vol_osc[-k] < self.compression_threshold:
                    compression = True
                    break

        expansion_trigger = (osc_prev is not None and osc is not None
                             and not np.isnan(osc_prev) and not np.isnan(osc)
                             and osc_prev < self.expansion_threshold
                             and osc >= self.expansion_threshold)

        # Manage open position
        if self.position:
            bars_held = len(self.data) - self.position.entry_bar
            if self.position.is_long:
                # time stop
                if bars_held >= self.time_stop_bars and self.data.Close[-1] < self.position.entry_price + atr * self.atr_stop_mult:
                    print(f"🌙⏰ Long TIME STOP at {price:.2f}")
                    self.position.close()
                    return
                # volatility contraction exit
                if osc < 0.0:
                    print(f"🌙💤 Long VOL CONTRACTION exit at {price:.2f}")
                    self.position.close()
                    return
            else:
                if bars_held >= self.time_stop_bars and self.data.Close[-1] > self.position.entry_price - atr * self.atr_stop_mult:
                    print(f"🌙⏰ Short TIME STOP at {price:.2f}")
                    self.position.close()
                    return
                if osc < 0.0:
                    print(f"🌙💤 Short VOL CONTRACTION exit at {price:.2f}")
                    self.position.close()
                    return
            return

        if not compression or not expansion_trigger:
            return

        # Position sizing based on risk
        risk_amount = self.equity * self.risk_pct
        stop_dist = atr * self.atr_stop_mult
        if stop_dist <= 0 or np.isnan(stop_dist):
            return
        position_size = int(round(risk_amount / stop_dist))
        if position_size <= 0:
            return

        trend_up = price > self.ema[-1]
        trend_down = price < self.ema[-1]

        # Long entry
        if price > self.donchian_high[-2] and trend_up:
            sl = price - stop_dist
            tp = price + atr * self.atr_tp_mult
            print(f"🌙🚀 LONG breakout! price={price:.2f} osc={osc:.3f} SL={sl:.2f} TP={tp:.2f} size={position_size}")
            self.buy(size=position_size, sl=sl, tp=tp)
        # Short entry
        elif price < self.donchian_low[-2] and trend_down:
            sl = price + stop_dist
            tp = price - atr * self.atr_tp_mult
            print(f"🌙🔻 SHORT breakout! price={price:.2f} osc={osc:.3f} SL={sl:.2f} TP={tp:.2f} size={position_size}")
            self.sell(size=position_size, sl=sl, tp=tp)


bt = Backtest(data, KineticVolatility, cash=1_000_000, commission=0.0002, exclusive_orders=True)
stats = bt.run()
print(stats)
print(stats._strategy)