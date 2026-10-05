import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

print("🌙✨ Moon Dev VolatilityArbitrage Backtest Starting... 🚀")

DATA_PATH = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"

data = pd.read_csv(DATA_PATH)
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
print(f"🌙 Data loaded: {len(data)} rows ✨")


class VolatilityArbitrage(Strategy):
    rv_period = 20
    vol_ma_period = 20
    atr_period = 14
    risk_pct = 0.01
    spread_entry = 0.05
    spread_exit = 0.01
    stop_extra = 0.03
    time_stop_bars = 10 * 96
    rel_vol_entry = 1.5
    rel_vol_exit = 0.5

    def init(self):
        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        volume = pd.Series(self.data.Volume)

        log_ret = np.log(close / close.shift(1))
        self.rv = self.I(
            lambda s: s.rolling(self.rv_period).std() * np.sqrt(252 * 96),
            log_ret,
            name="RV"
        )

        self.vol_ma = self.I(talib.SMA, volume.values, timeperiod=self.vol_ma_period, name="VolMA")

        self.rel_vol = self.I(
            lambda v, ma: v / ma,
            volume.values,
            self.vol_ma,
            name="RelVol"
        )

        self.atr = self.I(talib.ATR, high.values, low.values, close.values,
                          timeperiod=self.atr_period, name="ATR")

        self.iv_proxy = self.I(
            lambda rv: rv * 1.15,
            self.rv,
            name="IV_proxy"
        )

        self.spread = self.I(
            lambda iv, rv: iv - rv,
            self.iv_proxy,
            self.rv,
            name="IV-RV_Spread"
        )

        self.entry_bar = None
        self.entry_spread = None
        self.trade_dir = None

        print("🌙✨ Indicators initialized! 🚀")

    def next(self):
        price = self.data.Close[-1]
        spread = self.spread[-1]
        rel_vol = self.rel_vol[-1]
        atr = self.atr[-1]

        if np.isnan(spread) or np.isnan(rel_vol) or np.isnan(atr) or atr <= 0:
            return

        if self.position:
            bars_held = len(self.data) - self.entry_bar if self.entry_bar else 0

            if self.trade_dir == 'short_vol':
                if spread <= self.spread_exit or spread >= (self.entry_spread + self.stop_extra):
                    print(f"🌙 EXIT short_vol @ {price:.2f} | spread={spread:.4f} ✨")
                    self.position.close()
                    self.entry_bar = None
                    self.trade_dir = None
                    return
            elif self.trade_dir == 'long_vol':
                if spread >= -self.spread_exit or spread <= (self.entry_spread - self.stop_extra):
                    print(f"🌙 EXIT long_vol @ {price:.2f} | spread={spread:.4f} ✨")
                    self.position.close()
                    self.entry_bar = None
                    self.trade_dir = None
                    return

            if bars_held >= self.time_stop_bars:
                print(f"⏰ TIME STOP @ {price:.2f} 🌙")
                self.position.close()
                self.entry_bar = None
                self.trade_dir = None
                return

            if rel_vol < self.rel_vol_exit:
                print(f"💧 VOLUME EXIT @ {price:.2f} | rel_vol={rel_vol:.2f} 🌙")
                self.position.close()
                self.entry_bar = None
                self.trade_dir = None
                return
            return

        risk_amount = self.equity * self.risk_pct
        stop_dist = atr * 2
        if stop_dist <= 0:
            return
        size = int(round(risk_amount / stop_dist))
        if size < 1:
            size = 1
        if size > 1000000:
            size = 1000000

        if spread < -self.spread_entry and rel_vol > self.rel_vol_entry:
            print(f"🚀 LONG VOL ARB @ {price:.2f} | spread={spread:.4f} rel_vol={rel_vol:.2f} size={size} 🌙")
            self.buy(size=size)
            self.entry_bar = len(self.data)
            self.entry_spread = spread
            self.trade_dir = 'long_vol'

        elif spread > self.spread_entry and rel_vol > self.rel_vol_entry:
            print(f"🚀 SHORT VOL ARB @ {price:.2f} | spread={spread:.4f} rel_vol={rel_vol:.2f} size={size} 🌙")
            self.sell(size=size)
            self.entry_bar = len(self.data)
            self.entry_spread = spread
            self.trade_dir = 'short_vol'


bt = Backtest(
    data,
    VolatilityArbitrage,
    cash=1_000_000,
    commission=0.001,
    exclusive_orders=True
)

stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙✨ Moon Dev Backtest Complete! 🚀")