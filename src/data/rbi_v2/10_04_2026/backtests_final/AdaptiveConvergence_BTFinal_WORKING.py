import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

print("🌙 Moon Dev's AdaptiveConvergence Backtest Loading... ✨🚀")

data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

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

print(f"🌙 Data loaded: {len(data)} bars from {data.index[0]} to {data.index[-1]} ✨")


class AdaptiveConvergence(Strategy):
    sma_period = 50
    ema_period = 20
    atr_period = 14
    trend_period = 200
    spread_threshold_pct = 0.005
    atr_target_mult = 0.03
    atr_stop_mult = 0.01
    leverage = 2.0
    risk_pct = 0.01

    def init(self):
        print("🌙 Initializing AdaptiveConvergence indicators... ✨")
        self.sma50 = self.I(talib.SMA, self.data.Close, timeperiod=self.sma_period)
        self.ema20 = self.I(talib.EMA, self.data.Close, timeperiod=self.ema_period)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period)
        self.sma200 = self.I(talib.SMA, self.data.Close, timeperiod=self.trend_period)
        print("🚀 Indicators ready: SMA50, EMA20, ATR14, SMA200 🌙")

    def next(self):
        price = self.data.Close[-1]
        sma50 = self.sma50[-1]
        ema20 = self.ema20[-1]
        atr = self.atr[-1]
        sma200 = self.sma200[-1]

        if np.isnan(sma50) or np.isnan(ema20) or np.isnan(atr) or np.isnan(sma200):
            return

        spread = sma50 - ema20
        spread_pct = spread / sma50 if sma50 > 0 else 0

        if self.position:
            for trade in self.trades:
                if trade.is_long:
                    if price >= trade.tp:
                        print(f"🌙✨ TP HIT at {price:.2f} | spread_pct={spread_pct:.4f} 🚀")
                        trade.close()
                    elif price <= trade.sl:
                        print(f"🌙💥 SL HIT at {price:.2f} | spread_pct={spread_pct:.4f}")
                        trade.close()
                    elif spread <= 0:
                        print(f"🌙🔄 Convergence exit at {price:.2f} | spread_pct={spread_pct:.4f}")
                        trade.close()
            return

        if spread_pct > self.spread_threshold_pct and price > sma200:
            stop_price = price - self.atr_stop_mult * atr
            target_price = price + self.atr_target_mult * atr

            if stop_price >= price or target_price <= price:
                return

            risk_per_unit = price - stop_price
            if risk_per_unit <= 0:
                return

            equity = self.equity
            risk_amount = equity * self.risk_pct
            position_size = risk_amount / risk_per_unit
            position_size = position_size * self.leverage * 0.5

            max_units_by_cash = equity / price
            position_size = min(position_size, max_units_by_cash)

            position_size = int(round(position_size))

            if position_size < 1:
                return

            print(f"🌙🚀 ENTRY LONG | price={price:.2f} | spread_pct={spread_pct:.4f} | "
                  f"SL={stop_price:.2f} | TP={target_price:.2f} | size={position_size} ✨")

            self.buy(size=position_size, sl=stop_price, tp=target_price)


bt = Backtest(data, AdaptiveConvergence, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)