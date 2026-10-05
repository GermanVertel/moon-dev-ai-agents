import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# 🌙 Moon Dev's KineticDrift Strategy ✨
DATA_PATH = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"

print("🌙✨ Moon Dev Backtest AI initializing KineticDrift... 🚀")

# Load & clean data
data = pd.read_csv(DATA_PATH)
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open', 'high': 'High', 'low': 'Low',
    'close': 'Close', 'volume': 'Volume'
})
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')
print(f"🌙 Data loaded: {len(data)} bars ✨")


class KineticDrift(Strategy):
    # Parameters
    sma_period = 50
    ema_period = 50
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    mrr_window = 20
    atr_period = 14
    atr_avg_period = 50
    atr_sl_mult = 1.0
    atr_tp_mult = 1.5
    max_hold_bars = 20
    risk_pct = 0.02
    vol_circuit_breaker = 2.5

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low

        # 🌙 Trend Layer
        self.sma = self.I(talib.SMA, close, timeperiod=self.sma_period)
        self.ema = self.I(talib.EMA, close, timeperiod=self.ema_period)

        # ✨ MACD Layer
        self.macd, self.macd_sig, self.macd_hist = self.I(
            talib.MACD, close,
            fastperiod=self.macd_fast,
            slowperiod=self.macd_slow,
            signalperiod=self.macd_signal
        )
        # MACD Delta = histogram
        self.macd_delta = self.macd_hist

        # Rolling avg & std of MACD Delta
        self.macd_delta_avg = self.I(
            lambda x: pd.Series(x).rolling(self.mrr_window).mean().values,
            self.macd_delta
        )
        self.macd_delta_std = self.I(
            lambda x: pd.Series(x).rolling(self.mrr_window).std().values,
            self.macd_delta
        )

        # MRR = (delta - avg) / std
        self.mrr = self.I(
            lambda d, a, s: np.where(s > 0, (d - a) / s, 0.0),
            self.macd_delta, self.macd_delta_avg, self.macd_delta_std
        )

        # 🚀 Volatility Layer
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)
        self.atr_avg = self.I(talib.SMA, self.atr, timeperiod=self.atr_avg_period)
        self.atr_ratio = self.I(
            lambda a, av: np.where(av > 0, a / av, 1.0),
            self.atr, self.atr_avg
        )

        self.trade_count = 0
        self.consec_losses = 0
        self.entry_bar = None
        self.entry_price = None
        self.stop_price = None
        self.tp_price = None

        print("🌙✨ KineticDrift indicators initialized 🚀")

    def next(self):
        if len(self.data) < max(self.sma_period, self.atr_avg_period, self.mrr_window) + 5:
            return

        price = self.data.Close[-1]
        ema = self.ema[-1]
        sma = self.sma[-1]
        mrr = self.mrr[-1]
        mrr_prev = self.mrr[-2]
        macd_delta = self.macd_delta[-1]
        macd_delta_avg = self.macd_delta_avg[-1]
        atr = self.atr[-1]
        atr_ratio = self.atr_ratio[-1]

        if np.isnan([ema, sma, mrr, mrr_prev, macd_delta, macd_delta_avg, atr, atr_ratio]).any():
            return

        # 🌙 Position management (exits)
        if self.position:
            bars_held = len(self.data) - self.entry_bar
            is_long = self.position.is_long

            # Primary exit: opposite crossover
            if is_long and ema < sma:
                print(f"🌙 Primary Exit LONG (EMA<SMA) @ {price:.2f} ✨")
                self.position.close()
                return
            if not is_long and ema > sma:
                print(f"🌙 Primary Exit SHORT (EMA>SMA) @ {price:.2f} ✨")
                self.position.close()
                return

            # Momentum exit: MRR crosses zero against position
            if is_long and mrr < 0:
                print(f"🌙 Momentum Exit LONG (MRR<0) @ {price:.2f} 🚀")
                self.position.close()
                return
            if not is_long and mrr > 0:
                print(f"🌙 Momentum Exit SHORT (MRR>0) @ {price:.2f} 🚀")
                self.position.close()
                return

            # Stop loss / Take profit
            if is_long:
                if self.data.Low[-1] <= self.stop_price:
                    print(f"🌙 STOP LOSS LONG @ {self.stop_price:.2f} 💥")
                    self.position.close()
                    return
                if self.data.High[-1] >= self.tp_price:
                    print(f"🌙 TAKE PROFIT LONG @ {self.tp_price:.2f} 🎯")
                    self.position.close()
                    return
            else:
                if self.data.High[-1] >= self.stop_price:
                    print(f"🌙 STOP LOSS SHORT @ {self.stop_price:.2f} 💥")
                    self.position.close()
                    return
                if self.data.Low[-1] <= self.tp_price:
                    print(f"🌙 TAKE PROFIT SHORT @ {self.tp_price:.2f} 🎯")
                    self.position.close()
                    return

            # Time-based exit
            if bars_held >= self.max_hold_bars:
                print(f"🌙 Time Exit ({bars_held} bars) @ {price:.2f} ⏰")
                self.position.close()
                return
            return

        # 🚀 Volatility circuit breaker
        if atr_ratio > self.vol_circuit_breaker:
            print(f"🌙 Circuit breaker: ATR ratio {atr_ratio:.2f} > {self.vol_circuit_breaker} — skipping ✨")
            return

        # 🌙 Entry signals
        long_signal = (
            ema > sma and
            mrr < 0 and mrr > mrr_prev and
            macd_delta < macd_delta_avg
        )
        short_signal = (
            ema < sma and
            mrr > 0 and mrr < mrr_prev and
            macd_delta > macd_delta_avg
        )

        # Drawdown control: halve size after 2 consecutive losses
        size_factor = 0.5 if self.consec_losses >= 2 else 1.0

        if long_signal:
            stop_dist = self.atr_sl_mult * atr * max(atr_ratio, 0.5)
            risk_amount = self.equity * self.risk_pct * size_factor
            raw_size = risk_amount / stop_dist if stop_dist > 0 else 0
            size = int(round(raw_size))
            if size < 1:
                return
            self.buy(size=size)
            self.entry_bar = len(self.data)
            self.entry_price = price
            self.stop_price = price - stop_dist
            self.tp_price = price + self.atr_tp_mult * atr
            self.trade_count += 1
            print(f"🌙🚀 LONG ENTRY #{self.trade_count} @ {price:.2f} | size={size} | SL={self.stop_price:.2f} TP={self.tp_price:.2f} | MRR={mrr:.3f} ATRr={atr_ratio:.2f} ✨")

        elif short_signal:
            stop_dist = self.atr_sl_mult * atr * max(atr_ratio, 0.5)
            risk_amount = self.equity * self.risk_pct * size_factor
            raw_size = risk_amount / stop_dist if stop_dist > 0 else 0
            size = int(round(raw_size))
            if size < 1:
                return
            self.sell(size=size)
            self.entry_bar = len(self.data)
            self.entry_price = price
            self.stop_price = price + stop_dist
            self.tp_price = price - self.atr_tp_mult * atr
            self.trade_count += 1
            print(f"🌙🚀 SHORT ENTRY #{self.trade_count} @ {price:.2f} | size={size} | SL={self.stop_price:.2f} TP={self.tp_price:.2f} | MRR={mrr:.3f} ATRr={atr_ratio:.2f} ✨")


bt = Backtest(
    data,
    KineticDrift,
    cash=1_000_000,
    commission=0.002,
    exclusive_orders=True
)

stats = bt.run()
print(stats)
print(stats._strategy)