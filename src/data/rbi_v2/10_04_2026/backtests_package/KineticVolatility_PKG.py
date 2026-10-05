import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# 🌙 Moon Dev's KineticVolatility Backtest 🚀

data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper case mapping
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]


class KineticVolatility(Strategy):
    # Parameters
    atr_period = 14
    bb_period = 20
    bb_std = 2.0
    ema_period = 20
    rsi_period = 14
    vol_ma_period = 20
    risk_pct = 0.0075  # 0.75% risk per trade
    atr_stop_mult = 1.5
    tp1_mult = 1.0
    tp2_mult = 2.0
    trail_mult = 1.5
    time_stop_bars = 20
    size_fraction = 1.0  # full equity fraction base

    def init(self):
        print("🌙✨ Initializing KineticVolatility indicators...")
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)
        self.ema = self.I(talib.EMA, close, timeperiod=self.ema_period)
        self.rsi = self.I(talib.RSI, close, timeperiod=self.rsi_period)

        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS, close, timeperiod=self.bb_period,
            nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0
        )
        self.vol_ma = self.I(talib.SMA, volume, timeperiod=self.vol_ma_period)

        # MACD
        self.macd, self.macd_signal, self.macd_hist = self.I(
            talib.MACD, close, fastperiod=12, slowperiod=26, signalperiod=9
        )

        # State
        self.entry_price = None
        self.stop_price = None
        self.tp1_price = None
        self.tp2_price = None
        self.bars_in_trade = 0
        self.tp1_hit = False
        self.highest_high = None
        self.lowest_low = None
        self.trade_direction = None
        print("🌙✨ Indicators ready! Let's ride the volatility waves 🚀")

    def regime(self, i):
        """Classify volatility regime using ATR and BBW."""
        if i < self.bb_period + 1 or np.isnan(self.atr[i]) or np.isnan(self.bb_middle[i]):
            return "Normal"
        bbw = (self.bb_upper[i] - self.bb_lower[i]) / self.bb_middle[i] if self.bb_middle[i] != 0 else 0
        # Compute rolling BBW history for percentile
        hist_start = max(0, i - 50)
        bbw_hist = []
        for j in range(hist_start, i):
            if not np.isnan(self.bb_middle[j]) and self.bb_middle[j] != 0:
                bbw_hist.append((self.bb_upper[j] - self.bb_lower[j]) / self.bb_middle[j])
        if len(bbw_hist) < 10:
            return "Normal"
        pct = np.sum(np.array(bbw_hist) < bbw) / len(bbw_hist)
        if pct < 0.25:
            return "Low"
        elif pct < 0.75:
            return "Normal"
        elif pct < 0.95:
            return "High"
        else:
            return "Extreme"

    def band_mult(self, regime):
        if regime == "Low":
            return 1.0
        elif regime == "Normal":
            return 1.5
        elif regime == "High":
            return 2.0
        else:
            return 2.5

    def size_mult(self, regime):
        if regime == "Low":
            return 1.0
        elif regime == "Normal":
            return 1.0
        elif regime == "High":
            return 0.5
        else:
            return 0.25

    def next(self):
        i = len(self.data) - 1
        if i < 50:
            return

        price = self.data.Close[-1]
        atr = self.atr[-1]
        ema = self.ema[-1]
        rsi = self.rsi[-1]
        rsi_prev = self.rsi[-2]
        vol = self.data.Volume[-1]
        vol_ma = self.vol_ma[-1]
        macd_hist = self.macd_hist[-1]
        macd_hist_prev = self.macd_hist[-2]

        if np.isnan(atr) or np.isnan(ema) or np.isnan(rsi) or np.isnan(vol_ma):
            return

        reg = self.regime(i)
        mult = self.band_mult(reg)

        upper_band = ema + mult * atr
        lower_band = ema - mult * atr

        ema_slope = ema - self.ema[-2] if not np.isnan(self.ema[-2]) else 0
        vol_ok = vol > 1.2 * vol_ma

        # ===== Manage open position =====
        if self.position:
            self.bars_in_trade += 1
            if self.trade_direction == "long":
                if self.data.High[-1] > self.highest_high:
                    self.highest_high = self.data.High[-1]

                # TP1 scale out
                if not self.tp1_hit and self.data.High[-1] >= self.tp1_price:
                    print(f"🌙✨ TP1 hit LONG @ {self.tp1_price:.2f} — scaling out 50% 🚀")
                    self.position.close(portion=0.5)
                    self.tp1_hit = True
                    self.stop_price = max(self.stop_price, self.entry_price)

                # TP2 full exit
                if self.data.High[-1] >= self.tp2_price:
                    print(f"🌙🎯 TP2 hit LONG @ {self.tp2_price:.2f} — closing full position 💰")
                    self.position.close()
                    self._reset()
                    return

                # Trailing stop after TP1
                if self.tp1_hit:
                    trail_stop = self.highest_high - self.trail_mult * atr
                    if trail_stop > self.stop_price:
                        self.stop_price = trail_stop

                # Stop loss check
                if self.data.Low[-1] <= self.stop_price:
                    print(f"🌙🛑 Stop hit LONG @ {self.stop_price:.2f}")
                    self.position.close()
                    self._reset()
                    return

                # Time stop
                if not self.tp1_hit and self.bars_in_trade >= self.time_stop_bars:
                    if rsi < 50 or macd_hist < macd_hist_prev:
                        print(f"🌙⏰ Time stop LONG — momentum stalled")
                        self.position.close()
                        self._reset()
                        return

                # Extreme regime tighten
                if reg == "Extreme" and not self.tp1_hit:
                    tight_stop = self.entry_price - 0.75 * atr
                    if tight_stop > self.stop_price:
                        self.stop_price = tight_stop

            elif self.trade_direction == "short":
                if self.data.Low[-1] < self.lowest_low:
                    self.lowest_low = self.data.Low[-1]

                if not self.tp1_hit and self.data.Low[-1] <= self.tp1_price:
                    print(f"🌙✨ TP1 hit SHORT @ {self.tp1_price:.2f} — scaling out 50% 🚀")
                    self.position.close(portion=0.5)
                    self.tp1_hit = True
                    self.stop_price = min(self.stop_price, self.entry_price)

                if self.data.Low[-1] <= self.tp2_price:
                    print(f"🌙🎯 TP2 hit SHORT @ {self.tp2_price:.2f} — closing full position 💰")
                    self.position.close()
                    self._reset()
                    return

                if self.tp1_hit:
                    trail_stop = self.lowest_low + self.trail_mult * atr
                    if trail_stop < self.stop_price:
                        self.stop_price = trail_stop

                if self.data.High[-1] >= self.stop_price:
                    print(f"🌙🛑 Stop hit SHORT @ {self.stop_price:.2f}")
                    self.position.close()
                    self._reset()
                    return

                if not self.tp1_hit and self.bars_in_trade >= self.time_stop_bars:
                    if rsi > 50 or macd_hist > macd_hist_prev:
                        print(f"🌙⏰ Time stop SHORT — momentum stalled")
                        self.position.close()
                        self._reset()
                        return

                if reg == "Extreme" and not self.tp1_hit:
                    tight_stop = self.entry_price + 0.75 * atr
                    if tight_stop < self.stop_price:
                        self.stop_price = tight_stop

            return

        # ===== Entry Logic =====
        if reg == "Extreme":
            return  # avoid extreme unless reduced size — we skip for simplicity

        if not vol_ok:
            return

        # Long entry
        long_signal = (
            price > upper_band and
            rsi > 55 and rsi > rsi_prev and
            ema_slope > 0 and
            reg in ("Normal", "High")
        )

        # Short entry
        short_signal = (
            price < lower_band and
            rsi < 45 and rsi < rsi_prev and
            ema_slope < 0 and
            reg in ("Normal", "High")
        )

        # Risk-based sizing
        stop_dist = self.atr_stop_mult * atr
        if stop_dist <= 0:
            return

        equity = self.equity
        risk_amount = equity * self.risk_pct * self.size_mult(reg)
        raw_size = risk_amount / stop_dist
        size = int(round(raw_size))
        if size < 1:
            size = 1

        if long_signal:
            self.entry_price = price
            self.stop_price = price - stop_dist
            self.tp1_price = price + self.tp1_mult * atr
            self.tp2_price = price + self.tp2_mult * atr
            self.bars_in_trade = 0
            self.tp1_hit = False
            self.highest_high = price
            self.trade_direction = "long"
            print(f"🌙🚀 LONG ENTRY @ {price:.2f} | Regime: {reg} | Size: {size} | Stop: {self.stop_price:.2f} | TP1: {self.tp1_price:.2f} | TP2: {self.tp2_price:.2f}")
            self.buy(size=size)

        elif short_signal:
            self.entry_price = price
            self.stop_price = price + stop_dist
            self.tp1_price = price - self.tp1_mult * atr
            self.tp2_price = price - self.tp2_mult * atr
            self.bars_in_trade = 0
            self.tp1_hit = False
            self.lowest_low = price
            self.trade_direction = "short"
            print(f"🌙🔻 SHORT ENTRY @ {price:.2f} | Regime: {reg} | Size: {size} | Stop: {self.stop_price:.2f} | TP1: {self.tp1_price:.2f} | TP2: {self.tp2_price:.2f}")
            self.sell(size=size)

    def _reset(self):
        self.entry_price = None
        self.stop_price = None
        self.tp1_price = None
        self.tp2_price = None
        self.bars_in_trade = 0
        self.tp1_hit = False
        self.highest_high = None
        self.lowest_low = None
        self.trade_direction = None


bt = Backtest(
    data,
    KineticVolatility,
    cash=1_000_000,
    commission=0.0002,
    exclusive=False,
    margin=1.0,
    trade_on_close=False
)

stats = bt.run()
print(stats)
print(stats._strategy)