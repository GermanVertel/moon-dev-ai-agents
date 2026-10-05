import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Load data
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={'datetime': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
data['Date'] = pd.to_datetime(data['Date'])
data = data.set_index('Date')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

print("🌙✨ Moon Dev Backtest Initializing... KineticOscillator loading! 🚀")


class KineticOscillator(Strategy):
    k_period = 14
    d_period = 3
    slowing = 3
    vpr_sma_period = 20
    atr_period = 14
    ema_period = 50
    risk_pct = 0.02
    atr_stop_mult = 1.5
    rr_ratio = 2.0
    max_bars_in_trade = 15

    def init(self):
        high = self.data.High
        low = self.data.Low
        close = self.data.Close
        volume = self.data.Volume

        # Stochastic Oscillator
        self.k, self.d = self.I(talib.STOCH, high, low, close,
                                fastk_period=self.k_period,
                                slowk_period=self.slowing,
                                slowk_matype=0,
                                slowd_period=self.d_period,
                                slowd_matype=0)

        # Range and Volume-per-Range
        rng = high - low
        rng = np.where(rng == 0, 1e-10, rng)
        vpr = volume / rng
        self.vpr = self.I(lambda: vpr, name='VPR')
        self.vpr_sma = self.I(talib.SMA, self.vpr, timeperiod=self.vpr_sma_period)

        # ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        # EMA trend filter
        self.ema = self.I(talib.EMA, close, timeperiod=self.ema_period)

        self.bar_count = 0
        self.entry_bar = None
        self.stop_price = None
        self.tp_price = None

        print("🌙 Indicators initialized: STOCH, VPR, VPR_SMA, ATR, EMA ✨")

    def next(self):
        self.bar_count += 1

        if len(self.data) < max(self.ema_period, self.vpr_sma_period, self.k_period + self.slowing) + 2:
            return

        k = self.k[-1]
        d = self.d[-1]
        k_prev = self.k[-2]
        d_prev = self.d[-2]
        vpr = self.vpr[-1]
        vpr_sma = self.vpr_sma[-1]
        atr = self.atr[-1]
        ema = self.ema[-1]
        close = self.data.Close[-1]
        prev_high = self.data.High[-2]
        prev_low = self.data.Low[-2]

        if np.isnan(k) or np.isnan(d) or np.isnan(vpr_sma) or np.isnan(atr) or np.isnan(ema):
            return

        # Manage open position
        if self.position:
            bars_held = self.bar_count - self.entry_bar

            # Trailing stop update
            if self.position.is_long:
                new_stop = close - self.atr_stop_mult * atr
                if self.stop_price is None or new_stop > self.stop_price:
                    self.stop_price = new_stop

                # Exit conditions
                if k < d and k_prev >= d_prev:
                    print(f"🌙 Long exit: %K crossed below %D at {close:.2f} ✨")
                    self.position.close()
                elif k > 80:
                    print(f"🌙 Long exit: %K overbought ({k:.2f}) at {close:.2f} 🚀")
                    self.position.close()
                elif close <= self.stop_price:
                    print(f"🌙 Long STOP hit at {close:.2f} (stop={self.stop_price:.2f}) 💥")
                    self.position.close()
                elif self.tp_price and close >= self.tp_price:
                    print(f"🌙 Long TP hit at {close:.2f} (tp={self.tp_price:.2f}) 🎯")
                    self.position.close()
                elif bars_held >= self.max_bars_in_trade:
                    print(f"🌙 Long time exit after {bars_held} bars at {close:.2f} ⏰")
                    self.position.close()

            elif self.position.is_short:
                new_stop = close + self.atr_stop_mult * atr
                if self.stop_price is None or new_stop < self.stop_price:
                    self.stop_price = new_stop

                if k > d and k_prev <= d_prev:
                    print(f"🌙 Short exit: %K crossed above %D at {close:.2f} ✨")
                    self.position.close()
                elif k < 20:
                    print(f"🌙 Short exit: %K oversold ({k:.2f}) at {close:.2f} 🚀")
                    self.position.close()
                elif close >= self.stop_price:
                    print(f"🌙 Short STOP hit at {close:.2f} (stop={self.stop_price:.2f}) 💥")
                    self.position.close()
                elif self.tp_price and close <= self.tp_price:
                    print(f"🌙 Short TP hit at {close:.2f} (tp={self.tp_price:.2f}) 🎯")
                    self.position.close()
                elif bars_held >= self.max_bars_in_trade:
                    print(f"🌙 Short time exit after {bars_held} bars at {close:.2f} ⏰")
                    self.position.close()

            return

        # Entry logic
        vol_confirm = vpr > vpr_sma

        # Long entry
        long_signal = (k_prev < d_prev and k > d and k < 20 and d < 20
                       and vol_confirm and close > prev_high and close > ema)

        # Short entry
        short_signal = (k_prev > d_prev and k < d and k > 80 and d > 80
                        and vol_confirm and close < prev_low and close < ema)

        if long_signal:
            stop_dist = self.atr_stop_mult * atr
            if stop_dist <= 0:
                return
            risk_amount = self.equity * self.risk_pct
            position_size = int(round(risk_amount / stop_dist))
            if position_size < 1:
                position_size = 1
            max_size = int(self.equity / close)
            position_size = min(position_size, max_size)
            if position_size < 1:
                return
            self.stop_price = close - stop_dist
            self.tp_price = close + stop_dist * self.rr_ratio
            self.entry_bar = self.bar_count
            print(f"🚀🌙 LONG ENTRY at {close:.2f} | %K={k:.2f} %D={d:.2f} | VPR={vpr:.4f} > SMA={vpr_sma:.4f} | size={position_size} | stop={self.stop_price:.2f} tp={self.tp_price:.2f} ✨")
            self.buy(size=position_size)

        elif short_signal:
            stop_dist = self.atr_stop_mult * atr
            if stop_dist <= 0:
                return
            risk_amount = self.equity * self.risk_pct
            position_size = int(round(risk_amount / stop_dist))
            if position_size < 1:
                position_size = 1
            max_size = int(self.equity / close)
            position_size = min(position_size, max_size)
            if position_size < 1:
                return
            self.stop_price = close + stop_dist
            self.tp_price = close - stop_dist * self.rr_ratio
            self.entry_bar = self.bar_count
            print(f"🚀🌙 SHORT ENTRY at {close:.2f} | %K={k:.2f} %D={d:.2f} | VPR={vpr:.4f} > SMA={vpr_sma:.4f} | size={position_size} | stop={self.stop_price:.2f} tp={self.tp_price:.2f} ✨")
            self.sell(size=position_size)


bt = Backtest(data, KineticOscillator, cash=1_000_000, commission=0.001, exclusive_orders=True)
stats = bt.run()
print(stats)
print(stats._strategy)