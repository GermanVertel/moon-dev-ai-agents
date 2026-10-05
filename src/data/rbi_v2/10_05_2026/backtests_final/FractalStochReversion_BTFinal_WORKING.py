import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

print("🌙✨ Moon Dev's FractalStochReversion Backtest Initializing... 🚀")

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
print(f"🌙 Data loaded: {len(data)} bars from {data.index[0]} to {data.index[-1]} ✨")


class FractalStochReversion(Strategy):
    # Parameters
    stochrsi_period = 14
    stochrsi_k = 3
    stochrsi_d = 3
    vdma_period = 20
    atr_period = 14
    fractal_period = 2
    risk_pct = 0.01
    rr_min = 1.5
    time_stop = 15
    atr_lookback = 100
    atr_pct_threshold = 0.95

    def init(self):
        print("🌙 Initializing indicators... ✨")
        close = self.data.Close
        high = self.data.High
        low = self.data.Low

        # StochRSI
        rsi = self.I(talib.RSI, close, timeperiod=self.stochrsi_period)
        rsi_min = self.I(talib.MIN, rsi, timeperiod=self.stochrsi_period)
        rsi_max = self.I(talib.MAX, rsi, timeperiod=self.stochrsi_period)

        # StochRSI raw
        stoch_raw = (rsi - rsi_min) / (rsi_max - rsi_min + 1e-10) * 100
        self.stoch_k = self.I(talib.SMA, stoch_raw, timeperiod=self.stochrsi_k)
        self.stoch_d = self.I(talib.SMA, self.stoch_k, timeperiod=self.stochrsi_d)

        # VDMA - use SMA as adaptive
        self.vdma = self.I(talib.SMA, close, timeperiod=self.vdma_period)

        # ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        # ATR percentile filter
        self.atr_max = self.I(talib.MAX, self.atr, timeperiod=self.atr_lookback)

        # Williams Fractals - use MAX/MIN
        fp = self.fractal_period
        self.fractal_high = self.I(talib.MAX, high, timeperiod=2 * fp + 1)
        self.fractal_low = self.I(talib.MIN, low, timeperiod=2 * fp + 1)

        # Track state
        self.trade_bar = None
        self.stop_price = None
        self.target_price = None
        self.breakeven_set = False
        self.entry_price = None

        print("🌙 Indicators ready! 🚀")

    def next(self):
        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]

        # Volatility filter
        atr_val = self.atr[-1]
        atr_max_val = self.atr_max[-1]
        if atr_max_val > 0 and atr_val > self.atr_pct_threshold * atr_max_val:
            return

        # Manage open trade
        if self.position:
            self._manage_trade(price, high, low)
            return

        # Need enough bars
        if len(self.data) < max(self.vdma_period, self.atr_period, self.stochrsi_period) + 10:
            return

        stoch_k = self.stoch_k[-1]
        stoch_k_prev = self.stoch_k[-2]
        vdma = self.vdma[-1]
        vdma_prev = self.vdma[-3] if len(self.data) > 3 else vdma

        # Fractal levels (confirmed fractals are at index -1-fp)
        fp = self.fractal_period
        if len(self.data) < 2 * fp + 5:
            return

        frac_high = self.fractal_high[-1 - fp]
        frac_low = self.fractal_low[-1 - fp]

        # ---- LONG SETUP ----
        # Bullish divergence: price lower low, StochRSI higher low, oversold
        lookback = 10
        if len(self.data) > lookback + 5:
            recent_low = min(self.data.Low[-lookback:-1])
            prev_low = min(self.data.Low[-2 * lookback:-lookback])

            stoch_recent_low = min(self.stoch_k[-lookback:-1])
            stoch_prev_low = min(self.stoch_k[-2 * lookback:-lookback])

            bullish_div = (recent_low < prev_low) and (stoch_recent_low > stoch_prev_low)
            oversold = stoch_k < 20 or stoch_recent_low < 20

            # VDMA filter: price at/below VDMA or VDMA flat/rising
            vdma_rising = vdma >= vdma_prev
            vdma_ok_long = (price <= vdma * 1.005) or vdma_rising

            # Fractal confirmation: break above fractal high
            fractal_break_long = price > frac_high and self.data.Close[-2] <= frac_high

            if bullish_div and oversold and vdma_ok_long and fractal_break_long:
                stop = frac_low - 0.5 * atr_val
                risk = price - stop
                if risk > 0:
                    target = vdma if vdma > price else price + 1.5 * risk
                    reward = target - price
                    if reward / risk >= self.rr_min:
                        size = int(round((self.equity * self.risk_pct) / risk))
                        size = max(1, min(size, int(self.equity / price)))
                        print(f"🌙🚀 LONG SIGNAL! Price={price:.2f} Stop={stop:.2f} Target={target:.2f} Size={size} ✨")
                        self.buy(size=size)
                        self.stop_price = stop
                        self.target_price = target
                        self.trade_bar = len(self.data)
                        self.breakeven_set = False
                        self.entry_price = price
                        return

        # ---- SHORT SETUP ----
        if len(self.data) > lookback + 5:
            recent_high = max(self.data.High[-lookback:-1])
            prev_high = max(self.data.High[-2 * lookback:-lookback])

            stoch_recent_high = max(self.stoch_k[-lookback:-1])
            stoch_prev_high = max(self.stoch_k[-2 * lookback:-lookback])

            bearish_div = (recent_high > prev_high) and (stoch_recent_high < stoch_prev_high)
            overbought = stoch_k > 80 or stoch_recent_high > 80

            vdma_falling = vdma <= vdma_prev
            vdma_ok_short = (price >= vdma * 0.995) or vdma_falling

            fractal_break_short = price < frac_low and self.data.Close[-2] >= frac_low

            if bearish_div and overbought and vdma_ok_short and fractal_break_short:
                stop = frac_high + 0.5 * atr_val
                risk = stop - price
                if risk > 0:
                    target = vdma if vdma < price else price - 1.5 * risk
                    reward = price - target
                    if reward / risk >= self.rr_min:
                        size = int(round((self.equity * self.risk_pct) / risk))
                        size = max(1, min(size, int(self.equity / price)))
                        print(f"🌙🔻 SHORT SIGNAL! Price={price:.2f} Stop={stop:.2f} Target={target:.2f} Size={size} ✨")
                        self.sell(size=size)
                        self.stop_price = stop
                        self.target_price = target
                        self.trade_bar = len(self.data)
                        self.breakeven_set = False
                        self.entry_price = price
                        return

    def _manage_trade(self, price, high, low):
        vdma = self.vdma[-1]
        stoch_k = self.stoch_k[-1]
        atr_val = self.atr[-1]

        # Get entry price from last trade
        entry_price = self.trades[-1].entry_price if self.trades else self.entry_price

        if self.position.is_long:
            # Stop loss
            if low <= self.stop_price:
                print(f"🌙💥 LONG STOP HIT at {self.stop_price:.2f}")
                self.position.close()
                return
            # Target
            if high >= self.target_price:
                print(f"🌙🎯 LONG TARGET HIT at {self.target_price:.2f}")
                self.position.close()
                return
            # Move to breakeven when crossing VDMA
            if not self.breakeven_set and price > vdma:
                self.stop_price = max(self.stop_price, entry_price)
                self.breakeven_set = True
                print(f"🌙✨ LONG moved stop to breakeven at {self.stop_price:.2f}")
            # StochRSI opposite extreme exit
            if stoch_k > 80:
                print(f"🌙⚠️ LONG exit: StochRSI overbought {stoch_k:.1f}")
                self.position.close()
                return
            # Time stop
            if self.trade_bar and len(self.data) - self.trade_bar >= self.time_stop:
                print(f"🌙⏰ LONG time stop after {self.time_stop} bars")
                self.position.close()
                return

        elif self.position.is_short:
            if high >= self.stop_price:
                print(f"🌙💥 SHORT STOP HIT at {self.stop_price:.2f}")
                self.position.close()
                return
            if low <= self.target_price:
                print(f"🌙🎯 SHORT TARGET HIT at {self.target_price:.2f}")
                self.position.close()
                return
            if not self.breakeven_set and price < vdma:
                self.stop_price = min(self.stop_price, entry_price)
                self.breakeven_set = True
                print(f"🌙✨ SHORT moved stop to breakeven at {self.stop_price:.2f}")
            if stoch_k < 20:
                print(f"🌙⚠️ SHORT exit: StochRSI oversold {stoch_k:.1f}")
                self.position.close()
                return
            if self.trade_bar and len(self.data) - self.trade_bar >= self.time_stop:
                print(f"🌙⏰ SHORT time stop after {self.time_stop} bars")
                self.position.close()
                return


bt = Backtest(data, FractalStochReversion, cash=1000000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)