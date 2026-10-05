import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# 🌙 Moon Dev's VolatilityDivergence Backtest 🚀

data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"

data = pd.read_csv(data_path)
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'datetime': 'Date',
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})
data['Date'] = pd.to_datetime(data['Date'])
data = data.set_index('Date')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

print(f"🌙 Data loaded: {len(data)} rows, columns: {list(data.columns)} ✨")


class VolatilityDivergence(Strategy):
    rsi_period = 14
    atr_period = 14
    swing_window = 5
    lookback = 40
    ema_period = 50
    risk_pct = 0.02
    rr_target = 1.5
    atr_stop_buffer = 0.5
    max_bars_in_trade = 20

    def init(self):
        print("🌙✨ Initializing Moon Dev's VolatilityDivergence Strategy ✨🌙")
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=self.rsi_period)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close,
                          timeperiod=self.atr_period)
        self.ema = self.I(talib.EMA, self.data.Close, timeperiod=self.ema_period)
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=self.swing_window)
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=self.swing_window)
        self.entry_bar = 0
        self.stop_price = 0.0
        self.tp_price = 0.0

    def next(self):
        price = self.data.Close[-1]

        if self.position:
            # Time-based exit
            if len(self.data) - self.entry_bar >= self.max_bars_in_trade:
                print(f"⏰ Moon Dev time exit at {price:.2f} 🌙")
                self.position.close()
                return

            # RSI midline cross exit
            if self.position.is_long and self.rsi[-1] < 50:
                print(f"📉 RSI crossed below 50, exiting long at {price:.2f} 🌙")
                self.position.close()
                return
            if self.position.is_short and self.rsi[-1] > 50:
                print(f"📈 RSI crossed above 50, exiting short at {price:.2f} 🌙")
                self.position.close()
                return
            return

        if len(self.data) < self.lookback + self.swing_window:
            return

        # Detect swing points in lookback window
        atr_val = self.atr[-1]
        if atr_val <= 0 or np.isnan(atr_val):
            return

        # Find prior swing high/low within lookback
        # A bar is a swing high if its High equals the MAX over swing_window bars ending at that bar
        prior_high_idx = None
        prior_low_idx = None
        start_i = max(self.swing_window, len(self.data) - self.lookback)
        end_i = len(self.data) - self.swing_window - 1
        for i in range(end_i, start_i - 1, -1):
            if prior_high_idx is None:
                if self.data.High[i] == self.swing_high[i]:
                    prior_high_idx = i
            if prior_low_idx is None:
                if self.data.Low[i] == self.swing_low[i]:
                    prior_low_idx = i
            if prior_high_idx is not None and prior_low_idx is not None:
                break

        # Volatility contraction: ATR making lower highs
        atr_contracting = False
        if len(self.atr) > 15:
            recent_atr_max = np.max(self.atr[-5:])
            older_atr_max = np.max(self.atr[-15:-5])
            if recent_atr_max < older_atr_max:
                atr_contracting = True

        # ---- BEARISH SETUP (Short) ----
        if prior_high_idx is not None:
            prior_high_price = self.data.High[prior_high_idx]
            prior_high_rsi = self.rsi[prior_high_idx]
            curr_high_price = self.data.High[-1]
            curr_high_rsi = self.rsi[-1]

            if not (np.isnan(prior_high_rsi) or np.isnan(curr_high_rsi)):
                price_higher_high = curr_high_price > prior_high_price
                rsi_lower_high = curr_high_rsi < prior_high_rsi
                below_ema = price < self.ema[-1]
                rsi_overbought = curr_high_rsi > 60

                if (price_higher_high and rsi_lower_high and atr_contracting
                        and below_ema and rsi_overbought):
                    stop_price = curr_high_price + self.atr_stop_buffer * atr_val
                    risk = stop_price - price
                    if risk > 0:
                        tp_price = price - self.rr_target * risk
                        size_frac = self.risk_pct * (price / risk)
                        size_frac = min(max(size_frac, 0.01), 0.99)
                        print(f"🚀🌙 SHORT signal! Price HH {curr_high_price:.2f} vs {prior_high_price:.2f}, "
                              f"RSI LH {curr_high_rsi:.1f} vs {prior_high_rsi:.1f}, ATR contracting ✨")
                        self.sell(size=size_frac, sl=stop_price, tp=tp_price)
                        self.entry_bar = len(self.data)
                        return

        # ---- BULLISH SETUP (Long) ----
        if prior_low_idx is not None:
            prior_low_price = self.data.Low[prior_low_idx]
            prior_low_rsi = self.rsi[prior_low_idx]
            curr_low_price = self.data.Low[-1]
            curr_low_rsi = self.rsi[-1]

            if not (np.isnan(prior_low_rsi) or np.isnan(curr_low_rsi)):
                price_lower_low = curr_low_price < prior_low_price
                rsi_higher_low = curr_low_rsi > prior_low_rsi
                above_ema = price > self.ema[-1]
                rsi_oversold = curr_low_rsi < 40

                if (price_lower_low and rsi_higher_low and atr_contracting
                        and above_ema and rsi_oversold):
                    stop_price = curr_low_price - self.atr_stop_buffer * atr_val
                    risk = price - stop_price
                    if risk > 0:
                        tp_price = price + self.rr_target * risk
                        size_frac = self.risk_pct * (price / risk)
                        size_frac = min(max(size_frac, 0.01), 0.99)
                        print(f"🚀🌙 LONG signal! Price LL {curr_low_price:.2f} vs {prior_low_price:.2f}, "
                              f"RSI HL {curr_low_rsi:.1f} vs {prior_low_rsi:.1f}, ATR contracting ✨")
                        self.buy(size=size_frac, sl=stop_price, tp=tp_price)
                        self.entry_bar = len(self.data)
                        return


bt = Backtest(data, VolatilityDivergence, cash=1_000_000, commission=0.002)
stats = bt.run()
print(stats)
print(stats._strategy)