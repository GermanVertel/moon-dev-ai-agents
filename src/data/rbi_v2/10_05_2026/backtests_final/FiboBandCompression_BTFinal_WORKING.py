import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Load data
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper column mapping
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

# Set datetime index
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

print("🌙✨ Moon Dev Data Loaded! ✨🌙")
print(f"📊 Shape: {data.shape}")
print(f"📅 Range: {data.index[0]} to {data.index[-1]}")


class FiboBandCompression(Strategy):
    # Strategy parameters
    swing_lookback = 50
    bb_period = 20
    bb_std = 2.0
    trend_sma_period = 200
    ema_fast = 50
    ema_slow = 200
    bbw_contraction_bars = 3
    bbw_min_threshold = 0.01
    fib_tolerance = 0.005  # 0.5% tolerance around fib levels
    risk_pct = 0.01  # 1% risk per trade

    def init(self):
        print("🌙 Initializing FiboBand Compression Strategy... ✨")

        close = self.data.Close

        # Bollinger Bands (20, 2)
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS, close, timeperiod=self.bb_period,
            nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0
        )

        # Bollinger Band Width
        self.bbw = self.I(
            lambda u, m, l: (u - l) / np.where(m == 0, np.nan, m),
            self.bb_upper, self.bb_middle, self.bb_lower
        )

        # Trend filters
        self.sma200 = self.I(talib.SMA, close, timeperiod=self.trend_sma_period)
        self.ema50 = self.I(talib.EMA, close, timeperiod=self.ema_fast)
        self.ema200 = self.I(talib.EMA, close, timeperiod=self.ema_slow)

        # Swing high / low
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=self.swing_lookback)
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=self.swing_lookback)

        # Track BBW contraction counter
        self.bbw_contract_count = 0

        print("🚀 Indicators ready! Moon Dev power engaged! 🌙")

    def next(self):
        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]

        # Skip early bars
        if len(self.data) < self.trend_sma_period + 5:
            return

        # --- Trend Filter ---
        uptrend = (price > self.sma200[-1]) and (self.ema50[-1] > self.ema200[-1])

        # --- Compute Fibonacci levels from swing ---
        sh = self.swing_high[-1]
        sl = self.swing_low[-1]
        rng = sh - sl

        if rng <= 0 or np.isnan(rng):
            return

        fib_382 = sh - 0.382 * rng
        fib_500 = sh - 0.500 * rng
        fib_618 = sh - 0.618 * rng

        fib_levels = [fib_382, fib_500, fib_618]

        # --- Touch detection ---
        near_fib = False
        matched_fib = None
        for fl in fib_levels:
            if fl > 0 and abs(price - fl) / fl <= self.fib_tolerance:
                near_fib = True
                matched_fib = fl
                break

        # --- BB proximity: distance from price to middle < 2 * band width ---
        bb_width_abs = self.bb_upper[-1] - self.bb_lower[-1]
        distance_to_mid = abs(price - self.bb_middle[-1])
        near_middle = distance_to_mid < 2 * bb_width_abs

        # --- BBW above minimum threshold ---
        bbw_ok = (not np.isnan(self.bbw[-1])) and (self.bbw[-1] > self.bbw_min_threshold)

        # --- Bullish confirmation candle ---
        prev_close = self.data.Close[-2]
        prev_open = self.data.Open[-2]
        curr_open = self.data.Open[-1]

        bullish_rejection = (low < min(prev_close, curr_open)) and (price > curr_open) and (price > prev_close)
        bullish_engulf = (price > prev_open) and (curr_open < prev_close) and (price > prev_open)

        bullish_confirm = bullish_rejection or bullish_engulf

        # --- BBW contraction counter ---
        if not np.isnan(self.bbw[-1]) and not np.isnan(self.bbw[-2]):
            if self.bbw[-1] < self.bbw[-2]:
                self.bbw_contract_count += 1
            else:
                self.bbw_contract_count = 0

        # --- ENTRY ---
        if not self.position:
            if uptrend and near_fib and near_middle and bbw_ok and bullish_confirm:
                # Stop loss: tighter of next fib down or swing low
                if matched_fib == fib_382:
                    stop_fib = fib_500
                elif matched_fib == fib_500:
                    stop_fib = fib_618
                else:
                    stop_fib = sl

                stop_price = max(stop_fib, sl)
                risk_per_unit = price - stop_price

                if risk_per_unit > 0:
                    risk_amount = self.equity * self.risk_pct
                    position_size = int(round(risk_amount / risk_per_unit))
                    position_size = max(1, min(position_size, 1000000))

                    # Take profit: prior swing high
                    tp_price = sh

                    print(f"🌙✨ MOON DEV LONG ENTRY! ✨🚀")
                    print(f"   Price: {price:.2f} | Fib: {matched_fib:.2f}")
                    print(f"   Stop: {stop_price:.2f} | TP: {tp_price:.2f}")
                    print(f"   Size: {position_size}")

                    self.buy(size=position_size, sl=stop_price, tp=tp_price)

        # --- EXIT ---
        else:
            if self.bbw_contract_count >= self.bbw_contraction_bars:
                print(f"🌙 BBW Contraction Exit! Volatility squeeze detected. Closing long. ✨")
                self.position.close()
                self.bbw_contract_count = 0


# Run backtest
print("\n🌙🚀 Starting Moon Dev Backtest... 🚀🌙\n")
bt = Backtest(data, FiboBandCompression, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)