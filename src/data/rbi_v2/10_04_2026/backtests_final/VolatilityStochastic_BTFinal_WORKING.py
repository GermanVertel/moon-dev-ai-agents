import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# 🌙 Moon Dev's VolatilityStochastic Backtest 🚀

data_path = '/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv'

print("🌙 Loading Moon Dev data from the cosmic archives...")
data = pd.read_csv(data_path)
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper column mapping
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

print(f"✨ Data loaded: {len(data)} cosmic candles ready for launch 🚀")


class VolatilityStochastic(Strategy):
    # Strategy parameters
    bb_period = 20
    bb_std = 2.0
    stoch_k = 14
    stoch_d = 3
    stoch_smooth = 3
    squeeze_lookback = 20
    squeeze_percentile = 10  # lowest 10%
    risk_pct = 0.02  # 2% risk per trade
    atr_period = 14

    def init(self):
        print("🌙 Initializing Moon Dev indicators...")
        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)

        # Bollinger Bands
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS, close, timeperiod=self.bb_period,
            nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0
        )

        # Bollinger Band Width
        self.bb_width = self.I(
            lambda u, l: u - l, self.bb_upper, self.bb_lower
        )

        # Squeeze threshold: lowest 10% of BB width over lookback
        self.squeeze_threshold = self.I(
            lambda w: pd.Series(w).rolling(self.squeeze_lookback).quantile(self.squeeze_percentile / 100.0).values,
            self.bb_width
        )

        # Stochastic Oscillator
        self.stoch_k_line, self.stoch_d_line = self.I(
            talib.STOCH, high, low, close,
            fastk_period=self.stoch_k,
            slowk_period=self.stoch_smooth,
            slowk_matype=0,
            slowd_period=self.stoch_d,
            slowd_matype=0
        )

        # ATR for stop placement
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        print("✨ Indicators initialized! Ready to hunt breakouts 🎯")

    def next(self):
        price = self.data.Close[-1]

        # Skip if not enough data
        if len(self.data) < self.squeeze_lookback + 5:
            return

        # Check for squeeze condition
        squeeze_thresh = self.squeeze_threshold[-1]
        current_width = self.bb_width[-1]
        is_squeeze = (not np.isnan(squeeze_thresh)) and (current_width <= squeeze_thresh)

        stoch_k = self.stoch_k_line[-1]
        stoch_d = self.stoch_d_line[-1]
        stoch_k_prev = self.stoch_k_line[-2]
        stoch_d_prev = self.stoch_d_line[-2]

        # Stochastic crossovers (replaced backtesting.lib.crossover)
        k_cross_above_d = stoch_k_prev <= stoch_d_prev and stoch_k > stoch_d
        k_cross_below_d = stoch_k_prev >= stoch_d_prev and stoch_k < stoch_d

        upper = self.bb_upper[-1]
        lower = self.bb_lower[-1]
        middle = self.bb_middle[-1]

        # ============ EXIT LOGIC ============
        if self.position:
            if self.position.is_long:
                # Exit on stoch overbought + bearish cross
                if stoch_k > 80 and k_cross_below_d:
                    print(f"🌙 EXIT LONG: Stoch overbought reversal at {price:.2f} 💫")
                    self.position.close()
                # Exit if price falls below middle band (trailing)
                elif price < middle:
                    print(f"🌙 EXIT LONG: Price broke middle band at {price:.2f} 🛑")
                    self.position.close()

            elif self.position.is_short:
                # Exit on stoch oversold + bullish cross
                if stoch_k < 20 and k_cross_above_d:
                    print(f"🌙 EXIT SHORT: Stoch oversold reversal at {price:.2f} 💫")
                    self.position.close()
                # Exit if price rises above middle band
                elif price > middle:
                    print(f"🌙 EXIT SHORT: Price broke middle band at {price:.2f} 🛑")
                    self.position.close()
            return

        # ============ ENTRY LOGIC ============
        if not is_squeeze:
            return

        # LONG ENTRY
        if (price > upper and
                k_cross_above_d and
                stoch_k > 20):
            atr_val = self.atr[-1]
            if np.isnan(atr_val):
                return
            stop_price = min(middle, price - 1.5 * atr_val)
            risk = price - stop_price
            if risk <= 0:
                return
            # Position sizing as fraction of equity (risk-based)
            risk_amount = 1_000_000 * self.risk_pct
            size_units = int(round(risk_amount / risk))
            if size_units <= 0:
                return
            # Convert to fraction of equity for backtesting.py
            equity = self.equity
            size_frac = min(0.95, (size_units * price) / equity)
            if size_frac <= 0 or size_frac >= 1:
                size_frac = 0.5
            print(f"🚀 LONG BREAKOUT! Price={price:.2f} | K={stoch_k:.1f} D={stoch_d:.1f} | SizeFrac={size_frac:.4f} 🌙")
            self.buy(size=size_frac, sl=stop_price)

        # SHORT ENTRY
        elif (price < lower and
                k_cross_below_d and
                stoch_k < 80):
            atr_val = self.atr[-1]
            if np.isnan(atr_val):
                return
            stop_price = max(middle, price + 1.5 * atr_val)
            risk = stop_price - price
            if risk <= 0:
                return
            risk_amount = 1_000_000 * self.risk_pct
            size_units = int(round(risk_amount / risk))
            if size_units <= 0:
                return
            equity = self.equity
            size_frac = min(0.95, (size_units * price) / equity)
            if size_frac <= 0 or size_frac >= 1:
                size_frac = 0.5
            print(f"🔻 SHORT BREAKOUT! Price={price:.2f} | K={stoch_k:.1f} D={stoch_d:.1f} | SizeFrac={size_frac:.4f} 🌙")
            self.sell(size=size_frac, sl=stop_price)


print("🌙✨ Launching Moon Dev VolatilityStochastic Backtest ✨🚀")
bt = Backtest(data, VolatilityStochastic, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)