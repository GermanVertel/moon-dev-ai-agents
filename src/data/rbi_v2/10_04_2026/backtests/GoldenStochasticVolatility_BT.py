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
data.columns = [col.capitalize() for col in data.columns]

# Set datetime index
if 'Datetime' in data.columns:
    data['Datetime'] = pd.to_datetime(data['Datetime'])
    data = data.set_index('Datetime')

# Ensure required columns
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

print("🌙✨ GoldenStochastic Volatility Backtest Loading... 🚀")
print(f"📊 Data shape: {data.shape}")
print(f"📈 Date range: {data.index[0]} to {data.index[-1]}")


class GoldenStochasticVolatility(Strategy):
    # Strategy parameters
    k_period = 14
    d_period = 3
    smooth_k = 3
    bb_period = 20
    bb_std = 2.0
    atr_period = 14
    bbw_avg_period = 50
    golden_ratio = 0.618
    risk_per_trade = 0.01  # 1% of equity
    stop_buffer_atr = 0.5

    def init(self):
        print("🌙 Initializing GoldenStochastic Volatility indicators... ✨")

        close = self.data.Close
        high = self.data.High
        low = self.data.Low

        # Stochastic Oscillator
        self.stoch_k = self.I(
            lambda c, h, l: talib.STOCH(h, l, c,
                                        fastk_period=self.k_period,
                                        slowk_period=self.smooth_k,
                                        slowk_matype=0,
                                        slowd_period=self.d_period,
                                        slowd_matype=0)[0],
            close, high, low, name="StochK"
        )
        self.stoch_d = self.I(
            lambda c, h, l: talib.STOCH(h, l, c,
                                        fastk_period=self.k_period,
                                        slowk_period=self.smooth_k,
                                        slowk_matype=0,
                                        slowd_period=self.d_period,
                                        slowd_matype=0)[1],
            close, high, low, name="StochD"
        )

        # Bollinger Bands
        self.bb_upper = self.I(talib.SMA, close, timeperiod=self.bb_period, name="BB_Upper")
        self.bb_middle = self.I(talib.SMA, close, timeperiod=self.bb_period, name="BB_Middle")
        self.bb_lower = self.I(talib.SMA, close, timeperiod=self.bb_period, name="BB_Lower")

        # Recalculate BB with std using custom lambda
        def bb_upper_fn(c):
            sma = talib.SMA(c, timeperiod=self.bb_period)
            std = talib.STDDEV(c, timeperiod=self.bb_period, nbdev=1.0)
            return sma + self.bb_std * std

        def bb_lower_fn(c):
            sma = talib.SMA(c, timeperiod=self.bb_period)
            std = talib.STDDEV(c, timeperiod=self.bb_period, nbdev=1.0)
            return sma - self.bb_std * std

        def bb_middle_fn(c):
            return talib.SMA(c, timeperiod=self.bb_period)

        self.bb_upper = self.I(bb_upper_fn, close, name="BB_Upper")
        self.bb_middle = self.I(bb_middle_fn, close, name="BB_Middle")
        self.bb_lower = self.I(bb_lower_fn, close, name="BB_Lower")

        # ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period, name="ATR")

        # BBW = (Upper - Lower) / Middle
        def bbw_fn(upper, lower, middle):
            return (upper - lower) / middle

        self.bbw = self.I(bbw_fn, self.bb_upper, self.bb_lower, self.bb_middle, name="BBW")

        # BBW Average
        self.bbw_avg = self.I(talib.SMA, self.bbw, timeperiod=self.bbw_avg_period, name="BBW_Avg")

        print("🌙✨ Indicators initialized successfully! 🚀")

    def next(self):
        # Need enough data
        if len(self.data) < max(self.bbw_avg_period, self.bb_period, self.atr_period) + 5:
            return

        # Skip if indicators not ready
        if np.isnan(self.stoch_k[-1]) or np.isnan(self.stoch_d[-1]):
            return
        if np.isnan(self.bb_upper[-1]) or np.isnan(self.bb_lower[-1]) or np.isnan(self.bb_middle[-1]):
            return
        if np.isnan(self.atr[-1]) or np.isnan(self.bbw[-1]) or np.isnan(self.bbw_avg[-1]):
            return

        price = self.data.Close[-1]
        k = self.stoch_k[-1]
        d = self.stoch_d[-1]
        k_prev = self.stoch_k[-2]
        d_prev = self.stoch_d[-2]

        upper = self.bb_upper[-1]
        middle = self.bb_middle[-1]
        lower = self.bb_lower[-1]
        atr = self.atr[-1]
        bbw = self.bbw[-1]
        bbw_avg = self.bbw_avg[-1]

        # Volatility Ratio
        if bbw_avg <= 0:
            return
        vr = bbw / bbw_avg

        # Position Size Multiplier
        if vr < 0.8:
            psm = 1.5
        elif vr <= 1.2:
            psm = 1.0
        elif vr <= 2.0:
            psm = 0.6
        else:
            psm = 0.0  # Skip trade during extreme volatility

        # Golden ratio levels
        golden_long_level = lower + (middle - lower) * self.golden_ratio
        golden_short_level = upper - (upper - middle) * self.golden_ratio

        # Manage existing position
        if self.position:
            if self.position.is_long:
                # Take profit at middle band
                if price >= middle:
                    print(f"🌙✨ LONG TP HIT at middle band! Price={price:.2f}, Middle={middle:.2f} 🚀")
                    self.position.close()
                    return
                # Stop loss: price closes below lower band by 0.5 ATR buffer
                stop_price = lower - self.stop_buffer_atr * atr
                if price < stop_price:
                    print(f"🌙💥 LONG SL HIT! Price={price:.2f}, Stop={stop_price:.2f}")
                    self.position.close()
                    return
                # Exit if %K crosses back above 80
                if k > 80 and k_prev <= 80:
                    print(f"🌙✨ LONG EXIT - %K crossed above 80! K={k:.2f}")
                    self.position.close()
                    return
            elif self.position.is_short:
                # Take profit at middle band
                if price <= middle:
                    print(f"🌙✨ SHORT TP HIT at middle band! Price={price:.2f}, Middle={middle:.2f} 🚀")
                    self.position.close()
                    return
                # Stop loss: price closes above upper band by 0.5 ATR buffer
                stop_price = upper + self.stop_buffer_atr * atr
                if price > stop_price:
                    print(f"🌙💥 SHORT SL HIT! Price={price:.2f}, Stop={stop_price:.2f}")
                    self.position.close()
                    return
                # Exit if %K crosses back below 20
                if k < 20 and k_prev >= 20:
                    print(f"🌙✨ SHORT EXIT - %K crossed below 20! K={k:.2f}")
                    self.position.close()
                    return
            return

        # Skip if extreme volatility
        if psm == 0.0:
            return

        # Long Entry
        long_cross = (k_prev <= d_prev) and (k > d) and (k < 20)
        long_golden = price <= golden_long_level
        if long_cross and long_golden:
            # Calculate stop distance
            stop_price = lower - self.stop_buffer_atr * atr
            stop_distance = price - stop_price
            if stop_distance <= 0:
                return

            # Position size based on risk
            risk_amount = self.equity * self.risk_per_trade
            position_size = (risk_amount / stop_distance) * psm
            position_size = int(round(position_size))
            # Cap at 1,000,000 units
            position_size = min(position_size, 1_000_000)
            if position_size <= 0:
                return

            print(f"🌙🚀 LONG ENTRY! Price={price:.2f}, K={k:.2f}, D={d:.2f}, "
                  f"GoldenLevel={golden_long_level:.2f}, VR={vr:.2f}, PSM={psm}, Size={position_size}")
            self.buy(size=position_size)

        # Short Entry
        short_cross = (k_prev >= d_prev) and (k < d) and (k > 80)
        short_golden = price >= golden_short_level
        if short_cross and short_golden:
            # Calculate stop distance
            stop_price = upper + self.stop_buffer_atr * atr
            stop_distance = stop_price - price
            if stop_distance <= 0:
                return

            risk_amount = self.equity * self.risk_per_trade
            position_size = (risk_amount / stop_distance) * psm
            position_size = int(round(position_size))
            position_size = min(position_size, 1_000_000)
            if position_size <= 0:
                return

            print(f"🌙🔻 SHORT ENTRY! Price={price:.2f}, K={k:.2f}, D={d:.2f}, "
                  f"GoldenLevel={golden_short_level:.2f}, VR={vr:.2f}, PSM={psm}, Size={position_size}")
            self.sell(size=position_size)


# Run backtest
bt = Backtest(data, GoldenStochasticVolatility, cash=1_000_000, commission=0.001)

print("🌙✨ Running GoldenStochastic Volatility Backtest... 🚀")
stats = bt.run()
print(stats)
print(stats._strategy)