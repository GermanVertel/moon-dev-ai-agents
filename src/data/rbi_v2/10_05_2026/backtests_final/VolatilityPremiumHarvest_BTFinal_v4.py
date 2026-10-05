import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# 🌙 Moon Dev's VolatilityPremiumHarvest Strategy 🚀
DATA_PATH = '/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv'

print("🌙 Loading data from Moon Dev's vault...")
data = pd.read_csv(DATA_PATH)

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Rename to backtesting.py format
data = data.rename(columns={
    'datetime': 'Datetime',
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

# Set datetime index
data['Datetime'] = pd.to_datetime(data['Datetime'])
data = data.set_index('Datetime')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

print(f"✨ Data loaded: {len(data)} bars from {data.index[0]} to {data.index[-1]}")


class VolatilityPremiumHarvest(Strategy):
    vol_short_period = 14
    vol_long_period = 60
    premium_lookback = 100
    premium_std_mult = 1.0
    atr_period = 14
    risk_pct = 0.02
    reward_ratio = 0.60
    stop_atr_mult = 1.5

    def init(self):
        print("🌙 Initializing Moon Dev's VolatilityPremiumHarvest indicators...")

        close = self.data.Close
        high = self.data.High
        low = self.data.Low

        # ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        # Short-term realized volatility
        def _short_vol(x):
            s = pd.Series(x)
            return (s.pct_change().rolling(self.vol_short_period).std() * np.sqrt(252)).values
        self.short_vol = self.I(_short_vol, close, name='short_vol')

        # Long-term realized volatility
        def _long_vol(x):
            s = pd.Series(x)
            return (s.pct_change().rolling(self.vol_long_period).std() * np.sqrt(252)).values
        self.long_vol = self.I(_long_vol, close, name='long_vol')

        # Volatility premium
        def _premium(s, l):
            return np.asarray(s) - np.asarray(l)
        self.vol_premium = self.I(_premium, self.short_vol, self.long_vol, name='vol_premium')

        # Rolling mean of premium
        def _roll_mean(x):
            return pd.Series(x).rolling(self.premium_lookback).mean().values
        self.premium_mean = self.I(_roll_mean, self.vol_premium, name='premium_mean')

        # Rolling std of premium
        def _roll_std(x):
            return pd.Series(x).rolling(self.premium_lookback).std().values
        self.premium_std = self.I(_roll_std, self.vol_premium, name='premium_std')

        # Upper threshold
        def _upper(m, s):
            return np.asarray(m) + self.premium_std_mult * np.asarray(s)
        self.premium_upper = self.I(_upper, self.premium_mean, self.premium_std, name='premium_upper')

        # Term structure slope
        def _slope(l, s):
            return np.asarray(l) - np.asarray(s)
        self.term_slope = self.I(_slope, self.long_vol, self.short_vol, name='term_slope')

        # SMA
        self.sma50 = self.I(talib.SMA, close, timeperiod=50)

        print("✨ Indicators initialized! Ready to harvest volatility premium 🌙")

    def next(self):
        # Need enough history for all indicators
        if len(self.data) < self.premium_lookback + self.vol_long_period + 5:
            return

        price = self.data.Close[-1]

        premium = self.vol_premium[-1]
        upper = self.premium_upper[-1]
        mean = self.premium_mean[-1]
        slope = self.term_slope[-1]
        atr = self.atr[-1]

        if np.isnan(premium) or np.isnan(upper) or np.isnan(mean) or np.isnan(slope) or np.isnan(atr):
            return

        if atr <= 0:
            return

        # ENTRY CONDITIONS
        entry_signal = (
            premium >= upper and
            slope > 0 and
            upper > 0
        )

        # EXIT CONDITIONS
        exit_signal = premium <= mean

        if not self.position:
            if entry_signal:
                risk_amount = self.equity * self.risk_pct
                stop_distance = atr * self.stop_atr_mult

                if stop_distance > 0:
                    # Use fractional sizing (fraction of equity) for backtesting.py compatibility
                    # Convert risk-based unit size into a fraction of equity
                    unit_size = risk_amount / stop_distance
                    fraction = (unit_size * price) / self.equity
                    # Cap fraction to at most 0.95 to avoid margin issues
                    fraction = min(fraction, 0.95)
                    # Ensure fraction is valid (0 < fraction < 1)
                    if fraction <= 0:
                        fraction = 0.01

                    stop_price = price - stop_distance
                    if self.reward_ratio < 1:
                        take_profit = price + stop_distance * self.reward_ratio / (1 - self.reward_ratio)
                    else:
                        take_profit = price + stop_distance

                    print(f"🌙✨ ENTRY SIGNAL DETECTED! ✨🌙")
                    print(f"   Premium: {premium:.4f} >= Upper: {upper:.4f}")
                    print(f"   Term Slope (contango): {slope:.4f}")
                    print(f"   Price: {price:.2f}, ATR: {atr:.2f}")
                    print(f"   Position Size (fraction): {fraction:.4f}")
                    print(f"   Stop: {stop_price:.2f}, Target: {take_profit:.2f}")
                    print(f"   🚀 Harvesting volatility premium!")

                    self.buy(size=fraction)
        else:
            # Use entry price from last trade
            if len(self.trades) > 0:
                entry_price = self.trades[-1].entry_price
            else:
                entry_price = price

            stop_price = entry_price - atr * self.stop_atr_mult
            target_price = entry_price + atr * self.stop_atr_mult * 1.5

            if exit_signal:
                print(f"🌙 EXIT: Premium reverted to mean ({premium:.4f} <= {mean:.4f})")
                print(f"   Closing position at {price:.2f} 🎯")
                self.position.close()

            elif price <= stop_price:
                print(f"🌙 STOP LOSS triggered at {price:.2f} (stop: {stop_price:.2f}) 🛑")
                self.position.close()

            elif price >= target_price:
                print(f"🌙 TAKE PROFIT hit at {price:.2f} (target: {target_price:.2f}) 💰")
                self.position.close()


print("=" * 60)
print("🌙 Moon Dev's VolatilityPremiumHarvest Backtest 🚀")
print("=" * 60)

bt = Backtest(
    data,
    VolatilityPremiumHarvest,
    cash=1_000_000,
    commission=0.002
)

stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙✨ Backtest complete! Moon Dev out! 🚀")