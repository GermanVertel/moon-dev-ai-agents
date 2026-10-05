import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Load data
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Map columns to backtesting.py required format
data.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')
data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

print("🌙 Moon Dev VolatilityCorridor Backtest Loading... ✨")
print(f"📊 Data shape: {data.shape}")
print(f"📈 Date range: {data.index[0]} to {data.index[-1]}")

class VolatilityCorridor(Strategy):
    """
    VolatilityCorridor Strategy 🌙
    Short-volatility premium harvesting conditioned on tight VIX/SPX co-movement regime.
    Since we only have BTC data here, we proxy the regime filter using price action:
    - Rolling correlation of returns vs a smoothed baseline (proxy for VIX/SPX correlation regime)
    - Entry on new daily high (momentum/complacency)
    - Exit on next day's low or stop loss
    """

    # Strategy parameters
    corr_window = 20          # Rolling correlation lookback window
    corr_band = 0.05          # 5% tolerance band
    high_lookback = 20        # Lookback for daily high detection
    low_lookback = 20         # Lookback for daily low (exit reference)
    stop_loss_pct = 0.015     # 1.5% stop loss
    take_profit_pct = 0.02    # 2% take profit (next-day-low reference)
    risk_pct = 0.02           # Risk 2% of capital per trade

    def init(self):
        print("🌙✨ Initializing VolatilityCorridor indicators... 🚀")

        # Rolling high (proxy for daily high threshold)
        self.rolling_high = self.I(talib.MAX, self.data.High, timeperiod=self.high_lookback)
        # Rolling low (proxy for next-day low exit reference)
        self.rolling_low = self.I(talib.MIN, self.data.Low, timeperiod=self.low_lookback)

        # Returns for correlation proxy
        close = pd.Series(self.data.Close)
        returns = close.pct_change().fillna(0).values

        # Proxy "VIX-like" series: rolling volatility of returns
        vol_series = pd.Series(returns).rolling(self.corr_window).std().fillna(0).values

        # Rolling correlation between returns and volatility proxy
        # In a "normal" regime, returns and vol are negatively correlated.
        # We compute rolling correlation then check it's within the band.
        ret_series = pd.Series(returns)
        vol_s = pd.Series(vol_series)
        roll_corr = ret_series.rolling(self.corr_window).corr(vol_s).fillna(0).values

        self.roll_corr = self.I(lambda: roll_corr, name='RollCorr')

        # Mean of correlation for band check
        self.corr_mean = self.I(talib.SMA, self.roll_corr, timeperiod=self.corr_window)

        print("🌙 Indicators ready! Let's hunt some premium! 🚀💰")

    def next(self):
        # Skip if not enough data
        if len(self.data) < self.corr_window + 2:
            return

        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]

        # --- Regime Filter: |correlation| within 5% band around mean ---
        corr = self.roll_corr[-1]
        corr_mean = self.corr_mean[-1]
        in_regime = abs(corr - corr_mean) <= self.corr_band

        # --- Entry Trigger: new daily high (complacency) ---
        new_high = high >= self.rolling_high[-1]

        # --- Exit reference: next day's low ---
        hit_low = low <= self.rolling_low[-1]

        # --- Signal logging ---
        if in_regime and new_high and not self.position:
            print(f"🌙✨ ENTRY SIGNAL @ {self.data.index[-1]} | Price: {price:.2f} | Corr: {corr:.4f} (band OK) | New High: {high:.2f} 🚀")

        # --- ENTRY: short put proxy -> we SELL (short) the underlying ---
        if in_regime and new_high and not self.position:
            # Position sizing: risk-based
            risk_amount = self.equity * self.risk_pct
            stop_distance = price * self.stop_loss_pct
            position_size = int(round(risk_amount / stop_distance)) if stop_distance > 0 else 0

            if position_size > 0:
                # Cap size to avoid insane leverage
                position_size = min(position_size, 1_000_000)
                print(f"🌙💰 SELLING (short put proxy) size={position_size} @ {price:.2f} 🚀")
                self.sell(size=position_size,
                          sl=price * (1 + self.stop_loss_pct),   # stop above entry for short
                          tp=price * (1 - self.take_profit_pct)) # take profit below entry

        # --- EXIT: hit next-day low (profit target for short) ---
        if self.position and self.position.is_short:
            if hit_low:
                print(f"🌙✨ EXIT SIGNAL (next-day low hit) @ {self.data.index[-1]} | Low: {low:.2f} | TP hit! 💰")
                self.position.close()

        # --- Regime exit: correlation breaks out of band ---
        if self.position and self.position.is_short and not in_regime:
            print(f"🌙⚠️ REGIME BREAK! Correlation out of band @ {self.data.index[-1]} | Corr: {corr:.4f} | Closing position! 🛑")
            self.position.close()


# --- Run backtest ---
print("🌙🚀 Starting VolatilityCorridor Backtest... ✨")
bt = Backtest(data, VolatilityCorridor, cash=1_000_000, commission=0.0002)

stats = bt.run()
print("🌙✨ Backtest complete! Printing full stats... 🚀💰")
print(stats)
print(stats._strategy)