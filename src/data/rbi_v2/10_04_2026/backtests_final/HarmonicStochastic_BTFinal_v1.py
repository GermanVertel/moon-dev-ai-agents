import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Load data
data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper column mapping
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

# Set datetime index
if 'datetime' in data.columns:
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')

print("🌙 Moon Dev HarmonicStochastic Backtest Loading... ✨")
print(f"📊 Data shape: {data.shape}")
print(f"🚀 First few rows:\n{data.head()}")


def _harm_osc_from_sine(x):
    """Convert HT_SINE output (-1..1) into 0..100 oscillator."""
    return (x + 1.0) * 50.0


class HarmonicStochastic(Strategy):
    # Strategy parameters
    stoch_k_period = 9
    stoch_k_smooth = 3
    stoch_d_period = 3
    atr_period = 14
    donchian_period = 20
    vol_ma_period = 20
    risk_pct = 0.01  # 1% risk per trade
    rr_ratio = 2.0   # 2R take profit
    time_stop_bars = 20

    def init(self):
        print("🌙 Initializing Moon Dev indicators... ✨")

        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        # Stochastic Oscillator (Fast %K, Slow %D)
        self.stoch_k, self.stoch_d = self.I(
            talib.STOCH,
            high, low, close,
            fastk_period=self.stoch_k_period,
            slowk_period=self.stoch_k_smooth,
            slowk_matype=0,
            slowd_period=self.stoch_d_period,
            slowd_matype=0
        )

        # ATR for volatility and stops
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        # Donchian Channel for breakout levels
        self.dc_upper = self.I(talib.MAX, high, timeperiod=self.donchian_period)
        self.dc_lower = self.I(talib.MIN, low, timeperiod=self.donchian_period)

        # Volume MA for confirmation
        self.vol_ma = self.I(talib.SMA, volume, timeperiod=self.vol_ma_period)

        # Harmonic Oscillator - Hilbert Transform based sine wave (HT_SINE)
        # HT_SINE returns (sine, leadsine) each in [-1, 1]
        self.harm_sine, self.harm_leadsine = self.I(talib.HT_SINE, close)
        # Normalize to 0-100 range for easier threshold comparison
        self.harm_osc = self.I(_harm_osc_from_sine, self.harm_sine)

        print("🌙 Indicators ready! Let's moon! 🚀")

    def next(self):
        # Skip if indicators not fully warmed up
        if len(self.data) < self.donchian_period + 5:
            return
        if np.isnan(self.harm_osc[-1]) or np.isnan(self.harm_osc[-2]):
            return
        if np.isnan(self.stoch_k[-1]) or np.isnan(self.stoch_d[-1]):
            return
        if np.isnan(self.atr[-1]) or self.atr[-1] <= 0:
            return
        if np.isnan(self.dc_upper[-1]) or np.isnan(self.dc_lower[-1]):
            return

        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        volume = self.data.Volume[-1]

        # Volume/volatility confirmation
        vol_expansion = volume > self.vol_ma[-1] * 1.1
        atr_expansion = self.atr[-1] > self.atr[-2] if len(self.atr) > 2 else False

        # Harmonic oscillator conditions
        harm_rising = self.harm_osc[-1] > self.harm_osc[-2]
        harm_falling = self.harm_osc[-1] < self.harm_osc[-2]
        harm_oversold = self.harm_osc[-2] < 20
        harm_overbought = self.harm_osc[-2] > 80

        # Stochastic convergence (manual crossover - no backtesting.lib!)
        stoch_bull_cross = self.stoch_k[-1] > self.stoch_d[-1] and self.stoch_k[-2] <= self.stoch_d[-2]
        stoch_bear_cross = self.stoch_k[-1] < self.stoch_d[-1] and self.stoch_k[-2] >= self.stoch_d[-2]
        stoch_bull_zone = self.stoch_k[-1] < 50
        stoch_bear_zone = self.stoch_k[-1] > 50

        # Breakout levels (previous bar's donchian)
        breakout_long = price > self.dc_upper[-2]
        breakout_short = price < self.dc_lower[-2]

        # ---- ENTRY LOGIC ----
        if not self.position:
            # LONG entry
            if (harm_rising and (harm_oversold or self.harm_osc[-1] > self.harm_osc[-2])) \
               and stoch_bull_cross and stoch_bull_zone \
               and breakout_long and (vol_expansion or atr_expansion):

                stop_loss = low - self.atr[-1] * 1.0
                risk = price - stop_loss
                if risk <= 0:
                    return
                take_profit = price + risk * self.rr_ratio

                # Position sizing: risk 1% of equity -> fraction of equity
                equity = self.equity
                risk_amount = equity * self.risk_pct
                position_size = risk_amount / risk
                # Cap fraction to (0, 1)
                if position_size <= 0:
                    return
                if position_size >= 1:
                    position_size = 0.99

                print(f"🌙🚀 MOON DEV LONG SIGNAL! Price={price:.2f} SL={stop_loss:.2f} TP={take_profit:.2f} Size={position_size:.4f}")
                self.buy(size=position_size, sl=stop_loss, tp=take_profit)

            # SHORT entry
            elif (harm_falling and (harm_overbought or self.harm_osc[-1] < self.harm_osc[-2])) \
                 and stoch_bear_cross and stoch_bear_zone \
                 and breakout_short and (vol_expansion or atr_expansion):

                stop_loss = high + self.atr[-1] * 1.0
                risk = stop_loss - price
                if risk <= 0:
                    return
                take_profit = price - risk * self.rr_ratio

                equity = self.equity
                risk_amount = equity * self.risk_pct
                position_size = risk_amount / risk
                if position_size <= 0:
                    return
                if position_size >= 1:
                    position_size = 0.99

                print(f"🌙🔻 MOON DEV SHORT SIGNAL! Price={price:.2f} SL={stop_loss:.2f} TP={take_profit:.2f} Size={position_size:.4f}")
                self.sell(size=position_size, sl=stop_loss, tp=take_profit)

        # ---- TIME STOP ----
        else:
            if self.trades:
                bars_in_trade = len(self.data) - self.trades[-1].entry_bar
                if bars_in_trade >= self.time_stop_bars:
                    print(f"⏰ Moon Dev Time Stop hit after {bars_in_trade} bars! Closing position. 🌙")
                    self.position.close()


# Run backtest
bt = Backtest(
    data,
    HarmonicStochastic,
    cash=1_000_000,
    commission=0.0005,
)

print("🌙✨ Running Moon Dev HarmonicStochastic Backtest... 🚀")
stats = bt.run()
print(stats)
print(stats._strategy)