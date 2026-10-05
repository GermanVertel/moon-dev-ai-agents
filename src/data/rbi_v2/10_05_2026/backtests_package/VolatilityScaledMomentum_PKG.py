import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# ============================================================
# 🌙 MOON DEV'S VOLATILITY-SCALED MOMENTUM STRATEGY 🚀
# ============================================================

print("🌙 Initializing Moon Dev Backtest AI...")
print("✨ Loading cosmic data from the Moon Dev vault...")

data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')

# 🧹 Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# 🗺️ Proper column mapping
data = data.rename(columns={
    'datetime': 'Datetime',
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

data['Datetime'] = pd.to_datetime(data['Datetime'])
data = data.set_index('Datetime')

print(f"✨ Data loaded: {len(data)} rows of cosmic price action 🌙")
print(f"🚀 Date range: {data.index[0]} to {data.index[-1]}")


class VolatilityScaledMomentum(Strategy):
    """
    🌙 Volatility-Scaled Momentum Strategy 🚀
    - 3-day high breakout entry
    - Volume confirmation filter
    - ATR-based volatility scaling
    - 3-day low exit
    """

    # 🎛️ Strategy Parameters
    breakout_period = 3
    volume_sma_period = 20
    volume_threshold = 0.8
    atr_period = 14
    atr_ema_period = 20
    risk_per_trade = 0.01  # 1% of equity per trade
    atr_stop_multiplier = 2.0
    max_position_pct = 0.95  # cap at 95% of equity

    def init(self):
        print("🌙 Initializing Moon Dev indicators...")

        # 📊 3-day high (excluding current bar via shift)
        self.high_3 = self.I(talib.MAX, self.data.High, timeperiod=self.breakout_period)
        self.high_3_prev = self.I(
            lambda x: pd.Series(x).shift(1).values,
            self.high_3,
            name='High3_Prev'
        )

        # 📉 3-day low (excluding current bar via shift)
        self.low_3 = self.I(talib.MIN, self.data.Low, timeperiod=self.breakout_period)
        self.low_3_prev = self.I(
            lambda x: pd.Series(x).shift(1).values,
            self.low_3,
            name='Low3_Prev'
        )

        # 📊 Volume SMA
        self.volume_sma = self.I(talib.SMA, self.data.Volume, timeperiod=self.volume_sma_period)

        # 📊 ATR and ATR EMA
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period)
        self.atr_ema = self.I(talib.EMA, self.atr, timeperiod=self.atr_ema_period)

        print("✨ Moon Dev indicators ready! 🚀")

    def next(self):
        # Skip warmup
        if len(self.data) < max(self.volume_sma_period, self.atr_ema_period, self.breakout_period) + 5:
            return

        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        volume = self.data.Volume[-1]

        prev_high_3 = self.high_3_prev[-1]
        prev_low_3 = self.low_3_prev[-1]
        vol_sma = self.volume_sma[-1]
        atr_ema_val = self.atr_ema[-1]
        atr_val = self.atr[-1]

        # Guard against NaN
        if (np.isnan(prev_high_3) or np.isnan(prev_low_3) or
                np.isnan(vol_sma) or np.isnan(atr_ema_val) or np.isnan(atr_val)):
            return

        # ============================================
        # 🚀 ENTRY LOGIC
        # ============================================
        if not self.position:
            # a) New 3-day high breakout
            breakout = high > prev_high_3
            # b) Volume confirmation
            volume_ok = volume >= self.volume_threshold * vol_sma

            if breakout and volume_ok and atr_ema_val > 0:
                # c) Volatility-scaled position sizing
                equity = self.equity
                risk_dollars = self.risk_per_trade * equity
                # Position size inversely proportional to ATR-EMA
                raw_size = risk_dollars / (atr_ema_val * self.atr_stop_multiplier)

                # 💰 Convert to units of the asset
                units = raw_size / price
                units = int(round(units))

                # 🛡️ Cap position size
                max_units = int(round((self.max_position_pct * equity) / price))
                if units > max_units:
                    units = max_units

                if units > 0:
                    # ATR-based stop loss
                    stop_price = price - (self.atr_stop_multiplier * atr_val)
                    self.buy(size=units, sl=stop_price)
                    print(f"🌙✨ MOON DEV BREAKOUT! 🚀 Entry @ {price:.2f} | "
                          f"Size: {units} | ATR_EMA: {atr_ema_val:.2f} | "
                          f"SL: {stop_price:.2f} | Vol: {volume:.2f} vs SMA: {vol_sma:.2f}")

        # ============================================
        # 🌙 EXIT LOGIC
        # ============================================
        else:
            # Exit on new 3-day low
            if low < prev_low_3:
                self.position.close()
                print(f"🌙💫 MOON DEV EXIT! 🛑 New 3-day low @ {low:.2f} | Close: {price:.2f}")


print("🌙 Launching Moon Dev Backtest... 🚀✨")
bt = Backtest(
    data,
    VolatilityScaledMomentum,
    cash=1_000_000,
    commission=0.002,
    exclusive_orders=True
)

stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙 Backtest complete! Moon Dev out! 🚀✨")