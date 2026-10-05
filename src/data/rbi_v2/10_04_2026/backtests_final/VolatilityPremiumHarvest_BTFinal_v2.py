import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# 🌙 Moon Dev's VolatilityPremiumHarvest Backtest 🚀
# Strategy: Sell premium in high-beta names during calm regimes,
# with a "strength exit" twist to avoid vega-driven drawdowns.

class VolatilityPremiumHarvest(Strategy):
    # --- Strategy Parameters ---
    atr_period = 14
    adx_period = 14
    ema_fast = 20
    ema_slow = 50
    vol_ma_period = 20
    risk_pct = 0.02          # 2% of equity per trade
    atr_stop_mult = 1.5
    profit_target_pct = 0.50  # 50% of premium (approximated)
    strength_vol_mult = 1.5
    size = 0.95               # Fractional position size (0 < size < 1)

    def init(self):
        # 🌙 Calculate all indicators with self.I() wrapper
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close,
                          timeperiod=self.atr_period)
        self.adx = self.I(talib.ADX, self.data.High, self.data.Low, self.data.Close,
                          timeperiod=self.adx_period)
        self.ema_fast_ind = self.I(talib.EMA, self.data.Close, timeperiod=self.ema_fast)
        self.ema_slow_ind = self.I(talib.EMA, self.data.Close, timeperiod=self.ema_slow)
        self.vol_ma = self.I(talib.SMA, self.data.Volume, timeperiod=self.vol_ma_period)

        # 🌙 Track entry state
        self.entry_price_val = None
        self.entry_atr = None
        self.stop_price = None
        self.take_profit_price = None
        self.bars_in_trade = 0

        print("🌙✨ VolatilityPremiumHarvest initialized — indicators ready! 🚀")

    def next(self):
        # Skip if not enough data
        if len(self.data) < max(self.ema_slow, self.vol_ma_period, self.atr_period) + 2:
            return

        price = self.data.Close[-1]
        atr_val = self.atr[-1]
        adx_val = self.adx[-1]
        ema_f = self.ema_fast_ind[-1]
        ema_s = self.ema_slow_ind[-1]
        vol = self.data.Volume[-1]
        vol_ma = self.vol_ma[-1]

        # 🌙 Guard against NaN indicators
        if (np.isnan(atr_val) or np.isnan(adx_val) or np.isnan(ema_f)
                or np.isnan(ema_s) or np.isnan(vol_ma)):
            return

        # 🌙 Regime filter: sideways/mild uptrend
        regime_ok = (adx_val < 25) and (price > ema_s) and (ema_f > ema_s)

        # --- ENTRY LOGIC ---
        if not self.position:
            # Simplified premium richness proxy: ATR% elevated
            atr_pct = atr_val / price if price > 0 else 0
            iv_proxy_ok = atr_pct > 0.008  # ~0.8% ATR as proxy for IVR >= 30

            if regime_ok and iv_proxy_ok:
                # 🌙 Simulate selling a put by going LONG the underlying
                # (short put ≈ bullish exposure with capped upside)
                # Use fractional sizing so backtesting.py accepts it
                self.buy(size=self.size)

                self.entry_price_val = price
                self.entry_atr = atr_val
                self.stop_price = price - (self.atr_stop_mult * atr_val)
                # Approximate 50% premium capture as a small % move
                self.take_profit_price = price + (0.5 * atr_val)
                self.bars_in_trade = 0

                print(f"🌙🚀 ENTRY | Price: {price:.2f} | ATR: {atr_val:.2f} | "
                      f"Stop: {self.stop_price:.2f} | TP: {self.take_profit_price:.2f} | "
                      f"ADX: {adx_val:.1f}")

        # --- EXIT LOGIC ---
        else:
            self.bars_in_trade += 1

            # Update trailing ATR stop upward as position profits
            new_stop = price - (self.atr_stop_mult * atr_val)
            if new_stop > self.stop_price:
                self.stop_price = new_stop
                print(f"🌙📈 Trailing stop raised to {self.stop_price:.2f}")

            # 1. Profit target (50% of premium captured)
            if price >= self.take_profit_price:
                self.position.close()
                print(f"🌙💰 PROFIT TARGET hit at {price:.2f} | Bars: {self.bars_in_trade}")
                return

            # 2. Stop-loss: underlying closes below ATR stop line
            if price < self.stop_price:
                self.position.close()
                print(f"🌙🛑 STOP LOSS triggered at {price:.2f} | Stop: {self.stop_price:.2f}")
                return

            # 3. Strength exit — 2 consecutive higher closes + volume expansion
            if len(self.data) >= 3:
                c1 = self.data.Close[-1]
                c2 = self.data.Close[-2]
                c3 = self.data.Close[-3]
                higher_closes = (c1 > c2) and (c2 > c3)
                vol_expansion = vol > (self.strength_vol_mult * vol_ma)

                if higher_closes and vol_expansion:
                    self.position.close()
                    print(f"🌙⚡ STRENGTH EXIT | 2 higher closes + vol spike "
                          f"({vol:.0f} > {self.strength_vol_mult * vol_ma:.0f}) at {price:.2f}")
                    return

            # 4. Time exit — approximate 21 DTE equivalent (proxy: many bars held)
            if self.bars_in_trade >= 60:
                self.position.close()
                print(f"🌙⏰ TIME EXIT at {price:.2f} | Bars held: {self.bars_in_trade}")
                return


# --- Data Loading & Cleaning ---
data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# 🌙 Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# 🌙 Proper case mapping for backtesting.py
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

# 🌙 Set datetime index
if 'datetime' in data.columns:
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')
elif 'date' in data.columns:
    data['date'] = pd.to_datetime(data['date'])
    data = data.set_index('date')

# 🌙 Ensure numeric dtypes
for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')

data = data.dropna(subset=['Open', 'High', 'Low', 'Close'])

print(f"🌙✨ Data loaded: {len(data)} bars from {data.index[0]} to {data.index[-1]} 🚀")

# --- Run Backtest ---
bt = Backtest(data, VolatilityPremiumHarvest, cash=1_000_000, commission=0.002)
stats = bt.run()
print(stats)
print(stats._strategy)