import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# ============================================================
# 🌙 MOON DEV - GoldenSurge Backtest 🌙
# ============================================================

print("🌙✨ Moon Dev is loading the cosmic data... 🚀")

data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Map to required format
data = data.rename(columns={
    'datetime': 'Datetime',
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

if 'Datetime' in data.columns:
    data['Datetime'] = pd.to_datetime(data['Datetime'])
    data = data.set_index('Datetime')

data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

print(f"🌙✨ Data loaded! Rows: {len(data)} 🚀")
print(f"🌙 Columns: {list(data.columns)}")


class GoldenSurge(Strategy):
    # Strategy parameters
    ema_fast_period = 50
    ema_slow_period = 200
    adx_period = 14
    adx_threshold = 25
    adx_exit_threshold = 20
    atr_period = 14
    atr_stop_mult = 2.0
    reward_risk = 3.0
    swing_lookback = 20
    risk_pct = 0.02  # 2% risk per trade

    def init(self):
        print("🌙✨ Initializing GoldenSurge indicators... 🚀")

        close = self.data.Close
        high = self.data.High
        low = self.data.Low

        # EMAs
        self.ema_fast = self.I(talib.EMA, close, timeperiod=self.ema_fast_period, name='EMA50')
        self.ema_slow = self.I(talib.EMA, close, timeperiod=self.ema_slow_period, name='EMA200')

        # ADX + DI
        self.adx = self.I(talib.ADX, high, low, close, timeperiod=self.adx_period, name='ADX')
        self.plus_di = self.I(talib.PLUS_DI, high, low, close, timeperiod=self.adx_period, name='+DI')
        self.minus_di = self.I(talib.MINUS_DI, high, low, close, timeperiod=self.adx_period, name='-DI')

        # ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period, name='ATR')

        # Swing high (recent resistance) - shift by 1 to avoid lookahead on current bar
        self.swing_high = self.I(talib.MAX, high, timeperiod=self.swing_lookback, name='SwingHigh')

        # Track entry state
        self.entry_price = None
        self.stop_price = None
        self.target_price = None
        self.breakeven_moved = False

        print("🌙✨ Indicators ready! Let the golden surge begin! 🌟")

    def next(self):
        price = self.data.Close[-1]

        # Skip warmup
        if len(self.data) < self.ema_slow_period + 5:
            return

        ema_f = self.ema_fast[-1]
        ema_s = self.ema_slow[-1]
        ema_f_prev = self.ema_fast[-2]
        ema_s_prev = self.ema_slow[-2]
        adx_val = self.adx[-1]
        adx_prev = self.adx[-2]
        atr_val = self.atr[-1]
        # Use previous bar's swing high to avoid comparing price to current bar's high (lookahead)
        swing = self.swing_high[-2]

        # Guard against NaN indicator values during warmup
        if (np.isnan(ema_f) or np.isnan(ema_s) or np.isnan(adx_val) or
                np.isnan(adx_prev) or np.isnan(atr_val) or np.isnan(swing)):
            return

        # Trend condition: EMA50 above EMA200
        golden_cross = ema_f > ema_s
        # Recent cross (was below, now above)
        recently_crossed = (ema_f_prev <= ema_s_prev) and (ema_f > ema_s)

        # ADX momentum
        adx_strong = adx_val > self.adx_threshold
        adx_rising = adx_val > adx_prev

        # Breakout above swing high
        breakout = price > swing

        # ---------------- ENTRY ----------------
        if not self.position:
            if golden_cross and adx_strong and adx_rising and breakout:
                # Position sizing based on risk
                stop_distance = self.atr_stop_mult * atr_val
                if stop_distance <= 0 or np.isnan(stop_distance):
                    return

                # Risk amount in $ terms
                risk_amount = self.equity * self.risk_pct
                # Units = risk_amount / stop_distance (per-unit stop loss)
                position_size = risk_amount / stop_distance
                position_size = int(round(position_size))

                if position_size < 1:
                    position_size = 1

                # Cap by available cash (backtesting uses whole units)
                max_affordable = int(self.equity // price)
                if max_affordable < 1:
                    return
                if position_size > max_affordable:
                    position_size = max_affordable

                # Ensure size is a valid positive integer
                if position_size < 1:
                    return

                self.entry_price = price
                self.stop_price = price - stop_distance
                self.target_price = price + (self.reward_risk * stop_distance)
                self.breakeven_moved = False

                print(f"🌙🚀 GOLDEN SURGE ENTRY! Price: {price:.2f} | Size: {position_size} | Stop: {self.stop_price:.2f} | Target: {self.target_price:.2f} | ADX: {adx_val:.2f}")
                self.buy(size=position_size)

        # ---------------- EXIT ----------------
        else:
            # Stop loss
            if self.data.Low[-1] <= self.stop_price:
                print(f"🌙🛑 STOP LOSS HIT at {self.stop_price:.2f} 🌙")
                self.position.close()
                self.entry_price = None
                return

            # Take profit
            if self.data.High[-1] >= self.target_price:
                print(f"🌙💰 TAKE PROFIT HIT at {self.target_price:.2f} 🚀✨")
                self.position.close()
                self.entry_price = None
                return

            # Move stop to breakeven at 1:1 R:R
            if self.entry_price is not None and not self.breakeven_moved:
                risk = self.entry_price - self.stop_price
                if self.data.High[-1] >= self.entry_price + risk:
                    self.stop_price = self.entry_price
                    self.breakeven_moved = True
                    print(f"🌙🔒 Stop moved to BREAKEVEN at {self.stop_price:.2f} 🌙")

            # Trend invalidation exits
            if ema_f < ema_s:
                print(f"🌙⚠️ EMA CROSS INVALIDATED - exiting long 🌙")
                self.position.close()
                self.entry_price = None
                return

            if adx_val < self.adx_exit_threshold:
                print(f"🌙⚠️ ADX WEAK ({adx_val:.2f}) - exiting long 🌙")
                self.position.close()
                self.entry_price = None
                return


print("🌙✨ Running GoldenSurge backtest... 🚀")
bt = Backtest(data, GoldenSurge, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙✨ GoldenSurge backtest complete! To the moon! 🚀🌟")