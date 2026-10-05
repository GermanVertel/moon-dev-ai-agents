import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

print("🌙 Moon Dev Backtest AI initializing... ✨")
print("🚀 Loading SqueezeVolatility strategy...")

# Load data
data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
print(f"📂 Loading data from: {data_path}")
data = pd.read_csv(data_path)

# Clean column names
data.columns = data.columns.str.strip().str.lower()
print("🧹 Cleaning column names...")

# Drop unnamed columns
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
print("🗑️ Dropped unnamed columns")

# Rename columns properly
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

# Set datetime as index
data = data.set_index(pd.to_datetime(data['datetime']))
data = data.drop(columns=['datetime'])
print(f"✅ Data loaded: {len(data)} rows")
print(f"📊 Date range: {data.index[0]} to {data.index[-1]}")

# Ensure numeric dtypes for talib
for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    data[col] = pd.to_numeric(data[col], errors='coerce').astype(np.float64)


class SqueezeVolatility(Strategy):
    # Strategy parameters
    bb_period = 20
    bb_std = 2.0
    squeeze_lookback = 126  # 6-month rolling window
    vol_sma_period = 20
    vol_multiplier = 1.5
    breakout_buffer = 0.02  # 2% buffer beyond bands
    kc_period = 20
    kc_atr_period = 10
    kc_multiplier = 2.0
    risk_pct = 0.02  # 2% risk per trade
    time_stop_bars = 10
    min_1r_profit = True

    def init(self):
        print("🌙 Initializing indicators... ✨")

        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        # Bollinger Bands
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS, close, timeperiod=self.bb_period,
            nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0,
            name=['BB_Upper', 'BB_Middle', 'BB_Lower']
        )

        # Bollinger Band Width
        self.bb_width = self.I(
            lambda u, m, l: (u - l) / m,
            self.bb_upper, self.bb_middle, self.bb_lower,
            name='BB_Width'
        )

        # 6-month rolling minimum of BB Width
        self.bb_width_min = self.I(
            lambda w: pd.Series(w).rolling(self.squeeze_lookback).min().values,
            self.bb_width,
            name='BB_Width_Min'
        )

        # Volume SMA - cast to float64
        volume_arr = np.asarray(volume, dtype=np.float64)
        self.vol_sma = self.I(talib.SMA, volume_arr, timeperiod=self.vol_sma_period, name='Vol_SMA')

        # Keltner Channels - EMA + ATR
        self.kc_ema = self.I(talib.EMA, close, timeperiod=self.kc_period, name='KC_EMA')
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.kc_atr_period, name='ATR')

        self.kc_upper = self.I(
            lambda e, a: e + self.kc_multiplier * a,
            self.kc_ema, self.atr,
            name='KC_Upper'
        )
        self.kc_lower = self.I(
            lambda e, a: e - self.kc_multiplier * a,
            self.kc_ema, self.atr,
            name='KC_Lower'
        )

        # Track trade state
        self.entry_bar = None
        self.entry_price = None
        self.initial_stop = None

        print("✅ All indicators initialized! 🚀")

    def next(self):
        # Skip if indicators not ready
        if len(self.data) < self.squeeze_lookback + 5:
            return

        if np.isnan(self.bb_width_min[-1]) or np.isnan(self.vol_sma[-1]):
            return
        if np.isnan(self.kc_upper[-1]) or np.isnan(self.kc_lower[-1]):
            return

        price = self.data.Close[-1]
        upper = self.bb_upper[-1]
        lower = self.bb_lower[-1]
        middle = self.bb_middle[-1]
        bw = self.bb_width[-1]
        bw_min = self.bb_width_min[-1]
        vol = self.data.Volume[-1]
        vol_avg = self.vol_sma[-1]
        kc_up = self.kc_upper[-1]
        kc_lo = self.kc_lower[-1]
        atr = self.atr[-1]

        # Squeeze condition: current BB width at or below 6-month low
        squeeze_active = bw <= bw_min * 1.001

        # Volume confirmation
        volume_confirmed = vol > vol_avg * self.vol_multiplier

        # Breakout confirmation
        long_breakout = price >= upper * (1 + self.breakout_buffer)
        short_breakout = price <= lower * (1 - self.breakout_buffer)

        # ---- EXIT LOGIC (check first) ----
        if self.position:
            if self.position.is_long:
                # Exit when close below lower Keltner Channel
                if price < kc_lo:
                    print(f"🌙 EXIT LONG @ {price:.2f} | KC_Lower={kc_lo:.2f} | 💰 PnL: {self.position.pl:.2f}")
                    self.position.close()
                    self.entry_bar = None
                    return
                # Time stop: exit if no 1R profit within N bars
                if self.entry_bar is not None and self.initial_stop is not None:
                    bars_held = len(self.data) - self.entry_bar
                    if bars_held >= self.time_stop_bars:
                        r_distance = self.entry_price - self.initial_stop
                        if r_distance > 0 and (price - self.entry_price) < r_distance:
                            print(f"⏰ TIME STOP LONG @ {price:.2f} | Bars held: {bars_held}")
                            self.position.close()
                            self.entry_bar = None
                            return

            elif self.position.is_short:
                # Exit when close above upper Keltner Channel
                if price > kc_up:
                    print(f"🌙 EXIT SHORT @ {price:.2f} | KC_Upper={kc_up:.2f} | 💰 PnL: {self.position.pl:.2f}")
                    self.position.close()
                    self.entry_bar = None
                    return
                if self.entry_bar is not None and self.initial_stop is not None:
                    bars_held = len(self.data) - self.entry_bar
                    if bars_held >= self.time_stop_bars:
                        r_distance = self.initial_stop - self.entry_price
                        if r_distance > 0 and (self.entry_price - price) < r_distance:
                            print(f"⏰ TIME STOP SHORT @ {price:.2f} | Bars held: {bars_held}")
                            self.position.close()
                            self.entry_bar = None
                            return
            return

        # ---- ENTRY LOGIC ----
        if not squeeze_active:
            return

        if not volume_confirmed:
            return

        if long_breakout:
            # Initial stop: opposite KC or middle BB, whichever is closer
            stop_kc = kc_lo
            stop_mb = middle
            initial_stop = max(stop_kc, stop_mb)  # closer to price for long
            risk_per_unit = price - initial_stop

            if risk_per_unit <= 0:
                return

            # Position sizing: risk fixed % of equity
            risk_amount = self.equity * self.risk_pct
            position_size = int(round(risk_amount / risk_per_unit))
            if position_size < 1:
                position_size = 1

            print(f"🚀 LONG SIGNAL @ {price:.2f} | BB_Upper={upper:.2f} | Vol={vol:.2f} vs Avg={vol_avg:.2f}")
            print(f"   💎 Squeeze active! BW={bw:.6f} vs Min={bw_min:.6f}")
            print(f"   🛡️ Stop={initial_stop:.2f} | Size={position_size} | Risk/unit={risk_per_unit:.2f}")

            self.buy(size=position_size)
            self.entry_bar = len(self.data)
            self.entry_price = price
            self.initial_stop = initial_stop

        elif short_breakout:
            stop_kc = kc_up
            stop_mb = middle
            initial_stop = min(stop_kc, stop_mb)  # closer to price for short
            risk_per_unit = initial_stop - price

            if risk_per_unit <= 0:
                return

            risk_amount = self.equity * self.risk_pct
            position_size = int(round(risk_amount / risk_per_unit))
            if position_size < 1:
                position_size = 1

            print(f"🔻 SHORT SIGNAL @ {price:.2f} | BB_Lower={lower:.2f} | Vol={vol:.2f} vs Avg={vol_avg:.2f}")
            print(f"   💎 Squeeze active! BW={bw:.6f} vs Min={bw_min:.6f}")
            print(f"   🛡️ Stop={initial_stop:.2f} | Size={position_size} | Risk/unit={risk_per_unit:.2f}")

            self.sell(size=position_size)
            self.entry_bar = len(self.data)
            self.entry_price = price
            self.initial_stop = initial_stop


print("🌙 Starting backtest... ✨")
bt = Backtest(
    data, SqueezeVolatility,
    cash=1_000_000,
    commission=0.002
)

stats = bt.run()
print("✨ Backtest complete! 🌙")
print(stats)
print(stats._strategy)