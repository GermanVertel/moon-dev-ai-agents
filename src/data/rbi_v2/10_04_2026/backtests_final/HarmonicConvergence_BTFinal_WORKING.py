import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# ==============================================================================
# 🌙 MOON DEV'S HARMONIC CONVERGENCE STRATEGY 🌙
# ==============================================================================

class HarmonicConvergence(Strategy):
    # Strategy parameters
    ho_period = 14
    ho_smooth = 3
    ho_signal = 3
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    atr_period = 14
    atr_mult = 2.0
    risk_pct = 0.02  # 2% risk per trade

    def init(self):
        print("🌙✨ Initializing HarmonicConvergence indicators...")

        # --- Harmonic Oscillator (sine-wave based smoothed momentum) ---
        rsi_raw = self.I(talib.RSI, self.data.Close, timeperiod=self.ho_period, name="RSI_Raw")

        def harmonic_osc(x):
            x = np.asarray(x, dtype=float)
            norm = (x - 50.0) / 50.0
            norm = np.clip(norm, -1.0, 1.0)
            return np.sin(norm * (np.pi / 2.0)) * 100.0

        self.ho = self.I(harmonic_osc, rsi_raw, name="HO")
        self.ho_sig = self.I(talib.SMA, self.ho, timeperiod=self.ho_signal, name="HO_Signal")

        # --- MACD ---
        self.macd, self.macd_sig, self.macd_hist = self.I(
            talib.MACD, self.data.Close,
            fastperiod=self.macd_fast,
            slowperiod=self.macd_slow,
            signalperiod=self.macd_signal,
            name="MACD"
        )

        # --- ATR for stops ---
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close,
                          timeperiod=self.atr_period, name="ATR")

        # --- Swing high/low for stop placement ---
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=10, name="SwingLow")
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=10, name="SwingHigh")

        print("🚀 Indicators loaded. Let's ride the harmonic waves!")

    def next(self):
        price = self.data.Close[-1]

        # Skip if insufficient data
        if len(self.data) < 30:
            return

        ho = self.ho[-1]
        ho_prev = self.ho[-2]
        ho_sig = self.ho_sig[-1]
        ho_sig_prev = self.ho_sig[-2]

        macd = self.macd[-1]
        macd_sig = self.macd_sig[-1]
        macd_prev = self.macd[-2]
        macd_sig_prev = self.macd_sig[-2]

        atr = self.atr[-1]
        swing_low = self.swing_low[-1]
        swing_high = self.swing_high[-1]

        # Guard against NaNs
        if (np.isnan(ho) or np.isnan(ho_prev) or np.isnan(ho_sig) or np.isnan(ho_sig_prev)
                or np.isnan(macd) or np.isnan(macd_sig) or np.isnan(macd_prev)
                or np.isnan(macd_sig_prev) or np.isnan(atr)
                or np.isnan(swing_low) or np.isnan(swing_high)):
            return

        # --- Trend filter ---
        bullish_trend = macd > macd_sig and macd > 0
        bearish_trend = macd < macd_sig and macd < 0

        # --- HO crosses (manual crossover detection) ---
        ho_bull_cross = ho_prev < ho_sig_prev and ho > ho_sig
        ho_bear_cross = ho_prev > ho_sig_prev and ho < ho_sig

        # --- Position management ---
        if self.position:
            if self.position.is_long:
                macd_bear_cross = macd_prev > macd_sig_prev and macd < macd_sig
                if ho > 80 and ho < ho_prev:
                    print(f"🌙💫 LONG EXIT: HO overbought reversal at {price:.2f}")
                    self.position.close()
                elif macd_bear_cross:
                    print(f"🌙⚠️ LONG EXIT: MACD bearish cross at {price:.2f}")
                    self.position.close()

            elif self.position.is_short:
                macd_bull_cross = macd_prev < macd_sig_prev and macd > macd_sig
                if ho < -80 and ho > ho_prev:
                    print(f"🌙💫 SHORT EXIT: HO oversold reversal at {price:.2f}")
                    self.position.close()
                elif macd_bull_cross:
                    print(f"🌙⚠️ SHORT EXIT: MACD bullish cross at {price:.2f}")
                    self.position.close()
            return

        # --- Long Entry ---
        if bullish_trend and ho_bull_cross and ho < 0:
            stop_price = min(swing_low, price - atr * self.atr_mult)
            risk_per_unit = price - stop_price
            if risk_per_unit > 0:
                risk_amount = self.equity * self.risk_pct
                # Use fractional sizing to avoid cash/unit issues
                pos_size = risk_amount / risk_per_unit
                fraction = pos_size * price / self.equity
                if 0 < fraction < 1:
                    print(f"🌙🚀 LONG ENTRY | Price: {price:.2f} | HO: {ho:.2f} | MACD: {macd:.2f} | Frac: {fraction:.4f}")
                    self.buy(size=fraction, sl=stop_price)
                else:
                    # Fallback to fixed fraction
                    print(f"🌙🚀 LONG ENTRY (fallback) | Price: {price:.2f} | HO: {ho:.2f} | MACD: {macd:.2f}")
                    self.buy(size=0.5, sl=stop_price)

        # --- Short Entry ---
        elif bearish_trend and ho_bear_cross and ho > 0:
            stop_price = max(swing_high, price + atr * self.atr_mult)
            risk_per_unit = stop_price - price
            if risk_per_unit > 0:
                risk_amount = self.equity * self.risk_pct
                pos_size = risk_amount / risk_per_unit
                fraction = pos_size * price / self.equity
                if 0 < fraction < 1:
                    print(f"🌙🔻 SHORT ENTRY | Price: {price:.2f} | HO: {ho:.2f} | MACD: {macd:.2f} | Frac: {fraction:.4f}")
                    self.sell(size=fraction, sl=stop_price)
                else:
                    print(f"🌙🔻 SHORT ENTRY (fallback) | Price: {price:.2f} | HO: {ho:.2f} | MACD: {macd:.2f}")
                    self.sell(size=0.5, sl=stop_price)


# ==============================================================================
# 🌙 DATA LOADING & BACKTEST EXECUTION 🌙
# ==============================================================================

print("🌙 Loading Moon Dev data...")
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
if 'datetime' in data.columns:
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')

data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
print(f"🌙 Data loaded: {len(data)} bars ✨")

# Ensure numeric types
for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    data[col] = pd.to_numeric(data[col], errors='coerce')
data = data.dropna()

bt = Backtest(data, HarmonicConvergence, cash=1_000_000, commission=0.001, exclusive_orders=True)
stats = bt.run()
print(stats)
print(stats._strategy)