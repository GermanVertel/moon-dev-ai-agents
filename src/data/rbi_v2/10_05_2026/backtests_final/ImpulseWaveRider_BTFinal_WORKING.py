import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# ============================================================
# 🌙 Moon Dev's ImpulseWaveRider Backtest 🌙
# ============================================================

print("🌙✨ Moon Dev is warming up the ImpulseWaveRider engine... 🚀")

data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Map to proper case
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

# Ensure numeric dtypes (talib requires double/float64)
for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    data[col] = pd.to_numeric(data[col], errors='coerce').astype('float64')

data = data.dropna()

print(f"🌙✨ Data loaded: {len(data)} bars 🚀")


class ImpulseWaveRider(Strategy):
    # --- Parameters ---
    ema_fast_period = 20
    ema_slow_period = 50
    rsi_period = 14
    adx_period = 14
    atr_period = 14
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    swing_lookback = 20

    risk_pct = 0.02          # 2% risk per trade
    atr_stop_mult = 2.0      # 2x ATR stop
    reward_ratio = 2.5       # 2.5R target
    adx_threshold = 25
    rsi_long_threshold = 60
    rsi_short_threshold = 40

    def init(self):
        print("🌙✨ Initializing ImpulseWaveRider indicators... 🚀")
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        # Convert to float64 numpy arrays to satisfy talib
        close_arr = np.asarray(close, dtype=np.float64)
        high_arr = np.asarray(high, dtype=np.float64)
        low_arr = np.asarray(low, dtype=np.float64)
        volume_arr = np.asarray(volume, dtype=np.float64)

        # EMAs
        self.ema_fast = self.I(talib.EMA, close_arr, timeperiod=self.ema_fast_period)
        self.ema_slow = self.I(talib.EMA, close_arr, timeperiod=self.ema_slow_period)

        # RSI
        self.rsi = self.I(talib.RSI, close_arr, timeperiod=self.rsi_period)

        # ADX
        self.adx = self.I(talib.ADX, high_arr, low_arr, close_arr, timeperiod=self.adx_period)

        # ATR
        self.atr = self.I(talib.ATR, high_arr, low_arr, close_arr, timeperiod=self.atr_period)

        # MACD
        self.macd, self.macd_signal, self.macd_hist = self.I(
            talib.MACD, close_arr,
            fastperiod=self.macd_fast,
            slowperiod=self.macd_slow,
            signalperiod=self.macd_signal
        )

        # Swing highs/lows for wave structure
        self.swing_high = self.I(talib.MAX, high_arr, timeperiod=self.swing_lookback)
        self.swing_low = self.I(talib.MIN, low_arr, timeperiod=self.swing_lookback)

        # Volume SMA for confirmation
        self.vol_sma = self.I(talib.SMA, volume_arr, timeperiod=20)

        print("🌙✨ Indicators ready! Let the wave riding begin! 🌊🚀")

    def next(self):
        price = self.data.Close[-1]

        # Skip if indicators not ready
        if (np.isnan(self.ema_slow[-1]) or np.isnan(self.adx[-1]) or
                np.isnan(self.atr[-1]) or np.isnan(self.rsi[-1]) or
                np.isnan(self.macd[-1]) or np.isnan(self.swing_high[-1]) or
                np.isnan(self.swing_low[-1])):
            return

        ema_fast = self.ema_fast[-1]
        ema_slow = self.ema_slow[-1]
        rsi = self.rsi[-1]
        adx = self.adx[-1]
        atr = self.atr[-1]
        macd = self.macd[-1]
        macd_sig = self.macd_signal[-1]
        macd_hist = self.macd_hist[-1]
        macd_hist_prev = self.macd_hist[-2] if len(self.macd_hist) > 1 else 0
        swing_high = self.swing_high[-1]
        swing_low = self.swing_low[-1]
        vol = self.data.Volume[-1]
        vol_sma = self.vol_sma[-1]

        # ============================================================
        # 🌊 WAVE STRUCTURE / TREND CONFIRMATION
        # ============================================================
        bull_trend = ema_fast > ema_slow
        bear_trend = ema_fast < ema_slow

        # Momentum alignment
        macd_bull = macd > macd_sig and macd_hist > macd_hist_prev
        macd_bear = macd < macd_sig and macd_hist < macd_hist_prev

        # Trend strength
        strong_trend = adx > self.adx_threshold

        # Volume confirmation
        vol_confirm = vol > vol_sma

        # Breakout confirmation (impulse wave continuation)
        breakout_long = price > swing_high * 0.999
        breakout_short = price < swing_low * 1.001

        # ============================================================
        # 🌙 ENTRY LOGIC
        # ============================================================
        if not self.position:
            # LONG: bullish impulse wave + momentum + trend strength
            if (bull_trend and strong_trend and rsi > self.rsi_long_threshold and
                    macd_bull and vol_confirm):
                stop_price = price - (atr * self.atr_stop_mult)
                risk = price - stop_price
                if risk <= 0:
                    return
                target_price = price + (risk * self.reward_ratio)

                # Position sizing: 2% risk of equity -> fraction of equity
                risk_amount = self.equity * self.risk_pct
                position_size = int(round(risk_amount / risk))
                if position_size < 1:
                    position_size = 1

                # Convert to fraction of equity for safety
                size_frac = min(0.99, max(0.01, (position_size * price) / self.equity))

                print(f"🌙🚀 LONG IMPULSE DETECTED! Price={price:.2f} "
                      f"RSI={rsi:.1f} ADX={adx:.1f} MACD_hist={macd_hist:.4f} "
                      f"Stop={stop_price:.2f} Target={target_price:.2f} Size={position_size}")

                self.buy(size=size_frac, sl=stop_price, tp=target_price)

            # SHORT: bearish impulse wave + momentum + trend strength
            elif (bear_trend and strong_trend and rsi < self.rsi_short_threshold and
                  macd_bear and vol_confirm):
                stop_price = price + (atr * self.atr_stop_mult)
                risk = stop_price - price
                if risk <= 0:
                    return
                target_price = price - (risk * self.reward_ratio)

                risk_amount = self.equity * self.risk_pct
                position_size = int(round(risk_amount / risk))
                if position_size < 1:
                    position_size = 1

                size_frac = min(0.99, max(0.01, (position_size * price) / self.equity))

                print(f"🌙🔻 SHORT IMPULSE DETECTED! Price={price:.2f} "
                      f"RSI={rsi:.1f} ADX={adx:.1f} MACD_hist={macd_hist:.4f} "
                      f"Stop={stop_price:.2f} Target={target_price:.2f} Size={position_size}")

                self.sell(size=size_frac, sl=stop_price, tp=target_price)

        # ============================================================
        # 🌊 EXIT / TRAILING LOGIC (wave invalidation)
        # ============================================================
        else:
            if self.position.is_long:
                # Exit long if trend flips or wave invalidated (break below swing low)
                if bear_trend or price < swing_low:
                    print(f"🌙⚠️ LONG WAVE INVALIDATED — exiting at {price:.2f}")
                    self.position.close()
                # Trailing stop via EMA20
                elif price < ema_fast and self.position.pl > 0:
                    print(f"🌙✨ Trailing EMA exit on long at {price:.2f}")
                    self.position.close()

            elif self.position.is_short:
                if bull_trend or price > swing_high:
                    print(f"🌙⚠️ SHORT WAVE INVALIDATED — exiting at {price:.2f}")
                    self.position.close()
                elif price > ema_fast and self.position.pl > 0:
                    print(f"🌙✨ Trailing EMA exit on short at {price:.2f}")
                    self.position.close()


# ============================================================
# 🚀 RUN BACKTEST
# ============================================================
bt = Backtest(
    data,
    ImpulseWaveRider,
    cash=1_000_000,
    commission=0.001,
)

print("🌙✨ Running ImpulseWaveRider backtest... 🚀🌊")
stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙✨ Moon Dev's ImpulseWaveRider backtest complete! 🚀🌊")