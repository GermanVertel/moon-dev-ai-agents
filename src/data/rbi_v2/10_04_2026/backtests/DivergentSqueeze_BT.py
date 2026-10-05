import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# ============================================================
# 🌙 MOON DEV BACKTEST - DivergentSqueeze Strategy 🚀
# ============================================================

print("🌙✨ Loading Moon Dev data...")
data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Map to backtesting.py required columns
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
print(f"🌙✨ Data loaded: {len(data)} bars 🚀")


class DivergentSqueeze(Strategy):
    # Strategy parameters
    sma_fast = 50
    sma_slow = 200
    bb_period = 20
    bb_std = 2.0
    vol_avg_period = 20
    rsi_period = 14
    atr_period = 14
    bbw_lookback = 180          # bottom 20% of 6-month range (~180 bars on 15m)
    divergence_lookback = 20    # bars to look for bullish divergence
    risk_pct = 0.01             # 1% account risk
    atr_mult = 2.0

    def init(self):
        print("🌙✨ Initializing Moon Dev indicators...")

        close = pd.Series(self.data.Close, index=range(len(self.data)))
        high = pd.Series(self.data.High, index=range(len(self.data)))
        low = pd.Series(self.data.Low, index=range(len(self.data)))
        volume = pd.Series(self.data.Volume, index=range(len(self.data)))

        # --- Moving Averages ---
        self.sma_fast = self.I(talib.SMA, self.data.Close, timeperiod=self.sma_fast)
        self.sma_slow = self.I(talib.SMA, self.data.Close, timeperiod=self.sma_slow)

        # --- Bollinger Bands ---
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS, self.data.Close,
            timeperiod=self.bb_period, nbdevup=self.bb_std, nbdevdn=self.bb_std
        )

        # --- Bollinger Bandwidth ---
        def bbw_func(upper, middle, lower):
            return (upper - lower) / middle
        self.bbw = self.I(bbw_func, self.bb_upper, self.bb_middle, self.bb_lower)

        # --- Volume Average ---
        self.vol_avg = self.I(talib.SMA, self.data.Volume, timeperiod=self.vol_avg_period)

        # --- RSI ---
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=self.rsi_period)

        # --- ATR ---
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period)

        # --- Track death cross state ---
        self.death_cross_active = False
        self.death_cross_bar = -1
        self.death_cross_vol_ok = False

        print("🌙✨ Indicators ready! 🚀")

    def next(self):
        i = len(self.data) - 1
        if i < self.sma_slow + 5:
            return

        price = self.data.Close[-1]
        vol = self.data.Volume[-1]

        # -------- Detect Death Cross (50 SMA crosses below 200 SMA) --------
        if len(self.data) > self.sma_slow + 1:
            fast_prev = self.sma_fast[-2]
            slow_prev = self.sma_slow[-2]
            fast_now = self.sma_fast[-1]
            slow_now = self.sma_slow[-1]

            if (not np.isnan(fast_prev) and not np.isnan(slow_prev) and
                not np.isnan(fast_now) and not np.isnan(slow_now)):
                if fast_prev >= slow_prev and fast_now < slow_now:
                    # Death cross just happened
                    vol_avg = self.vol_avg[-1]
                    if not np.isnan(vol_avg) and vol < vol_avg:
                        self.death_cross_active = True
                        self.death_cross_bar = i
                        self.death_cross_vol_ok = True
                        print(f"💀🌙 Death Cross detected at bar {i} with LOW volume! Bearish regime active.")

        # -------- Exit Logic --------
        if self.position:
            # Primary exit: close below middle BB
            if not np.isnan(self.bb_middle[-1]) and price < self.bb_middle[-1]:
                print(f"🌙✨ EXIT: Close below middle BB @ {price:.2f} 🚀")
                self.position.close()
                return
            # Secondary exit: touch lower BB
            if not np.isnan(self.bb_lower[-1]) and price <= self.bb_lower[-1]:
                print(f"🌙✨ EXIT: Touched lower BB @ {price:.2f} 🚀")
                self.position.close()
                return
            # Stop-loss handled by SL order
            return

        # -------- Entry Conditions --------
        if not self.death_cross_active or not self.death_cross_vol_ok:
            return

        # Need enough bars since death cross for divergence + squeeze
        if i - self.death_cross_bar < 10:
            return

        # 1. Bearish regime: 50 SMA < 200 SMA
        if np.isnan(self.sma_fast[-1]) or np.isnan(self.sma_slow[-1]):
            return
        if self.sma_fast[-1] >= self.sma_slow[-1]:
            return

        # 2. Bollinger Bandwidth squeeze: bottom 20% of 6-month range
        lookback = min(self.bbw_lookback, i)
        if lookback < 20:
            return
        recent_bbw = np.array(self.bbw[-lookback:])
        recent_bbw = recent_bbw[~np.isnan(recent_bbw)]
        if len(recent_bbw) < 20:
            return
        bbw_threshold = np.percentile(recent_bbw, 20)
        if np.isnan(self.bbw[-1]) or self.bbw[-1] > bbw_threshold:
            return

        # 3. Bullish RSI divergence: price lower low, RSI higher low
        div_look = min(self.divergence_lookback, i)
        if div_look < 10:
            return
        prices = np.array(self.data.Close[-div_look:])
        rsis = np.array(self.rsi[-div_look:])
        if np.any(np.isnan(rsis)):
            return

        # Find lowest price point and its RSI
        price_min_idx = np.argmin(prices)
        price_min = prices[price_min_idx]
        rsi_at_price_min = rsis[price_min_idx]

        # Look for a prior lower-low in price with higher RSI (divergence)
        # Check if there's an earlier bar with lower price but lower RSI... actually
        # bullish divergence: price makes lower low, RSI makes higher low
        # Compare current low to earlier low
        if price_min_idx < 5:
            return
        earlier_prices = prices[:price_min_idx]
        earlier_rsis = rsis[:price_min_idx]
        if len(earlier_prices) < 3:
            return
        earlier_min_idx = np.argmin(earlier_prices)
        earlier_price_min = earlier_prices[earlier_min_idx]
        earlier_rsi = earlier_rsis[earlier_min_idx]

        # Bearish divergence condition met if:
        # current price_min < earlier_price_min (lower low), but RSI at current > RSI at earlier
        divergence_ok = (price_min < earlier_price_min) and (rsi_at_price_min > earlier_rsi)

        if not divergence_ok:
            return

        # 4. Entry trigger: close above upper BB with above-average volume
        if np.isnan(self.bb_upper[-1]) or np.isnan(self.vol_avg[-1]):
            return
        if price <= self.bb_upper[-1]:
            return
        if vol <= self.vol_avg[-1]:
            return

        # -------- Position Sizing (1% risk, ATR stop) --------
        atr_val = self.atr[-1]
        if np.isnan(atr_val) or atr_val <= 0:
            return

        stop_price = price - self.atr_mult * atr_val
        risk_per_unit = price - stop_price
        if risk_per_unit <= 0:
            return

        equity = self.equity
        risk_amount = equity * self.risk_pct
        position_size = int(round(risk_amount / risk_per_unit))

        if position_size <= 0:
            return

        print(f"🌙🚀 ENTRY SIGNAL! DivergentSqueeze LONG @ {price:.2f} | Size: {position_size} | Stop: {stop_price:.2f} | ATR: {atr_val:.2f}")

        self.buy(size=position_size, sl=stop_price)

        # Reset death cross state after entry to avoid re-triggering
        self.death_cross_active = False


print("🌙✨ Running Moon Dev Backtest... 🚀")
bt = Backtest(data, DivergentSqueeze, cash=1_000_000, commission=0.002)
stats = bt.run()
print(stats)
print(stats._strategy)