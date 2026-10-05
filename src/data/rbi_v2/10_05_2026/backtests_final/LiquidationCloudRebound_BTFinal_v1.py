import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# 🌙 Moon Dev's LiquidationCloudRebound Backtest 🚀

DATA_PATH = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"

print("🌙✨ Loading Moon Dev data from the lunar vault...")
data = pd.read_csv(DATA_PATH)
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
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
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
print(f"🌙 Data loaded: {len(data)} rows of lunar price action 🚀")


def ichimoku_components(high, low, close):
    """Compute Ichimoku Cloud components."""
    tenkan = (talib.MAX(high, timeperiod=9) + talib.MIN(low, timeperiod=9)) / 2
    kijun = (talib.MAX(high, timeperiod=26) + talib.MIN(low, timeperiod=26)) / 2
    senkou_a = (tenkan + kijun) / 2
    senkou_b = (talib.MAX(high, timeperiod=52) + talib.MIN(low, timeperiod=52)) / 2
    return tenkan, kijun, senkou_a, senkou_b


class LiquidationCloudRebound(Strategy):
    # Strategy parameters
    liq_lookback = 60 * 24 * 4  # 60 days in 15m bars (approx)
    liq_percentile = 90
    risk_pct = 0.015
    atr_mult = 1.5
    time_stop_bars = 72 * 4  # 72 hours in 15m bars

    def init(self):
        print("🌙 Initializing Moon Dev indicators...")
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        # Ichimoku
        tenkan, kijun, senkou_a, senkou_b = self.I(
            ichimoku_components, high, low, close,
            name=['Tenkan', 'Kijun', 'SenkouA', 'SenkouB']
        )
        self.tenkan = tenkan
        self.kijun = kijun
        self.senkou_a = senkou_a
        self.senkou_b = senkou_b

        # ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=14, name='ATR')

        # RSI
        self.rsi = self.I(talib.RSI, close, timeperiod=14, name='RSI')

        # Liquidation proxy: use volume spikes as proxy for liquidation events
        # Rolling 90th percentile of volume over lookback window
        vol_series = pd.Series(np.asarray(volume, dtype=float))
        window_size = min(self.liq_lookback, max(len(vol_series) - 1, 1))
        liq_threshold = vol_series.rolling(window=window_size).quantile(
            self.liq_percentile / 100.0
        )
        self.liq_threshold = self.I(lambda: liq_threshold.values, name='LiqThreshold')

        # Chikou reference: close 26 bars ago
        self.close_26 = self.I(lambda: pd.Series(np.asarray(close, dtype=float)).shift(26).values, name='Close26')

        print("🌙✨ Indicators initialized! Ready to hunt lunar rebounds 🚀")

    def next(self):
        price = self.data.Close[-1]

        # Skip warmup
        if len(self.data) < 60:
            return

        # Gather indicator values
        tenkan = self.tenkan[-1]
        kijun = self.kijun[-1]
        senkou_a = self.senkou_a[-1]
        senkou_b = self.senkou_b[-1]
        atr = self.atr[-1]
        rsi = self.rsi[-1]
        liq_thresh = self.liq_threshold[-1]
        vol = self.data.Volume[-1]
        close_26 = self.close_26[-1]

        if any(np.isnan([tenkan, kijun, senkou_a, senkou_b, atr, liq_thresh])):
            return

        cloud_top = max(senkou_a, senkou_b)
        cloud_bottom = min(senkou_a, senkou_b)

        # === ENTRY LOGIC ===
        if not self.position:
            # 1. Liquidation shock filter (volume proxy)
            liq_shock = vol > liq_thresh

            # 2. Price above cloud
            above_cloud = price > cloud_top

            # 3. Tenkan > Kijun
            tk_bull = tenkan > kijun

            # 4. Chikou bullish
            chikou_bull = (not np.isnan(close_26)) and (price > close_26)

            # 5. RSI confluence (optional)
            rsi_ok = rsi > 40

            # Was price recently inside/below cloud?
            recent_below = False
            lookback = min(10, len(self.data) - 1)
            for i in range(1, lookback + 1):
                if self.data.Close[-i] < cloud_top:
                    recent_below = True
                    break

            if liq_shock and above_cloud and tk_bull and chikou_bull and rsi_ok and recent_below:
                # Stop loss: ATR-based fallback
                stop = price - (self.atr_mult * atr)
                risk_per_unit = price - stop
                if risk_per_unit <= 0:
                    return

                risk_amount = self.equity * self.risk_pct
                size = int(round(risk_amount / risk_per_unit))
                if size < 1:
                    size = 1

                print(f"🌙🚀 MOON DEV ENTRY! Price={price:.2f} | CloudTop={cloud_top:.2f} | "
                      f"TK={tenkan:.2f}/{kijun:.2f} | RSI={rsi:.1f} | Size={size}")
                self.buy(size=size)
                self.stop_price = stop
                self.entry_bar = len(self.data)
                self.target_price = cloud_top * 1.02  # cloud resistance target

        # === EXIT LOGIC ===
        else:
            entry_price = self.trades[-1].entry_price
            bars_held = len(self.data) - self.entry_bar

            # Exit 1: Price closes back inside cloud
            if price < cloud_bottom:
                print(f"🌙⚠️ EXIT: Price fell back inside cloud at {price:.2f}")
                self.position.close()
                return

            # Exit 2: Trailing stop below Kijun after 2x ATR profit
            if price > entry_price + 2 * atr:
                if price < kijun:
                    print(f"🌙💰 EXIT: Trailing stop below Kijun at {price:.2f}")
                    self.position.close()
                    return

            # Exit 3: ATR stop hit
            if price < self.stop_price:
                print(f"🌙🛑 EXIT: ATR stop hit at {price:.2f}")
                self.position.close()
                return

            # Exit 4: Time stop
            if bars_held > self.time_stop_bars:
                print(f"🌙⏰ EXIT: Time stop after {bars_held} bars at {price:.2f}")
                self.position.close()
                return

            # Exit 5: Target reached (cloud resistance)
            if price >= self.target_price:
                print(f"🌙🎯 EXIT: Cloud resistance target hit at {price:.2f}")
                self.position.close()
                return


print("🌙✨ Launching Moon Dev LiquidationCloudRebound Backtest 🚀")
bt = Backtest(
    data,
    LiquidationCloudRebound,
    cash=1_000_000,
    commission=0.001
)
stats = bt.run()
print(stats)
print(stats._strategy)