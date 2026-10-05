import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# ============================================================
# Moon Dev's VolumetricOI-Breakout Strategy 🌙
# ============================================================

DATA_PATH = '/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv'

print("🌙 Moon Dev is loading the cosmic data...")
data = pd.read_csv(DATA_PATH)
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Map columns to backtesting.py required format
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume',
})

if 'datetime' in data.columns:
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')

data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
print(f"✨ Moon Dev loaded {len(data)} candles of cosmic data!")


class VolumetricOIBreakout(Strategy):
    # Strategy parameters
    vwma_period = 20
    atr_period = 20
    kc_mult = 2.0
    oi_roc_lookback = 10
    oi_roc_threshold = 2.0
    vol_sma_period = 20
    vol_mult = 1.5
    price_roc_period = 10
    risk_pct = 0.01
    tp_atr_mult = 2.0
    sl_atr_mult = 1.5
    time_stop_bars = 30
    max_positions = 1

    def init(self):
        print("🚀 Moon Dev initializing VolumetricOI-Breakout indicators...")
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        # Volume-Weighted Moving Average (VWMA) as channel center
        def vwma(close, volume, period):
            close = np.asarray(close, dtype=float)
            volume = np.asarray(volume, dtype=float)
            pv = close * volume
            out = np.full(len(close), np.nan)
            for i in range(period - 1, len(close)):
                v_sum = np.sum(volume[i - period + 1:i + 1])
                if v_sum > 0:
                    out[i] = np.sum(pv[i - period + 1:i + 1]) / v_sum
                else:
                    out[i] = np.nan
            return out

        self.vwma = self.I(vwma, close, volume, self.vwma_period, name="VWMA")
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period, name="ATR")

        # Keltner Bands
        self.upper = self.I(lambda: self.vwma + self.kc_mult * self.atr, name="UpperKC")
        self.lower = self.I(lambda: self.vwma - self.kc_mult * self.atr, name="LowerKC")

        # Volume SMA
        self.vol_sma = self.I(talib.SMA, volume, timeperiod=self.vol_sma_period, name="VolSMA")

        # Price ROC
        self.price_roc = self.I(talib.ROC, close, timeperiod=self.price_roc_period, name="PriceROC")

        # OI ROC — no real OI data available; use volume-based proxy
        # Proxy: cumulative volume delta trend as "open interest" proxy
        def oi_proxy_roc(volume, period):
            volume = np.asarray(volume, dtype=float)
            # Cumulative sum acts as proxy for OI accumulation
            cum = np.cumsum(volume)
            out = np.full(len(volume), np.nan)
            for i in range(period, len(volume)):
                prev = cum[i - period]
                if prev > 0:
                    out[i] = ((cum[i] - prev) / prev) * 100.0
            return out

        self.oi_roc = self.I(oi_proxy_roc, volume, self.oi_roc_lookback, name="OI_ROC")

        self.entry_bar = None
        self.entry_price = None
        self.stop_price = None
        self.tp_price = None
        print("🌙✨ Indicators ready for launch!")

    def next(self):
        if len(self.data) < max(self.vwma_period, self.atr_period, self.vol_sma_period, self.oi_roc_lookback) + 2:
            return

        price = self.data.Close[-1]
        upper = self.upper[-1]
        lower = self.lower[-1]
        atr = self.atr[-1]
        vol = self.data.Volume[-1]
        vol_sma = self.vol_sma[-1]
        oi_roc = self.oi_roc[-1]
        price_roc = self.price_roc[-1]

        if any(np.isnan(x) for x in [upper, lower, atr, vol_sma, oi_roc, price_roc]):
            return

        # ---------- MANAGE OPEN POSITION ----------
        if self.position:
            bars_held = len(self.data) - self.entry_bar

            # Time stop
            if bars_held >= self.time_stop_bars:
                print(f"⏰ Moon Dev time stop hit after {bars_held} bars! Exiting.")
                self.position.close()
                self.entry_bar = None
                return

            # Failed breakout — price closes back inside band
            if self.position.is_long and price < upper:
                print(f"💥 Failed breakout (long) — price {price:.2f} < upper {upper:.2f}. Exiting!")
                self.position.close()
                self.entry_bar = None
                return
            if self.position.is_short and price > lower:
                print(f"💥 Failed breakout (short) — price {price:.2f} > lower {lower:.2f}. Exiting!")
                self.position.close()
                self.entry_bar = None
                return

            # Momentum exit — OI ROC flips negative
            if oi_roc < 0:
                print(f"📉 OI ROC flipped negative ({oi_roc:.2f}%) — losing conviction. Exiting!")
                self.position.close()
                self.entry_bar = None
                return

            # Volume continuation exit
            if vol < 0.8 * vol_sma:
                print(f"🔇 Volume weak ({vol:.2f} < 0.8×{vol_sma:.2f}). Exiting!")
                self.position.close()
                self.entry_bar = None
                return

            # Take profit / Stop loss checks
            if self.position.is_long:
                if price >= self.tp_price:
                    print(f"🎯 Moon Dev TP hit (long) at {price:.2f}! 🚀")
                    self.position.close()
                    self.entry_bar = None
                elif price <= self.stop_price:
                    print(f"🛑 Moon Dev SL hit (long) at {price:.2f}!")
                    self.position.close()
                    self.entry_bar = None
            elif self.position.is_short:
                if price <= self.tp_price:
                    print(f"🎯 Moon Dev TP hit (short) at {price:.2f}! 🚀")
                    self.position.close()
                    self.entry_bar = None
                elif price >= self.stop_price:
                    print(f"🛑 Moon Dev SL hit (short) at {price:.2f}!")
                    self.position.close()
                    self.entry_bar = None
            return

        # ---------- ENTRY LOGIC ----------
        volume_ok = vol >= self.vol_mult * vol_sma
        oi_ok = oi_roc >= self.oi_roc_threshold

        # Long entry
        if price > upper and oi_ok and volume_ok and price_roc > 0:
            risk = self.sl_atr_mult * atr
            if risk <= 0:
                return
            equity = self.equity
            risk_amount = equity * self.risk_pct
            position_size = int(round(risk_amount / risk))
            if position_size < 1:
                position_size = 1

            self.stop_price = price - risk
            self.tp_price = price + self.tp_atr_mult * atr
            print(f"🚀🌙 LONG BREAKOUT! Price={price:.2f} > Upper={upper:.2f} | OI_ROC={oi_roc:.2f}% | VolRatio={vol/vol_sma:.2f} | Size={position_size}")
            self.buy(size=position_size)
            self.entry_bar = len(self.data)
            self.entry_price = price

        # Short entry
        elif price < lower and oi_ok and volume_ok and price_roc < 0:
            risk = self.sl_atr_mult * atr
            if risk <= 0:
                return
            equity = self.equity
            risk_amount = equity * self.risk_pct
            position_size = int(round(risk_amount / risk))
            if position_size < 1:
                position_size = 1

            self.stop_price = price + risk
            self.tp_price = price - self.tp_atr_mult * atr
            print(f"🔻🌙 SHORT BREAKOUT! Price={price:.2f} < Lower={lower:.2f} | OI_ROC={oi_roc:.2f}% | VolRatio={vol/vol_sma:.2f} | Size={position_size}")
            self.sell(size=position_size)
            self.entry_bar = len(self.data)
            self.entry_price = price


print("🌙✨ Launching Moon Dev VolumetricOI-Breakout Backtest... 🚀")
bt = Backtest(
    data,
    VolumetricOIBreakout,
    cash=1_000_000,
    commission=0.0005,
    exclusive_orders=True,
)
stats = bt.run()
print(stats)
print(stats._strategy)