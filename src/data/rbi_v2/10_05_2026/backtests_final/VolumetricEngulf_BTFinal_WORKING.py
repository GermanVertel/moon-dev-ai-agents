import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# ============================================================
# 🌙 MOON DEV VOLUMETRIC ENGULF STRATEGY 🚀
# ============================================================

class VolumetricEngulf(Strategy):
    # Strategy parameters
    vol_sma_period = 20
    atr_period = 14
    swing_period = 10
    vol_multiplier = 1.8
    risk_pct = 0.01
    rr_target = 2.0
    time_exit_bars = 5

    def init(self):
        print("🌙✨ Initializing VolumetricEngulf Strategy...")
        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        open_ = pd.Series(self.data.Open)
        volume = pd.Series(self.data.Volume)

        # 📊 Volume SMA for spike detection
        self.vol_sma = self.I(talib.SMA, volume, timeperiod=self.vol_sma_period)

        # 📈 ATR for volatility expansion filter
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)
        self.atr_avg = self.I(talib.SMA, self.atr, timeperiod=self.atr_period)

        # 🎯 Swing highs/lows for breakout confirmation
        self.swing_high = self.I(talib.MAX, high, timeperiod=self.swing_period)
        self.swing_low = self.I(talib.MIN, low, timeperiod=self.swing_period)

        # 🕯️ Candle body metrics
        self.body = self.I(lambda o, c: np.abs(c - o), open_, close)
        self.range_ = self.I(lambda h, l: h - l, high, low)

        # Track entry bar for time exit
        self.entry_bar = None
        print("🌙✨ Indicators loaded successfully! 🚀")

    def _is_bullish_engulf(self, i):
        """Check if candle i is bullish engulfing candle i-1"""
        if i < 1:
            return False
        o, c = self.data.Open[i], self.data.Close[i]
        po, pc = self.data.Open[i-1], self.data.Close[i-1]
        # Current is bullish
        if c <= o:
            return False
        # Prior is bearish
        if pc >= po:
            return False
        # Current body engulfs prior body
        return o <= pc and c >= po

    def _is_bearish_engulf(self, i):
        """Check if candle i is bearish engulfing candle i-1"""
        if i < 1:
            return False
        o, c = self.data.Open[i], self.data.Close[i]
        po, pc = self.data.Open[i-1], self.data.Close[i-1]
        # Current is bearish
        if c >= o:
            return False
        # Prior is bullish
        if pc <= po:
            return False
        # Current body engulfs prior body
        return o >= pc and c <= po

    def next(self):
        i = len(self.data) - 1
        if i < 25:
            return

        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        vol = self.data.Volume[-1]
        vol_sma = self.vol_sma[-1]
        atr = self.atr[-1]
        atr_avg = self.atr_avg[-1]
        swing_high = self.swing_high[-2]  # prior swing high (exclude current)
        swing_low = self.swing_low[-2]

        if vol_sma is None or atr is None or atr_avg is None:
            return
        if np.isnan(vol_sma) or np.isnan(atr) or np.isnan(atr_avg):
            return

        # ============ MANAGE OPEN POSITION ============
        if self.position:
            bars_held = i - self.entry_bar if self.entry_bar is not None else 0
            # Time exit
            if bars_held >= self.time_exit_bars:
                print(f"⏰ Moon Dev Time Exit at bar {i} | PnL: {self.position.pl:.2f}")
                self.position.close()
                self.entry_bar = None
                return
            # Momentum exit: volume dries up
            if vol < vol_sma:
                print(f"💨 Momentum Exit — volume dried up at bar {i}")
                self.position.close()
                self.entry_bar = None
                return
            return

        # ============ ENTRY LOGIC ============
        bull_engulf = self._is_bullish_engulf(i)
        bear_engulf = self._is_bearish_engulf(i)

        vol_spike = vol >= self.vol_multiplier * vol_sma
        vol_expansion = atr > atr_avg

        # Long Entry
        if bull_engulf and vol_spike and vol_expansion and price > swing_high:
            entry_price = price
            stop_price = low
            risk = entry_price - stop_price
            if risk <= 0:
                return
            reward = risk * self.rr_target
            target = entry_price + reward

            # Position sizing: risk 1% of equity
            equity = self.equity
            risk_amount = equity * self.risk_pct
            size = int(round(risk_amount / risk))
            if size < 1:
                size = 1

            print(f"🚀🌙 LONG ENTRY | bar {i} | Price: {entry_price:.2f} | SL: {stop_price:.2f} | TP: {target:.2f} | Size: {size}")
            self.buy(size=size, sl=stop_price, tp=target)
            self.entry_bar = i

        # Short Entry
        elif bear_engulf and vol_spike and vol_expansion and price < swing_low:
            entry_price = price
            stop_price = high
            risk = stop_price - entry_price
            if risk <= 0:
                return
            reward = risk * self.rr_target
            target = entry_price - reward

            equity = self.equity
            risk_amount = equity * self.risk_pct
            size = int(round(risk_amount / risk))
            if size < 1:
                size = 1

            print(f"🔻🌙 SHORT ENTRY | bar {i} | Price: {entry_price:.2f} | SL: {stop_price:.2f} | TP: {target:.2f} | Size: {size}")
            self.sell(size=size, sl=stop_price, tp=target)
            self.entry_bar = i


# ============================================================
# 🌙 DATA LOADING & BACKTEST EXECUTION 🚀
# ============================================================
data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
print("🌙 Loading Moon Dev data...")
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
    'volume': 'Volume',
})

# Set datetime index
if 'datetime' in data.columns:
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')

print(f"🌙 Data loaded: {len(data)} bars ✨")

bt = Backtest(data, VolumetricEngulf, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)