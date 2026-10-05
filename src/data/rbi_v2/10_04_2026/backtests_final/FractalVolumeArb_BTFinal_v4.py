import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Load data
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

print("🌙 Moon Dev Data Loaded! Shape:", data.shape)
print("🚀 Columns:", list(data.columns))


class FractalVolumeArb(Strategy):
    # Risk management parameters
    risk_pct = 0.01          # 1% risk per trade
    atr_period = 14
    vol_ma_period = 20
    rsi_period = 14
    adx_period = 14
    adx_threshold = 20

    def init(self):
        # ATR for volatility and stop sizing
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period)
        # Volume MA
        self.vol_ma = self.I(talib.SMA, self.data.Volume, timeperiod=self.vol_ma_period)
        # RSI for momentum
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=self.rsi_period)
        # ADX for regime filter
        self.adx = self.I(talib.ADX, self.data.High, self.data.Low, self.data.Close, timeperiod=self.adx_period)

        # Volume-weighted momentum: RSI * (Volume / Volume MA)
        def vw_rsi(close, volume, vol_ma, rsi):
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = np.where(vol_ma > 0, volume / vol_ma, 1.0)
            return rsi * ratio
        self.vw_rsi = self.I(vw_rsi, self.data.Close, self.data.Volume, self.vol_ma, self.rsi)

        # Williams Fractals (5-bar)
        # Bullish fractal: middle bar has lowest low
        def bull_fractal(low):
            n = len(low)
            out = np.zeros(n, dtype=bool)
            for i in range(2, n - 2):
                if (low[i] < low[i-1] and low[i] < low[i-2] and
                    low[i] < low[i+1] and low[i] < low[i+2]):
                    out[i] = True
            return out
        self.bull_frac = self.I(bull_fractal, self.data.Low)

        # Bearish fractal: middle bar has highest high
        def bear_fractal(high):
            n = len(high)
            out = np.zeros(n, dtype=bool)
            for i in range(2, n - 2):
                if (high[i] > high[i-1] and high[i] > high[i-2] and
                    high[i] > high[i+1] and high[i] > high[i+2]):
                    out[i] = True
            return out
        self.bear_frac = self.I(bear_fractal, self.data.High)

        # Track recent fractal levels
        self.last_bull_low = None
        self.last_bear_high = None
        self.entry_price = None
        self.stop_price = None
        self.target_price = None
        self.trade_direction = None

    def next(self):
        # Need enough bars
        if len(self.data) < 10:
            return

        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        atr = self.atr[-1]
        vw = self.vw_rsi[-1]
        vw_prev = self.vw_rsi[-2] if len(self.vw_rsi) > 2 else vw
        vol = self.data.Volume[-1]
        vol_ma = self.vol_ma[-1]
        adx = self.adx[-1]

        if np.isnan(atr) or np.isnan(vw) or np.isnan(adx) or np.isnan(vol_ma):
            return

        # Volatility filter
        if atr <= 0:
            return

        # Regime filter
        if adx < self.adx_threshold:
            return

        # Update fractal levels (use previous bar's confirmed fractal since current bar isn't confirmed yet)
        if len(self.bull_frac) > 3 and self.bull_frac[-2]:
            self.last_bull_low = self.data.Low[-3]
            print(f"🌙 Bullish Fractal detected at {self.last_bull_low:.2f} | vw_rsi={vw:.2f}")
        if len(self.bear_frac) > 3 and self.bear_frac[-2]:
            self.last_bear_high = self.data.High[-3]
            print(f"✨ Bearish Fractal detected at {self.last_bear_high:.2f} | vw_rsi={vw:.2f}")

        # ---- EXIT LOGIC ----
        if self.position:
            if self.trade_direction == 'long':
                # Trailing: move stop to last bear high if price moved 1 ATR in favor
                if self.last_bear_high and self.entry_price and price > self.entry_price + atr:
                    if self.stop_price is None or self.last_bear_high > self.stop_price:
                        self.stop_price = self.last_bear_high
                        print(f"🚀 Trailing long stop to {self.stop_price:.2f}")
                # Stop loss
                if self.stop_price and price <= self.stop_price:
                    print(f"🛑 Long stop hit at {price:.2f}")
                    self.position.close()
                    self.trade_direction = None
                # Take profit
                elif self.target_price and price >= self.target_price:
                    print(f"🎯 Long target hit at {price:.2f}")
                    self.position.close()
                    self.trade_direction = None
                # Momentum flip
                elif vw < 0 and vw_prev < 0:
                    print(f"⚠️ Momentum flip against long. Exiting.")
                    self.position.close()
                    self.trade_direction = None

            elif self.trade_direction == 'short':
                if self.last_bull_low and self.entry_price and price < self.entry_price - atr:
                    if self.stop_price is None or self.last_bull_low < self.stop_price:
                        self.stop_price = self.last_bull_low
                        print(f"🚀 Trailing short stop to {self.stop_price:.2f}")
                if self.stop_price and price >= self.stop_price:
                    print(f"🛑 Short stop hit at {price:.2f}")
                    self.position.close()
                    self.trade_direction = None
                elif self.target_price and price <= self.target_price:
                    print(f"🎯 Short target hit at {price:.2f}")
                    self.position.close()
                    self.trade_direction = None
                elif vw > 0 and vw_prev > 0:
                    print(f"⚠️ Momentum flip against short. Exiting.")
                    self.position.close()
                    self.trade_direction = None

        # ---- ENTRY LOGIC ----
        if not self.position:
            # Long entry: bullish fractal confirmed + close above fractal high + volume-momentum positive divergence
            if self.last_bull_low is not None and len(self.bull_frac) > 3 and self.bull_frac[-2]:
                # Momentum rising while price made lower low = hidden bullish arbitrage
                momentum_rising = vw > vw_prev
                # Price closes above recent 5-bar high
                recent_high = max(self.data.High[-5:])
                if momentum_rising and price > recent_high and vol > vol_ma:
                    stop = self.last_bull_low - 0.25 * atr
                    risk = price - stop
                    if risk > 0:
                        risk_amount = self.equity * self.risk_pct
                        # Use fractional sizing so backtesting.py handles it correctly
                        size_frac = risk_amount / (risk * price)
                        size_frac = max(0.001, min(0.99, size_frac))
                        self.buy(size=size_frac)
                        self.entry_price = price
                        self.stop_price = stop
                        self.target_price = price + 1.75 * atr
                        self.trade_direction = 'long'
                        print(f"🌙🚀 LONG ENTRY @ {price:.2f} | stop={stop:.2f} | target={self.target_price:.2f} | size={size_frac:.4f}")

            # Short entry: bearish fractal confirmed + close below fractal low + volume-momentum negative divergence
            elif self.last_bear_high is not None and len(self.bear_frac) > 3 and self.bear_frac[-2]:
                momentum_falling = vw < vw_prev
                recent_low = min(self.data.Low[-5:])
                if momentum_falling and price < recent_low and vol > vol_ma:
                    stop = self.last_bear_high + 0.25 * atr
                    risk = stop - price
                    if risk > 0:
                        risk_amount = self.equity * self.risk_pct
                        # Use fractional sizing so backtesting.py handles it correctly
                        size_frac = risk_amount / (risk * price)
                        size_frac = max(0.001, min(0.99, size_frac))
                        self.sell(size=size_frac)
                        self.entry_price = price
                        self.stop_price = stop
                        self.target_price = price - 1.75 * atr
                        self.trade_direction = 'short'
                        print(f"🌙✨ SHORT ENTRY @ {price:.2f} | stop={stop:.2f} | target={self.target_price:.2f} | size={size_frac:.4f}")


print("🌙 Moon Dev FractalVolumeArb Backtest Starting... 🚀")

bt = Backtest(data, FractalVolumeArb, cash=1_000_000, commission=0.001, exclusive_orders=True)
stats = bt.run()
print(stats)
print(stats._strategy)