import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Load data
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')

# Clean columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Map columns properly
data = data.rename(columns={
    'datetime': 'datetime',
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

# Set datetime index
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')

print("🌙✨ Moon Dev VortexFibonacci Backtest Initializing... 🚀")
print(f"📊 Data loaded: {len(data)} bars")
print(f"📅 Range: {data.index[0]} to {data.index[-1]}")


def vortex(high, low, close, n=14):
    """Calculate Vortex Indicator VI+ and VI-"""
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    vm_plus = np.abs(high - np.roll(low, 1))
    vm_minus = np.abs(low - np.roll(high, 1))
    vm_plus[0] = 0
    vm_minus[0] = 0
    tr_sum = talib.SMA(tr, timeperiod=n)
    vi_plus = talib.SMA(vm_plus, timeperiod=n) / tr_sum
    vi_minus = talib.SMA(vm_minus, timeperiod=n) / tr_sum
    return vi_plus, vi_minus


def efi(close, volume, n=13):
    """Elder's Force Index"""
    force = (close - np.roll(close, 1)) * volume
    force[0] = 0
    return talib.EMA(force, timeperiod=n)


class VortexFibonacci(Strategy):
    vi_period = 14
    efi_period = 13
    vol_ma_period = 20
    swing_lookback = 20
    risk_pct = 0.02
    atr_period = 14

    def init(self):
        print("🌙 Initializing indicators for VortexFibonacci...")
        high = self.data.High
        low = self.data.Low
        close = self.data.Close
        volume = self.data.Volume

        # Vortex Indicator
        vi_plus, vi_minus = self.I(vortex, high, low, close, self.vi_period)
        self.vi_plus = vi_plus
        self.vi_minus = vi_minus

        # Elder's Force Index
        self.efi = self.I(efi, close, volume, self.efi_period)

        # Volume MA
        self.vol_ma = self.I(talib.SMA, volume, timeperiod=self.vol_ma_period)

        # ATR for stops
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        # Swing highs/lows
        self.swing_high = self.I(talib.MAX, high, timeperiod=self.swing_lookback)
        self.swing_low = self.I(talib.MIN, low, timeperiod=self.swing_lookback)

        print("✨ Indicators ready!")

    def next(self):
        if len(self.data) < self.swing_lookback + 5:
            return

        price = self.data.Close[-1]
        vi_p = self.vi_plus[-1]
        vi_m = self.vi_minus[-1]
        vi_p_prev = self.vi_plus[-2]
        vi_m_prev = self.vi_minus[-2]

        efi_now = self.efi[-1]
        efi_prev = self.efi[-2]

        vol_now = self.data.Volume[-1]
        vol_ma = self.vol_ma[-1]

        sh = self.swing_high[-1]
        sl = self.swing_low[-1]

        # Trend confirmation
        bullish_trend = vi_p > vi_m and vi_p > vi_p_prev
        bearish_trend = vi_m > vi_p and vi_m > vi_m_prev

        # Volume confirmation
        vol_confirm = vol_now > vol_ma

        # EFI divergence checks
        # Bullish divergence: price makes lower low, EFI makes higher low
        price_lower_low = self.data.Low[-1] < self.data.Low[-2]
        efi_higher_low = efi_now > efi_prev

        # Bearish divergence: price makes higher high, EFI makes lower high
        price_higher_high = self.data.High[-1] > self.data.High[-2]
        efi_lower_high = efi_now < efi_prev

        if not self.position:
            # Long entry
            if bullish_trend and price_lower_low and efi_higher_low and vol_confirm and efi_now > 0:
                stop = sl - 0.5 * self.atr[-1]
                risk = price - stop
                if risk > 0:
                    fib_382 = sh - 0.382 * (sh - sl)
                    fib_500 = sh - 0.500 * (sh - sl)
                    fib_618 = sh - 0.618 * (sh - sl)
                    tp = fib_618 if fib_618 > price else fib_500
                    if tp <= price:
                        tp = price + 2 * risk
                    size = int(round((self.equity * self.risk_pct) / risk))
                    if size > 0:
                        print(f"🌙🚀 LONG ENTRY | Price: {price:.2f} | Stop: {stop:.2f} | TP: {tp:.2f} | Size: {size}")
                        self.buy(size=size, sl=stop, tp=tp)

            # Short entry
            elif bearish_trend and price_higher_high and efi_lower_high and vol_confirm and efi_now < 0:
                stop = sh + 0.5 * self.atr[-1]
                risk = stop - price
                if risk > 0:
                    fib_382 = sl + 0.382 * (sh - sl)
                    fib_500 = sl + 0.500 * (sh - sl)
                    fib_618 = sl + 0.618 * (sh - sl)
                    tp = fib_618 if fib_618 < price else fib_500
                    if tp >= price:
                        tp = price - 2 * risk
                    size = int(round((self.equity * self.risk_pct) / risk))
                    if size > 0:
                        print(f"🌙🔻 SHORT ENTRY | Price: {price:.2f} | Stop: {stop:.2f} | TP: {tp:.2f} | Size: {size}")
                        self.sell(size=size, sl=stop, tp=tp)


bt = Backtest(data, VortexFibonacci, cash=1_000_000, commission=0.001, exclusive_orders=True)
stats = bt.run()
print(stats)
print(stats._strategy)