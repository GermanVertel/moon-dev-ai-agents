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
    'datetime': 'Date',
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

data['Date'] = pd.to_datetime(data['Date'])
data = data.set_index('Date')

print("🌙 Moon Dev: Data loaded and cleaned! ✨")
print(f"🚀 Shape: {data.shape}")
print(data.head())


class ContrarianCascade(Strategy):
    # Strategy parameters
    sma_fast = 50
    sma_slow = 200
    rsi_period = 14
    rsi_entry = 40
    rsi_exit = 50
    bb_period = 20
    bb_std = 2.0
    atr_period = 14
    atr_mult = 2.0
    cross_window = 10
    max_hold = 20
    risk_pct = 0.02

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low

        # SMAs
        self.sma_fast = self.I(talib.SMA, close, timeperiod=self.sma_fast, name='SMA50')
        self.sma_slow = self.I(talib.SMA, close, timeperiod=self.sma_slow, name='SMA200')

        # RSI
        self.rsi = self.I(talib.RSI, close, timeperiod=self.rsi_period, name='RSI')

        # Bollinger Bands
        self.bb_upper = self.I(talib.BBANDS, close, timeperiod=self.bb_period,
                               nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0,
                               name='BB_Upper', which=0)
        self.bb_mid = self.I(talib.BBANDS, close, timeperiod=self.bb_period,
                             nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0,
                             name='BB_Mid', which=1)
        self.bb_lower = self.I(talib.BBANDS, close, timeperiod=self.bb_period,
                               nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0,
                               name='BB_Lower', which=2)

        # ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period, name='ATR')

        # Track death cross bars
        self.cross_bar = -1
        self.bars_since_cross = 10**9

        print("🌙 Moon Dev: Indicators initialized! ✨")

    def next(self):
        # Skip if indicators not ready
        if len(self.data) < self.sma_slow + 5:
            return

        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        open_ = self.data.Open[-1]

        sma_f = self.sma_fast[-1]
        sma_s = self.sma_slow[-1]
        sma_f_prev = self.sma_fast[-2]
        sma_s_prev = self.sma_slow[-2]
        rsi_val = self.rsi[-1]
        bb_u = self.bb_upper[-1]
        bb_m = self.bb_mid[-1]
        bb_l = self.bb_lower[-1]
        atr_val = self.atr[-1]

        # Detect fresh death cross
        if sma_f_prev >= sma_s_prev and sma_f < sma_s:
            self.cross_bar = len(self.data)
            print(f"💀 Moon Dev: Death cross detected at bar {len(self.data)}! Price: {price:.2f}")

        # Track bars since cross
        if self.cross_bar > 0:
            self.bars_since_cross = len(self.data) - self.cross_bar

        # === EXIT LOGIC ===
        if self.position:
            # Primary exit: RSI reset below 50
            if rsi_val < self.rsi_exit:
                print(f"🌙 Moon Dev: RSI reset ({rsi_val:.2f} < {self.rsi_exit}) — EXIT LONG 🚀")
                self.position.close()
                self.cross_bar = -1
                return

            # Secondary exit: time stop
            if len(self.data) - self.position.entry_bar >= self.max_hold:
                print(f"⏰ Moon Dev: Time stop hit ({self.max_hold} bars) — EXIT LONG 🌙")
                self.position.close()
                self.cross_bar = -1
                return

            # Tertiary exit: stop loss below lower BB or 2x ATR
            if self.position.is_long:
                stop_price = min(bb_l, self.position.entry_price - self.atr_mult * atr_val)
                if low <= stop_price:
                    print(f"🛑 Moon Dev: Stop loss hit at {stop_price:.2f} — EXIT LONG")
                    self.position.close()
                    self.cross_bar = -1
                    return

        # === ENTRY LOGIC ===
        if not self.position and self.cross_bar > 0 and self.bars_since_cross <= self.cross_window:
            # Confirmation 1: RSI > 40
            cond_rsi = rsi_val > self.rsi_entry
            # Confirmation 2: price closed above upper BB
            cond_bb = price > bb_u
            # Optional: bullish candle
            cond_bull = price > open_

            if cond_rsi and cond_bb and cond_bull:
                # ATR-based stop distance
                stop_dist = self.atr_mult * atr_val
                if stop_dist <= 0:
                    return

                # Risk-based position sizing
                equity = self.equity
                risk_amount = equity * self.risk_pct
                position_size = int(round(risk_amount / stop_dist))
                if position_size < 1:
                    position_size = 1

                print(f"🚀 Moon Dev: CONTRARIAN CASCADE ENTRY! 🌙")
                print(f"   💀 Death cross {self.bars_since_cross} bars ago")
                print(f"   📈 RSI: {rsi_val:.2f} > {self.rsi_entry}")
                print(f"   🎯 Price {price:.2f} > Upper BB {bb_u:.2f}")
                print(f"   📊 Size: {position_size} | Stop dist: {stop_dist:.2f}")

                self.buy(size=position_size)


# Run backtest
bt = Backtest(data, ContrarianCascade, cash=1_000_000, commission=0.001, exclusive_orders=True)
stats = bt.run()
print(stats)
print(stats._strategy)