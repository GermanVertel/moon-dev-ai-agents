import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Load data
data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Rename columns properly
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

# Set datetime index
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

print("🌙 Moon Dev Backtest Loading... ✨")
print(f"📊 Data shape: {data.shape}")
print(f"🚀 Date range: {data.index[0]} to {data.index[-1]}")


class ConfluencePattern(Strategy):
    """
    ConfluencePattern Strategy 🌙
    Trades when multiple indicators align at the same price/time zone.
    """
    # Parameters
    ema_fast_period = 50
    ema_slow_period = 200
    rsi_period = 14
    atr_period = 14
    rsi_oversold = 35
    rsi_overbought = 65
    risk_pct = 0.02          # 2% risk per trade
    rr_ratio = 2.0           # Minimum 1:2 risk-reward
    swing_lookback = 20

    def init(self):
        print("🌙 Initializing ConfluencePattern indicators... ✨")
        close = self.data.Close
        high = self.data.High
        low = self.data.Low

        # EMAs (talib wrapped in self.I - NO backtesting.lib)
        self.ema_fast = self.I(talib.EMA, close, timeperiod=self.ema_fast_period, name='EMA50')
        self.ema_slow = self.I(talib.EMA, close, timeperiod=self.ema_slow_period, name='EMA200')

        # RSI
        self.rsi = self.I(talib.RSI, close, timeperiod=self.rsi_period, name='RSI')

        # MACD
        self.macd, self.macd_signal, self.macd_hist = self.I(
            talib.MACD, close, fastperiod=12, slowperiod=26, signalperiod=9,
            name='MACD'
        )

        # ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period, name='ATR')

        # Swing highs/lows for S/R
        self.swing_high = self.I(talib.MAX, high, timeperiod=self.swing_lookback, name='SwingHigh')
        self.swing_low = self.I(talib.MIN, low, timeperiod=self.swing_lookback, name='SwingLow')

        # Volume SMA for confirmation
        self.vol_sma = self.I(talib.SMA, self.data.Volume, timeperiod=20, name='VolSMA')

        print("✅ All indicators initialized! 🚀")

    def next(self):
        # Need enough data
        if len(self.data) < self.ema_slow_period + 5:
            return

        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        open_ = self.data.Open[-1]

        # Current values
        ema_f = self.ema_fast[-1]
        ema_s = self.ema_slow[-1]
        rsi = self.rsi[-1]
        rsi_prev = self.rsi[-2]
        macd = self.macd[-1]
        macd_sig = self.macd_signal[-1]
        macd_prev = self.macd[-2]
        macd_sig_prev = self.macd_signal[-2]
        atr = self.atr[-1]
        swing_hi = self.swing_high[-1]
        swing_lo = self.swing_low[-1]
        volume = self.data.Volume[-1]
        vol_avg = self.vol_sma[-1]

        if atr <= 0 or np.isnan(atr):
            return

        # ---- Candlestick patterns ----
        body = abs(price - open_)
        candle_range = high - low
        if candle_range <= 0:
            return

        # Bullish engulfing / strong bullish candle
        bullish_candle = (price > open_) and (body / candle_range > 0.5)
        # Bearish engulfing / strong bearish candle
        bearish_candle = (price < open_) and (body / candle_range > 0.5)

        # Pin bar (long lower wick = bullish rejection)
        lower_wick = min(price, open_) - low
        upper_wick = high - max(price, open_)
        bullish_pin = lower_wick > 2 * body and lower_wick > upper_wick
        bearish_pin = upper_wick > 2 * body and upper_wick > lower_wick

        # ---- Confluence checks for LONG ----
        # 1. RSI oversold turning up
        rsi_long = rsi < self.rsi_oversold and rsi > rsi_prev
        # 2. MACD bullish crossover (manual - NO backtesting.lib.crossover)
        macd_long = macd_prev < macd_sig_prev and macd > macd_sig
        # 3. Price near 50 EMA (dynamic support) or swing low (structural support)
        near_ema_support = abs(price - ema_f) / price < 0.005
        near_swing_support = abs(price - swing_lo) / price < 0.005
        structural_long = near_ema_support or near_swing_support
        # 4. Price above 200 EMA (trend filter)
        trend_long = price > ema_s
        # 5. Bullish reversal candle
        reversal_long = bullish_candle or bullish_pin

        long_signals = sum([rsi_long, macd_long, structural_long, trend_long, reversal_long])

        # ---- Confluence checks for SHORT ----
        rsi_short = rsi > self.rsi_overbought and rsi < rsi_prev
        # MACD bearish crossover (manual - NO backtesting.lib.crossover)
        macd_short = macd_prev > macd_sig_prev and macd < macd_sig
        near_ema_resist = abs(price - ema_f) / price < 0.005
        near_swing_resist = abs(price - swing_hi) / price < 0.005
        structural_short = near_ema_resist or near_swing_resist
        trend_short = price < ema_s
        reversal_short = bearish_candle or bearish_pin

        short_signals = sum([rsi_short, macd_short, structural_short, trend_short, reversal_short])

        # Volume confirmation
        vol_confirm = volume > vol_avg * 0.8

        # ---- Manage existing positions ----
        if self.position:
            # Position object does NOT have .entry_price - use last trade's entry price
            if len(self.trades) > 0:
                entry = self.trades[-1].entry_price
            else:
                return

            if self.position.is_long:
                # Stop loss
                stop = entry - 1.5 * atr
                # Take profit at 1:2 RR
                risk = entry - stop
                tp = entry + self.rr_ratio * risk

                if price <= stop:
                    print(f"🛑 LONG STOP hit at {price:.2f} | entry {entry:.2f} 🌙")
                    self.position.close()
                elif price >= tp:
                    print(f"🎯 LONG TP hit at {price:.2f} | entry {entry:.2f} ✨")
                    self.position.close()
            else:
                stop = entry + 1.5 * atr
                risk = stop - entry
                tp = entry - self.rr_ratio * risk

                if price >= stop:
                    print(f"🛑 SHORT STOP hit at {price:.2f} | entry {entry:.2f} 🌙")
                    self.position.close()
                elif price <= tp:
                    print(f"🎯 SHORT TP hit at {price:.2f} | entry {entry:.2f} ✨")
                    self.position.close()
            return

        # ---- Entry logic ----
        # LONG: need at least 3 confluence signals (including reversal candle)
        if long_signals >= 3 and reversal_long and vol_confirm:
            stop = min(low, swing_lo) - 0.5 * atr
            risk = price - stop
            if risk > 0:
                rr = (price + self.rr_ratio * risk - price) / risk
                if rr >= self.rr_ratio:
                    # Position sizing: use fraction of equity (0 < size < 1)
                    size = 0.95
                    print(f"🚀🌙 LONG CONFLUENCE ENTRY | signals={long_signals} | "
                          f"price={price:.2f} stop={stop:.2f} RSI={rsi:.1f} ✨")
                    self.buy(size=size, sl=stop, tp=price + self.rr_ratio * risk)

        # SHORT: need at least 3 confluence signals
        elif short_signals >= 3 and reversal_short and vol_confirm:
            stop = max(high, swing_hi) + 0.5 * atr
            risk = stop - price
            if risk > 0:
                rr = (price - (price - self.rr_ratio * risk)) / risk
                if rr >= self.rr_ratio:
                    # Position sizing: use fraction of equity (0 < size < 1)
                    size = 0.95
                    print(f"🔻🌙 SHORT CONFLUENCE ENTRY | signals={short_signals} | "
                          f"price={price:.2f} stop={stop:.2f} RSI={rsi:.1f} ✨")
                    self.sell(size=size, sl=stop, tp=price - self.rr_ratio * risk)


# Run backtest
print("🌙✨ Starting Moon Dev ConfluencePattern Backtest... 🚀")
bt = Backtest(
    data,
    ConfluencePattern,
    cash=1000000,
    commission=0.001,
    trade_on_close=True,
)

stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙 Moon Dev Backtest Complete! ✨🚀")