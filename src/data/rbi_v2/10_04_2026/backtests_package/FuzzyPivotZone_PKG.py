import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# 🌙 Moon Dev's FuzzyPivotZone Strategy ✨🚀

def compute_rsi(close, period=14):
    return talib.RSI(close, timeperiod=period)

def compute_atr(high, low, close, period=14):
    return talib.ATR(high, low, close, timeperiod=period)

def compute_macd_hist(close):
    macd, signal, hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    return hist

def compute_bb_upper(close, period=20, nbdevup=2, nbdevdn=2):
    upper, middle, lower = talib.BBANDS(close, timeperiod=period, nbdevup=nbdevup, nbdevdn=nbdevdn, matype=0)
    return upper

def compute_bb_lower(close, period=20, nbdevup=2, nbdevdn=2):
    upper, middle, lower = talib.BBANDS(close, timeperiod=period, nbdevup=nbdevup, nbdevdn=nbdevdn, matype=0)
    return lower

class FuzzyPivotZone(Strategy):
    # Parameters
    rsi_period = 14
    atr_period = 14
    pivot_window = 20
    vol_ma_period = 20
    risk_per_trade = 0.01
    max_trades = 2
    fuzzy_threshold = 0.7
    medium_fuzzy = 0.5

    def init(self):
        # 🌙 Moon Dev indicators
        self.rsi = self.I(compute_rsi, self.data.Close, self.rsi_period)
        self.atr = self.I(compute_atr, self.data.High, self.data.Low, self.data.Close, self.atr_period)
        self.macd_hist = self.I(compute_macd_hist, self.data.Close)
        self.vol_ma = self.I(talib.SMA, self.data.Volume, timeperiod=self.vol_ma_period)
        self.pivot_high = self.I(talib.MAX, self.data.High, timeperiod=self.pivot_window)
        self.pivot_low = self.I(talib.MIN, self.data.Low, timeperiod=self.pivot_window)
        # Bollinger Band width for volatility filter
        self.bb_upper = self.I(compute_bb_upper, self.data.Close)
        self.bb_lower = self.I(compute_bb_lower, self.data.Close)
        self.bb_width = self.bb_upper - self.bb_lower
        self.bb_width_ma = self.I(talib.SMA, self.bb_width, timeperiod=20)

        # Track trade info
        self.entry_bar = 0
        self.trailing_active = False
        self.trail_stop = 0.0

    def fuzzy_membership(self, x, low, mid, high):
        # Triangular membership function
        if x <= low or x >= high:
            return 0.0
        if x == mid:
            return 1.0
        if x < mid:
            return (x - low) / (mid - low)
        return (high - x) / (high - mid)

    def candle_patterns(self, o, h, l, c, prev_o, prev_c):
        body = abs(c - o)
        range_ = h - l
        if range_ == 0:
            return 0, 0
        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l
        # Hammer: small body, long lower wick
        hammer = 0
        if lower_wick > 2 * body and upper_wick < body:
            hammer = 1
        # Bullish engulfing
        bull_engulf = 0
        if prev_c < prev_o and c > o and c >= prev_o and o <= prev_c:
            bull_engulf = 1
        # Shooting star: small body, long upper wick
        shooting_star = 0
        if upper_wick > 2 * body and lower_wick < body:
            shooting_star = 1
        # Bearish engulfing
        bear_engulf = 0
        if prev_c > prev_o and c < o and c <= prev_o and o >= prev_c:
            bear_engulf = 1
        return hammer + bull_engulf, shooting_star + bear_engulf

    def next(self):
        i = len(self.data) - 1
        if i < self.pivot_window + 5:
            return

        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        open_ = self.data.Open[-1]
        atr = self.atr[-1]
        rsi = self.rsi[-1]
        vol = self.data.Volume[-1]
        vol_ma = self.vol_ma[-1]
        macd_hist = self.macd_hist[-1]
        prev_macd_hist = self.macd_hist[-2]

        if atr == 0 or np.isnan(atr) or np.isnan(rsi):
            return

        # Volatility filter
        bb_width = self.bb_width[-1]
        bb_width_ma = self.bb_width_ma[-1]
        high_vol = bb_width > bb_width_ma if not np.isnan(bb_width_ma) else True

        # Fuzzy memberships
        # Near support: price close to pivot_low
        support = self.pivot_low[-1]
        resistance = self.pivot_high[-1]
        dist_to_support = abs(price - support) / atr
        dist_to_resistance = abs(price - resistance) / atr
        near_support = self.fuzzy_membership(dist_to_support, 0.0, 0.5, 2.0)
        near_resistance = self.fuzzy_membership(dist_to_resistance, 0.0, 0.5, 2.0)

        # RSI oversold/overbought
        rsi_oversold = self.fuzzy_membership(rsi, 20, 35, 50)
        rsi_overbought = self.fuzzy_membership(rsi, 50, 65, 80)

        # Volume spike
        vol_ratio = vol / vol_ma if vol_ma > 0 else 1.0
        vol_spike = self.fuzzy_membership(vol_ratio, 1.0, 2.0, 4.0)

        # Candle patterns
        bull_patterns, bear_patterns = self.candle_patterns(
            open_, high, low, price,
            self.data.Open[-2], self.data.Close[-2]
        )
        bull_rejection = min(1.0, bull_patterns)
        bear_rejection = min(1.0, bear_patterns)

        # MACD turning
        macd_turn_up = 1.0 if macd_hist > prev_macd_hist else 0.0
        macd_turn_down = 1.0 if macd_hist < prev_macd_hist else 0.0

        # Fuzzy score for long
        long_score = (0.35 * near_support +
                      0.20 * rsi_oversold +
                      0.15 * vol_spike +
                      0.20 * bull_rejection +
                      0.10 * macd_turn_up)
        # Fuzzy score for short
        short_score = (0.35 * near_resistance +
                       0.20 * rsi_overbought +
                       0.15 * vol_spike +
                       0.20 * bear_rejection +
                       0.10 * macd_turn_down)

        # 🌙 Debug prints
        if long_score > 0.5 or short_score > 0.5:
            print(f"🌙 Bar {i} | Price: {price:.2f} | LongScore: {long_score:.2f} | ShortScore: {short_score:.2f} | RSI: {rsi:.1f} | ATR: {atr:.2f}")

        # Manage existing trades
        for trade in self.trades:
            # Time exit: 5 bars
            bars_held = i - trade.entry_bar if hasattr(trade, 'entry_bar') else 0
            if bars_held >= 5 and not self.trailing_active:
                print(f"⏰ Time exit at bar {i}")
                trade.close()
                continue

            # Trailing stop
            if trade.is_long:
                if price - trade.entry_price >= atr:
                    self.trailing_active = True
                    new_trail = price - 0.75 * atr
                    if new_trail > self.trail_stop:
                        self.trail_stop = new_trail
                if self.trailing_active and price <= self.trail_stop:
                    print(f"🚀 Trailing stop hit for long at {price:.2f}")
                    trade.close()
            else:
                if trade.entry_price - price >= atr:
                    self.trailing_active = True
                    new_trail = price + 0.75 * atr
                    if new_trail < self.trail_stop or self.trail_stop == 0:
                        self.trail_stop = new_trail
                if self.trailing_active and price >= self.trail_stop:
                    print(f"🚀 Trailing stop hit for short at {price:.2f}")
                    trade.close()

        # Entry logic - only if no open trades
        if len(self.trades) >= self.max_trades:
            return

        equity = self.equity
        risk_amount = equity * self.risk_per_trade

        # Long entry
        if long_score > self.fuzzy_threshold and high_vol and price > support:
            stop_loss = support - 0.5 * atr
            risk = price - stop_loss
            if risk > 0:
                size = int(round(risk_amount / risk))
                if long_score < self.fuzzy_threshold + 0.2:
                    size = int(round(size * 0.5))  # medium confidence
                if size > 0:
                    tp = price + 1.5 * atr
                    print(f"✨ LONG ENTRY at {price:.2f} | SL: {stop_loss:.2f} | TP: {tp:.2f} | Size: {size} | Score: {long_score:.2f}")
                    self.buy(size=size, sl=stop_loss, tp=tp)
                    self.entry_bar = i
                    self.trailing_active = False
                    self.trail_stop = 0.0

        # Short entry
        elif short_score > self.fuzzy_threshold and high_vol and price < resistance:
            stop_loss = resistance + 0.5 * atr
            risk = stop_loss - price
            if risk > 0:
                size = int(round(risk_amount / risk))
                if short_score < self.fuzzy_threshold + 0.2:
                    size = int(round(size * 0.5))
                if size > 0:
                    tp = price - 1.5 * atr
                    print(f"✨ SHORT ENTRY at {price:.2f} | SL: {stop_loss:.2f} | TP: {tp:.2f} | Size: {size} | Score: {short_score:.2f}")
                    self.sell(size=size, sl=stop_loss, tp=tp)
                    self.entry_bar = i
                    self.trailing_active = False
                    self.trail_stop = 0.0


# 🌙 Load data
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data.columns = [c.capitalize() for c in data.columns]
if 'Datetime' in data.columns:
    data['Datetime'] = pd.to_datetime(data['Datetime'])
    data = data.set_index('Datetime')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

print("🌙✨ Moon Dev's FuzzyPivotZone Backtest Starting 🚀🌙")
print(f"Data shape: {data.shape}")

bt = Backtest(data, FuzzyPivotZone, cash=1000000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)