import pandas as pd
import numpy as np
import talib
from backtesting import Backtest, Strategy

print("🌙✨ CompressionDivergence Strategy Loading... 🚀")

data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper case mapping
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

print(f"🌙 Data loaded: {len(data)} bars ✨")


class CompressionDivergence(Strategy):
    ema_fast_period = 20
    ema_slow_period = 50
    ema_macro_period = 200
    bb_period = 20
    bb_dev = 2.0
    kc_period = 20
    kc_mult = 1.5
    atr_period = 14
    rsi_period = 14
    vol_period = 20
    risk_pct = 0.01
    max_squeeze_lookback = 15
    min_squeeze_bars = 5

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        self.ema_fast = self.I(talib.EMA, close, timeperiod=self.ema_fast_period)
        self.ema_slow = self.I(talib.EMA, close, timeperiod=self.ema_slow_period)
        self.ema_macro = self.I(talib.EMA, close, timeperiod=self.ema_macro_period)

        # Bollinger Bands
        self.bb_upper, self.bb_mid, self.bb_lower = self.I(
            talib.BBANDS, close, timeperiod=self.bb_period,
            nbdevup=self.bb_dev, nbdevdn=self.bb_dev, matype=0
        )

        # Keltner Channels
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)
        self.atr_sma = self.I(talib.SMA, self.atr, timeperiod=20)
        self.kc_mid = self.I(talib.EMA, close, timeperiod=self.kc_period)

        # KC bands computed manually via self.I with proper array handling
        def kc_upper_fn():
            atr_arr = np.array(self.atr)
            kc_mid_arr = np.array(self.kc_mid)
            return kc_mid_arr + self.kc_mult * atr_arr

        def kc_lower_fn():
            atr_arr = np.array(self.atr)
            kc_mid_arr = np.array(self.kc_mid)
            return kc_mid_arr - self.kc_mult * atr_arr

        self.kc_upper = self.I(kc_upper_fn)
        self.kc_lower = self.I(kc_lower_fn)

        self.rsi = self.I(talib.RSI, close, timeperiod=self.rsi_period)
        self.vol_sma = self.I(talib.SMA, volume, timeperiod=self.vol_period)

        print("🌙 Indicators initialized ✨")

    def next(self):
        if len(self.data) < 210:
            return

        close = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        volume = self.data.Volume[-1]

        ema_f = self.ema_fast[-1]
        ema_s = self.ema_slow[-1]
        ema_m = self.ema_macro[-1]
        bb_u = self.bb_upper[-1]
        bb_l = self.bb_lower[-1]
        kc_u = self.kc_upper[-1]
        kc_l = self.kc_lower[-1]
        atr = self.atr[-1]
        atr_sma = self.atr_sma[-1]
        rsi = self.rsi[-1]
        vol_sma = self.vol_sma[-1]

        if np.isnan([ema_f, ema_s, ema_m, bb_u, bb_l, kc_u, kc_l, atr, atr_sma, rsi, vol_sma]).any():
            return

        # --- Compression detection ---
        squeeze_on = (bb_u < kc_u) and (bb_l > kc_l)
        atr_contracting = atr < atr_sma
        emas_flat = abs(ema_f - ema_s) < 0.5 * atr

        # Count consecutive closes inside BB
        inside_count = 0
        for k in range(1, min(20, len(self.data))):
            c = self.data.Close[-k]
            if bb_l <= c <= bb_u:
                inside_count += 1
            else:
                break

        compression = squeeze_on and atr_contracting and emas_flat and inside_count >= self.min_squeeze_bars

        # --- RSI Divergence detection (simple swing-based) ---
        lookback = min(self.max_squeeze_lookback, len(self.data) - 1)
        bullish_div = False
        bearish_div = False

        if lookback >= 5:
            recent_lows = [self.data.Low[-k] for k in range(1, lookback + 1)]
            recent_highs = [self.data.High[-k] for k in range(1, lookback + 1)]
            recent_rsi = [self.rsi[-k] for k in range(1, lookback + 1)]

            price_min_idx = int(np.argmin(recent_lows))
            rsi_min_idx = int(np.argmin(recent_rsi))
            if price_min_idx != rsi_min_idx:
                if recent_lows[price_min_idx] < recent_lows[rsi_min_idx] and recent_rsi[rsi_min_idx] > recent_rsi[price_min_idx]:
                    bullish_div = True

            price_max_idx = int(np.argmax(recent_highs))
            rsi_max_idx = int(np.argmax(recent_rsi))
            if price_max_idx != rsi_max_idx:
                if recent_highs[price_max_idx] > recent_highs[rsi_max_idx] and recent_rsi[rsi_max_idx] < recent_rsi[price_max_idx]:
                    bearish_div = True

        # --- Trend filter ---
        bull_trend = (ema_f > ema_s) or (close > ema_m)
        bear_trend = (ema_f < ema_s) or (close < ema_m)

        # --- Breakout conditions ---
        vol_confirm = volume > 1.5 * vol_sma
        long_breakout = close > bb_u and vol_confirm
        short_breakout = close < bb_l and vol_confirm

        # --- Confluence scoring ---
        def score_setup(direction):
            score = 0
            if inside_count >= 10:
                score += 1
            if lookback <= 5:
                score += 1
            if volume > 2.0 * vol_sma:
                score += 1
            if direction == 'long' and ema_f > ema_s:
                score += 1
            if direction == 'short' and ema_f < ema_s:
                score += 1
            rng = high - low
            if rng > 0:
                if direction == 'long' and (close - low) / rng >= 0.75:
                    score += 1
                if direction == 'short' and (high - close) / rng >= 0.75:
                    score += 1
            return score

        # --- Entry logic ---
        if not self.position:
            if compression and bullish_div and bull_trend and long_breakout:
                sc = score_setup('long')
                if sc >= 4:
                    stop_bb = bb_l
                    stop_atr = close - 1.5 * atr
                    stop = max(stop_bb, stop_atr)
                    risk = close - stop
                    if risk > 0:
                        risk_amount = self.equity * self.risk_pct
                        size = int(round(risk_amount / risk))
                        if size > 0:
                            print(f"🌙🚀 LONG ENTRY | Price: {close:.2f} | Stop: {stop:.2f} | Score: {sc} | Size: {size} ✨")
                            self.buy(size=size, sl=stop, tp=close + 3 * atr)

            elif compression and bearish_div and bear_trend and short_breakout:
                sc = score_setup('short')
                if sc >= 4:
                    stop_bb = bb_u
                    stop_atr = close + 1.5 * atr
                    stop = min(stop_bb, stop_atr)
                    risk = stop - close
                    if risk > 0:
                        risk_amount = self.equity * self.risk_pct
                        size = int(round(risk_amount / risk))
                        if size > 0:
                            print(f"🌙🔻 SHORT ENTRY | Price: {close:.2f} | Stop: {stop:.2f} | Score: {sc} | Size: {size} ✨")
                            self.sell(size=size, sl=stop, tp=close - 3 * atr)

        # --- Trailing / time stop management ---
        else:
            for trade in self.trades:
                bars_held = len(self.data) - 1 - trade.entry_bar
                if trade.is_long:
                    if close > trade.entry_price + atr and trade.sl < trade.entry_price:
                        trade.sl = trade.entry_price
                        print(f"🌙🔒 Long moved to BE @ {close:.2f} ✨")
                    if close > trade.entry_price + 2 * atr:
                        if trade.sl < ema_f:
                            trade.sl = ema_f
                            print(f"🌙📈 Long trailing stop -> EMA20 {ema_f:.2f} 🚀")
                    if bars_held >= 15 and close < trade.entry_price + 2 * atr:
                        print(f"🌙⏰ Long time stop hit @ {close:.2f} ✨")
                        trade.close()
                else:
                    if close < trade.entry_price - atr and trade.sl > trade.entry_price:
                        trade.sl = trade.entry_price
                        print(f"🌙🔒 Short moved to BE @ {close:.2f} ✨")
                    if close < trade.entry_price - 2 * atr:
                        if trade.sl > ema_f:
                            trade.sl = ema_f
                            print(f"🌙📉 Short trailing stop -> EMA20 {ema_f:.2f} 🚀")
                    if bars_held >= 15 and close > trade.entry_price - 2 * atr:
                        print(f"🌙⏰ Short time stop hit @ {close:.2f} ✨")
                        trade.close()


bt = Backtest(data, CompressionDivergence, cash=1_000_000, commission=0.001, exclusive_orders=True)
stats = bt.run()
print(stats)
print(stats._strategy)