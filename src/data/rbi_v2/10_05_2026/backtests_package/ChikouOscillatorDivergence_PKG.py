import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Load data
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Ensure proper column mapping
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')

print("🌙 Moon Dev Backtest AI - Chikou Oscillator Divergence Strategy 🚀")
print(f"📊 Data loaded: {len(data)} bars")
print(f"📅 Date range: {data.index[0]} to {data.index[-1]}")
print("=" * 60)


class ChikouOscillatorDivergence(Strategy):
    # Strategy parameters
    chikou_period = 26
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    stoch_k = 14
    stoch_d = 3
    stoch_smooth = 3
    ema_period = 50
    atr_period = 14
    adx_period = 14
    adx_threshold = 20

    # Oscillator weights
    w_chikou = 0.40
    w_macd = 0.30
    w_stoch = 0.30

    # Risk management
    risk_pct = 0.02
    stop_loss_pct = 0.025
    take_profit_pct = 0.025
    breakeven_trigger = 0.015

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low

        # 50 EMA trend filter
        self.ema50 = self.I(talib.EMA, close, timeperiod=self.ema_period)

        # ATR for normalization
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        # ADX filter
        self.adx = self.I(talib.ADX, high, low, close, timeperiod=self.adx_period)

        # Chikou Span deviation: (close - close[N]) / close[N] * 100
        self.chikou_dev = self.I(
            lambda c, n: (c - c.shift(n)) / c.shift(n) * 100,
            close, self.chikou_period
        )

        # MACD histogram (extract only the histogram component)
        def _macd_hist(c, fast, slow, signal):
            macd, sig, hist = talib.MACD(c, fastperiod=fast, slowperiod=slow, signalperiod=signal)
            return hist

        self.macd_hist = self.I(
            _macd_hist,
            close, self.macd_fast, self.macd_slow, self.macd_signal
        )

        # Normalize MACD histogram by ATR
        self.macd_norm = self.I(
            lambda m, a: np.where(a > 0, (m / a) * 100, 0),
            self.macd_hist, self.atr
        )

        # Stochastic %K and %D — proper talib.STOCH signature:
        # STOCH(high, low, close, fastk_period, slowk_period, slowk_matype, slowd_period, slowd_matype)
        def _stoch_k(h, l, c, fastk, slowk, slowd):
            k, d = talib.STOCH(
                h, l, c,
                fastk_period=fastk,
                slowk_period=slowk,
                slowk_matype=0,
                slowd_period=slowd,
                slowd_matype=0
            )
            return k

        def _stoch_d(h, l, c, fastk, slowk, slowd):
            k, d = talib.STOCH(
                h, l, c,
                fastk_period=fastk,
                slowk_period=slowk,
                slowk_matype=0,
                slowd_period=slowd,
                slowd_matype=0
            )
            return d

        self.stoch_k_line = self.I(
            _stoch_k,
            high, low, close, self.stoch_k, self.stoch_smooth, self.stoch_d
        )
        self.stoch_d_line = self.I(
            _stoch_d,
            high, low, close, self.stoch_k, self.stoch_smooth, self.stoch_d
        )

        # Stochastic centered around 0 (-100 to +100)
        self.stoch_centered = self.I(
            lambda k: k - 50,
            self.stoch_k_line
        )

        # Custom oscillator blend
        self.oscillator = self.I(
            lambda ch, mc, st, w1, w2, w3: w1 * ch + w2 * mc + w3 * st,
            self.chikou_dev, self.macd_norm, self.stoch_centered,
            self.w_chikou, self.w_macd, self.w_stoch
        )

        # Track entry price for breakeven logic
        self.entry_price = None
        self.breakeven_moved = False

        print("🌙✨ Indicators initialized successfully!")
        print(f"   Chikou period: {self.chikou_period}")
        print(f"   MACD: ({self.macd_fast}, {self.macd_slow}, {self.macd_signal})")
        print(f"   Stochastic: ({self.stoch_k}, {self.stoch_d}, {self.stoch_smooth})")
        print(f"   EMA: {self.ema_period}, ATR: {self.atr_period}, ADX: {self.adx_period}")
        print(f"   Oscillator weights: Chikou={self.w_chikou}, MACD={self.w_macd}, Stoch={self.w_stoch}")

    def next(self):
        # Need enough data
        if len(self.data) < max(self.ema_period, self.chikou_period + 5, 50):
            return

        price = self.data.Close[-1]
        ema = self.ema50[-1]
        osc = self.oscillator[-1]
        osc_prev = self.oscillator[-2]
        adx = self.adx[-1]
        atr = self.atr[-1]

        stoch_k = self.stoch_k_line[-1]
        stoch_d = self.stoch_d_line[-1]
        stoch_k_prev = self.stoch_k_line[-2]
        stoch_d_prev = self.stoch_d_line[-2]

        macd_hist = self.macd_hist[-1]
        macd_hist_prev = self.macd_hist[-2]

        # ADX filter - avoid choppy markets
        if adx < self.adx_threshold:
            return

        # --- Manage existing position ---
        if self.position:
            entry = self.entry_price
            is_long = self.position.is_long

            # Trailing stop to breakeven
            if not self.breakeven_moved and entry is not None:
                if is_long and price >= entry * (1 + self.breakeven_trigger):
                    self.breakeven_moved = True
                    print(f"🌙💰 BREAKEVEN moved for LONG at {price:.2f}")
                elif not is_long and price <= entry * (1 - self.breakeven_trigger):
                    self.breakeven_moved = True
                    print(f"🌙💰 BREAKEVEN moved for SHORT at {price:.2f}")

            # Signal exit - oscillator crosses against position
            if is_long and osc < 0 and osc_prev >= 0:
                print(f"🌙🔄 LONG exit: oscillator crossed below zero at {price:.2f}")
                self.position.close()
                self.entry_price = None
                self.breakeven_moved = False
                return
            if not is_long and osc > 0 and osc_prev <= 0:
                print(f"🌙🔄 SHORT exit: oscillator crossed above zero at {price:.2f}")
                self.position.close()
                self.entry_price = None
                self.breakeven_moved = False
                return

            # Confirmation exit: MACD flips + Stochastic crosses opposite
            if is_long:
                if macd_hist < 0 and macd_hist_prev >= 0 and stoch_k < stoch_d and stoch_k_prev >= stoch_d_prev:
                    print(f"🌙⚠️ LONG confirmation exit at {price:.2f}")
                    self.position.close()
                    self.entry_price = None
                    self.breakeven_moved = False
                    return
            else:
                if macd_hist > 0 and macd_hist_prev <= 0 and stoch_k > stoch_d and stoch_k_prev <= stoch_d_prev:
                    print(f"🌙⚠️ SHORT confirmation exit at {price:.2f}")
                    self.position.close()
                    self.entry_price = None
                    self.breakeven_moved = False
                    return
            return

        # --- Entry logic ---
        if atr <= 0:
            return

        # Position sizing based on risk
        risk_amount = self.equity * self.risk_pct
        stop_distance = price * self.stop_loss_pct
        if stop_distance <= 0:
            return
        position_size = int(round(risk_amount / stop_distance))
        if position_size < 1:
            position_size = 1

        # LONG conditions
        uptrend = price > ema
        osc_cross_up = osc > 0 and osc_prev <= 0
        macd_confirm_long = macd_hist > 0 and macd_hist > macd_hist_prev
        stoch_confirm_long = stoch_k > stoch_d and stoch_k_prev <= stoch_d_prev and stoch_k < 30

        if uptrend and osc_cross_up and macd_confirm_long and stoch_confirm_long:
            sl = price * (1 - self.stop_loss_pct)
            tp = price * (1 + self.take_profit_pct)
            print(f"🚀🌙 LONG ENTRY at {price:.2f} | SL: {sl:.2f} | TP: {tp:.2f} | Size: {position_size}")
            print(f"   Osc: {osc:.2f} (prev {osc_prev:.2f}) | MACD hist: {macd_hist:.4f} | Stoch K/D: {stoch_k:.1f}/{stoch_d:.1f}")
            self.buy(size=position_size, sl=sl, tp=tp)
            self.entry_price = price
            self.breakeven_moved = False
            return

        # SHORT conditions
        downtrend = price < ema
        osc_cross_down = osc < 0 and osc_prev >= 0
        macd_confirm_short = macd_hist < 0 and macd_hist < macd_hist_prev
        stoch_confirm_short = stoch_k < stoch_d and stoch_k_prev >= stoch_d_prev and stoch_k > 70

        if downtrend and osc_cross_down and macd_confirm_short and stoch_confirm_short:
            sl = price * (1 + self.stop_loss_pct)
            tp = price * (1 - self.take_profit_pct)
            print(f"🔻🌙 SHORT ENTRY at {price:.2f} | SL: {sl:.2f} | TP: {tp:.2f} | Size: {position_size}")
            print(f"   Osc: {osc:.2f} (prev {osc_prev:.2f}) | MACD hist: {macd_hist:.4f} | Stoch K/D: {stoch_k:.1f}/{stoch_d:.1f}")
            self.sell(size=position_size, sl=sl, tp=tp)
            self.entry_price = price
            self.breakeven_moved = False
            return


# Run backtest
bt = Backtest(
    data,
    ChikouOscillatorDivergence,
    cash=1000000,
    commission=0.001,
    exclusive=False
)

print("\n🌙 Running initial backtest with default parameters...")
stats = bt.run()
print("\n" + "=" * 60)
print("🌙✨ FINAL BACKTEST STATISTICS ✨🌙")
print("=" * 60)
print(stats)
print("\n" + "=" * 60)
print("🌙 STRATEGY DETAILS 🌙")
print("=" * 60)
print(stats._strategy)
print("\n🚀 Moon Dev Backtest Complete! 🌙")