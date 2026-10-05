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

print("🌙✨ Moon Dev OscillatorSqueeze Backtest Loading... 🚀")
print(f"📊 Data shape: {data.shape}")
print(f"📈 Columns: {list(data.columns)}")


class OscillatorSqueeze(Strategy):
    # Strategy parameters
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    bb_period = 20
    bb_dev = 2.0
    rsi_period = 14
    atr_period = 14
    channel_period = 20
    atr_avg_period = 20
    atr_expansion_mult = 2.0
    rsi_oversold = 30
    rsi_overbought = 70
    risk_pct = 0.02
    rr_ratio = 2.0
    max_bars_in_trade = 20

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low

        # MACD
        self.macd, self.macd_signal_line, self.macd_hist = self.I(
            talib.MACD, close,
            fastperiod=self.macd_fast,
            slowperiod=self.macd_slow,
            signalperiod=self.macd_signal
        )

        # Bollinger Bands
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS, close,
            timeperiod=self.bb_period,
            nbdevup=self.bb_dev,
            nbdevdn=self.bb_dev,
            matype=0
        )

        # Bollinger Band Width
        self.bbw = self.I(
            lambda u, m, l: (u - l) / m,
            self.bb_upper, self.bb_middle, self.bb_lower
        )

        # RSI
        self.rsi = self.I(talib.RSI, close, timeperiod=self.rsi_period)

        # ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        # ATR average
        self.atr_avg = self.I(talib.SMA, self.atr, timeperiod=self.atr_avg_period)

        # Donchian channel
        self.donchian_high = self.I(talib.MAX, high, timeperiod=self.channel_period)
        self.donchian_low = self.I(talib.MIN, low, timeperiod=self.channel_period)

        # Swing highs/lows
        self.swing_high = self.I(talib.MAX, high, timeperiod=10)
        self.swing_low = self.I(talib.MIN, low, timeperiod=10)

        print("🌙 Moon Dev indicators initialized! ✨")

    def next(self):
        if len(self.data) < 50:
            return

        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]

        # Current indicator values
        macd_now = self.macd[-1]
        macd_sig_now = self.macd_signal_line[-1]
        macd_prev = self.macd[-2]
        macd_sig_prev = self.macd_signal_line[-2]

        rsi_now = self.rsi[-1]
        rsi_prev = self.rsi[-2]

        bbw_now = self.bbw[-1]
        bbw_prev = self.bbw[-2]
        bbw_prev2 = self.bbw[-3]

        atr_now = self.atr[-1]
        atr_avg_now = self.atr_avg[-1]

        bb_upper = self.bb_upper[-1]
        bb_lower = self.bb_lower[-1]

        # MACD crossover signals (no backtesting.lib)
        macd_bull_cross = macd_prev <= macd_sig_prev and macd_now > macd_sig_now
        macd_bear_cross = macd_prev >= macd_sig_prev and macd_now < macd_sig_now

        # BBW expanding
        bbw_expanding = bbw_now > bbw_prev and bbw_prev > bbw_prev2

        # ATR rising
        atr_rising = atr_now > self.atr[-2]

        # ATR expansion
        atr_expansion = atr_now > (atr_avg_now * self.atr_expansion_mult) if atr_avg_now > 0 else False

        # RSI conditions
        rsi_oversold_now = rsi_now < self.rsi_oversold
        rsi_cross_up_30 = rsi_prev < self.rsi_oversold and rsi_now >= self.rsi_oversold
        rsi_overbought_now = rsi_now > self.rsi_overbought
        rsi_cross_down_70 = rsi_prev > self.rsi_overbought and rsi_now <= self.rsi_overbought

        # Price near bands
        near_lower_bb = low <= bb_lower * 1.005
        near_upper_bb = high >= bb_upper * 0.995

        # Volume filter
        vol_now = self.data.Volume[-1]
        vol_avg = np.mean(self.data.Volume[-20:]) if len(self.data) >= 20 else vol_now
        volume_spike = vol_now > vol_avg * 1.2

        # ---------------- LONG ENTRY ----------------
        if not self.position:
            long_signal = (
                macd_bull_cross and
                bbw_expanding and
                (rsi_oversold_now or rsi_cross_up_30) and
                atr_rising
            )

            if long_signal:
                # Stop at swing low or 1.5x ATR
                swing_low_val = self.swing_low[-1]
                atr_stop = price - 1.5 * atr_now
                stop_price = max(swing_low_val, atr_stop) if swing_low_val > 0 else atr_stop

                risk_per_unit = price - stop_price
                if risk_per_unit <= 0:
                    risk_per_unit = atr_now * 1.5
                    stop_price = price - risk_per_unit

                # Position sizing: risk 2% of equity
                equity = self.equity
                risk_amount = equity * self.risk_pct
                position_size = int(round(risk_amount / risk_per_unit))
                if position_size < 1:
                    position_size = 1

                # Cap position size to avoid absurd sizing
                max_size = int(equity / price) if price > 0 else 1
                if position_size > max_size:
                    position_size = max_size

                take_profit = price + (risk_per_unit * self.rr_ratio)

                print(f"🌙🚀 LONG SIGNAL! Price: {price:.2f} | RSI: {rsi_now:.1f} | MACD Cross ✅ | BBW Expanding ✅ | ATR Rising ✅")
                print(f"   Entry: {price:.2f} | Stop: {stop_price:.2f} | TP: {take_profit:.2f} | Size: {position_size}")

                self.buy(size=position_size, sl=stop_price, tp=take_profit)

        # ---------------- SHORT ENTRY ----------------
        if not self.position:
            short_signal = (
                macd_bear_cross and
                bbw_expanding and
                (rsi_overbought_now or rsi_cross_down_70) and
                atr_rising
            )

            if short_signal:
                swing_high_val = self.swing_high[-1]
                atr_stop = price + 1.5 * atr_now
                stop_price = min(swing_high_val, atr_stop) if swing_high_val > 0 else atr_stop

                risk_per_unit = stop_price - price
                if risk_per_unit <= 0:
                    risk_per_unit = atr_now * 1.5
                    stop_price = price + risk_per_unit

                equity = self.equity
                risk_amount = equity * self.risk_pct
                position_size = int(round(risk_amount / risk_per_unit))
                if position_size < 1:
                    position_size = 1

                max_size = int(equity / price) if price > 0 else 1
                if position_size > max_size:
                    position_size = max_size

                take_profit = price - (risk_per_unit * self.rr_ratio)

                print(f"🌙🔻 SHORT SIGNAL! Price: {price:.2f} | RSI: {rsi_now:.1f} | MACD Cross ✅ | BBW Expanding ✅ | ATR Rising ✅")
                print(f"   Entry: {price:.2f} | Stop: {stop_price:.2f} | TP: {take_profit:.2f} | Size: {position_size}")

                self.sell(size=position_size, sl=stop_price, tp=take_profit)

        # ---------------- EXIT LOGIC ----------------
        if self.position:
            # Time-based exit
            bars_in_trade = len(self.data) - self.position.entry_bar
            if bars_in_trade >= self.max_bars_in_trade:
                print(f"⏰ Time-based exit after {bars_in_trade} bars 🌙")
                self.position.close()
                return

            # ATR expansion exit (volatility extreme)
            if atr_expansion:
                print(f"🌪️ ATR expansion exit — volatility extreme! ATR: {atr_now:.2f} vs Avg: {atr_avg_now:.2f}")
                self.position.close()
                return

            # Channel breakout exit
            if self.position.is_long:
                if price > self.donchian_high[-2]:
                    print(f"📈 Channel breakout UP — trend continuation exit! Price: {price:.2f}")
                    self.position.close()
            elif self.position.is_short:
                if price < self.donchian_low[-2]:
                    print(f"📉 Channel breakout DOWN — trend continuation exit! Price: {price:.2f}")
                    self.position.close()


# Run initial backtest
print("\n🌙 Starting Moon Dev OscillatorSqueeze Backtest... 🚀✨")
bt = Backtest(
    data,
    OscillatorSqueeze,
    cash=1_000_000,
    commission=0.002,
    exclusive_orders=True
)

stats = bt.run()
print(stats)
print(stats._strategy)