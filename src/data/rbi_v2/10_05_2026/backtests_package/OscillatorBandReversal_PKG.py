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
data.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')
data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

print("🌙 Moon Dev OscillatorBandReversal backtest starting! 🚀")
print(f"✨ Data loaded: {len(data)} bars")
print(f"📊 Date range: {data.index[0]} to {data.index[-1]}")


class OscillatorBandReversal(Strategy):
    # Strategy parameters
    so_period = 14
    so_k = 3
    so_d = 3
    bb_period = 20
    bb_std = 2.0
    vol_period = 20
    vol_mult = 1.5
    atr_period = 14
    atr_sl_mult = 1.5
    atr_tp1_mult = 1.0
    atr_tp2_mult = 2.0
    risk_pct = 0.01
    time_stop_bars = 15

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        # Bollinger Bands
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS, close, timeperiod=self.bb_period,
            nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0
        )

        # Stochastic Oscillator
        self.so_k, self.so_d = self.I(
            talib.STOCH, high, low, close,
            fastk_period=self.so_period, slowk_period=self.so_k,
            slowk_matype=0, slowd_period=self.so_d, slowd_matype=0
        )

        # Volume SMA
        self.vol_sma = self.I(talib.SMA, volume, timeperiod=self.vol_period)

        # ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        # ATR 100-period range filter (for volatility filter)
        self.atr_max = self.I(talib.MAX, self.atr, timeperiod=100)
        self.atr_min = self.I(talib.MIN, self.atr, timeperiod=100)

        # Track trade state
        self.entry_bar = None
        self.entry_price = None
        self.stop_price = None
        self.tp1_price = None
        self.tp2_price = None
        self.tp1_hit = False

    def next(self):
        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        open_ = self.data.Open[-1]
        volume = self.data.Volume[-1]

        # Skip if indicators not ready
        if (np.isnan(self.bb_upper[-1]) or np.isnan(self.so_k[-1]) or
                np.isnan(self.atr[-1]) or np.isnan(self.vol_sma[-1]) or
                np.isnan(self.atr_max[-1]) or np.isnan(self.atr_min[-1])):
            return

        # Manage open position
        if self.position:
            bars_held = len(self.data) - self.entry_bar

            # Time stop
            if bars_held >= self.time_stop_bars:
                print(f"⏰ Moon Dev Time Stop hit after {bars_held} bars! Closing. 🌙")
                self.position.close()
                self._reset_trade_state()
                return

            # Check stop loss
            if high >= self.stop_price:
                print(f"🛑 Moon Dev STOP hit at {self.stop_price:.2f}! Price: {high:.2f} 💥")
                self.position.close()
                self._reset_trade_state()
                return

            # TP1 - scale out 50%
            if not self.tp1_hit and low <= self.tp1_price:
                print(f"🎯 Moon Dev TP1 hit at {self.tp1_price:.2f}! Scaling out 50% ✨")
                self.position.close(0.5)
                self.tp1_hit = True

            # TP2 - close remainder
            if self.tp1_hit and low <= self.tp2_price:
                print(f"🚀 Moon Dev TP2 hit at {self.tp2_price:.2f}! Full exit! 🌙")
                self.position.close()
                self._reset_trade_state()
                return

            # Mean reversion target - middle band
            if low <= self.bb_middle[-1]:
                print(f"📉 Moon Dev Middle Band target hit at {self.bb_middle[-1]:.2f}! Closing. 🌙")
                self.position.close()
                self._reset_trade_state()
                return

            return

        # Entry logic (short side)
        # 1. SO overbought
        so_overbought = self.so_k[-1] > 80 and self.so_d[-1] > 80
        # 2. Price tags upper band
        band_tag = high >= self.bb_upper[-1]
        # 3. Volume spike
        vol_spike = volume >= self.vol_mult * self.vol_sma[-1]
        # 4. Bearish reversal candle closing back below upper band
        bearish_candle = price < open_
        close_below_band = price < self.bb_upper[-1]
        # 5. Volatility filter: ATR not in bottom 10% of 100-period range
        atr_range = self.atr_max[-1] - self.atr_min[-1]
        atr_threshold = self.atr_min[-1] + 0.1 * atr_range
        volatility_ok = self.atr[-1] > atr_threshold

        if (so_overbought and band_tag and vol_spike and
                bearish_candle and close_below_band and volatility_ok):

            # ATR-based levels
            atr_val = self.atr[-1]
            stop = price + self.atr_sl_mult * atr_val
            tp1 = price - self.atr_tp1_mult * atr_val
            tp2 = price - self.atr_tp2_mult * atr_val

            # Position sizing: risk 1% of equity
            equity = self.equity
            risk_amount = equity * self.risk_pct
            risk_per_unit = self.atr_sl_mult * atr_val
            if risk_per_unit <= 0:
                return
            position_size = int(round(risk_amount / risk_per_unit))
            if position_size <= 0:
                return

            print(f"🌙✨ Moon Dev SHORT signal! Price: {price:.2f}, "
                  f"SO_K: {self.so_k[-1]:.1f}, SO_D: {self.so_d[-1]:.1f}, "
                  f"Vol: {volume:.0f} vs SMA: {self.vol_sma[-1]:.0f}")
            print(f"🚀 Entry: {price:.2f} | Stop: {stop:.2f} | TP1: {tp1:.2f} | TP2: {tp2:.2f} | Size: {position_size}")

            self.sell(size=position_size)
            self.entry_bar = len(self.data)
            self.entry_price = price
            self.stop_price = stop
            self.tp1_price = tp1
            self.tp2_price = tp2
            self.tp1_hit = False

    def _reset_trade_state(self):
        self.entry_bar = None
        self.entry_price = None
        self.stop_price = None
        self.tp1_price = None
        self.tp2_price = None
        self.tp1_hit = False


bt = Backtest(data, OscillatorBandReversal, cash=1_000_000, commission=0.002)

stats = bt.run()
print(stats)
print(stats._strategy)