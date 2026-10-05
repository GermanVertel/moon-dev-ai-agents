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
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

print("🌙✨ Moon Dev Backtest Initializing... VolatilityContraction Strategy Loading 🚀")
print(f"📊 Data shape: {data.shape}")
print(f"📈 Date range: {data.index[0]} → {data.index[-1]}")


class VolatilityContraction(Strategy):
    # MACD params
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    # BB params
    bb_period = 20
    bb_std = 2.0
    # ATR
    atr_period = 14
    atr_mult = 1.5
    # Risk
    risk_pct = 0.02

    def init(self):
        print("🌙 Initializing indicators...")
        close = self.data.Close
        high = self.data.High
        low = self.data.Low

        # MACD
        def macd_line(close):
            macd, _, _ = talib.MACD(
                close,
                fastperiod=self.macd_fast,
                slowperiod=self.macd_slow,
                signalperiod=self.macd_signal
            )
            return macd

        def macd_signal_line(close):
            _, sig, _ = talib.MACD(
                close,
                fastperiod=self.macd_fast,
                slowperiod=self.macd_slow,
                signalperiod=self.macd_signal
            )
            return sig

        def macd_hist_line(close):
            _, _, hist = talib.MACD(
                close,
                fastperiod=self.macd_fast,
                slowperiod=self.macd_slow,
                signalperiod=self.macd_signal
            )
            return hist

        self.macd = self.I(macd_line, close, name='MACD')
        self.macd_sig = self.I(macd_signal_line, close, name='MACD_Signal')
        self.macd_hist = self.I(macd_hist_line, close, name='MACD_Hist')

        # Bollinger Bands
        def bb_upper(close):
            upper, _, _ = talib.BBANDS(
                close,
                timeperiod=self.bb_period,
                nbdevup=self.bb_std,
                nbdevdn=self.bb_std,
                matype=0
            )
            return upper

        def bb_middle(close):
            _, middle, _ = talib.BBANDS(
                close,
                timeperiod=self.bb_period,
                nbdevup=self.bb_std,
                nbdevdn=self.bb_std,
                matype=0
            )
            return middle

        def bb_lower(close):
            _, _, lower = talib.BBANDS(
                close,
                timeperiod=self.bb_period,
                nbdevup=self.bb_std,
                nbdevdn=self.bb_std,
                matype=0
            )
            return lower

        def bb_width(close):
            upper, _, lower = talib.BBANDS(
                close,
                timeperiod=self.bb_period,
                nbdevup=self.bb_std,
                nbdevdn=self.bb_std,
                matype=0
            )
            return upper - lower

        self.bb_upper = self.I(bb_upper, close, name='BB_Upper')
        self.bb_middle = self.I(bb_middle, close, name='BB_Middle')
        self.bb_lower = self.I(bb_lower, close, name='BB_Lower')
        self.bb_width = self.I(bb_width, close, name='BB_Width')

        # ATR
        def atr_func(high, low, close):
            return talib.ATR(high, low, close, timeperiod=self.atr_period)

        self.atr = self.I(atr_func, high, low, close, name='ATR')

        print("✨ Indicators ready!")

    def next(self):
        price = self.data.Close[-1]

        # Need enough history
        if len(self.data) < self.bb_period + 5:
            return

        # Guard against NaNs in indicators
        if (np.isnan(self.macd[-1]) or np.isnan(self.macd_sig[-1]) or
                np.isnan(self.macd_hist[-1]) or np.isnan(self.macd_hist[-2]) or
                np.isnan(self.bb_middle[-1]) or np.isnan(self.bb_middle[-2]) or
                np.isnan(self.bb_width[-1]) or np.isnan(self.bb_width[-2]) or
                np.isnan(self.bb_width[-3]) or np.isnan(self.bb_lower[-1])):
            return

        # --- Momentum condition ---
        macd_bullish = (
            (self.macd[-1] > self.macd_sig[-1] and self.macd[-2] <= self.macd_sig[-2])
            or (self.macd_hist[-1] > 0 and self.macd_hist[-2] <= 0)
        )

        # --- BB downtrend: middle band sloping down ---
        bb_downtrend = self.bb_middle[-1] < self.bb_middle[-2]

        # --- BB width expanding for 2+ consecutive bars ---
        width_expanding = (
            self.bb_width[-1] > self.bb_width[-2]
            and self.bb_width[-2] > self.bb_width[-3]
        )

        # --- Exit: close below lower band ---
        if self.position:
            if price < self.bb_lower[-1]:
                print(f"🚪🌙 EXIT — close {price:.2f} < lower band {self.bb_lower[-1]:.2f}")
                self.position.close()
            return

        # --- Entry ---
        if macd_bullish and bb_downtrend and width_expanding:
            atr_val = self.atr[-1]
            if np.isnan(atr_val) or atr_val <= 0:
                return

            # 2% risk per trade
            risk_amount = self.equity * self.risk_pct
            stop_distance = self.atr_mult * atr_val
            if stop_distance <= 0:
                return

            position_size = int(round(risk_amount / stop_distance))

            # Fallback to fixed notional sizing if calc fails
            if position_size <= 0:
                position_size = int(round((self.equity * 0.95) / price))

            if position_size <= 0:
                return

            # Ensure we have enough cash for the position
            if position_size * price > self.equity:
                position_size = int(self.equity / price)

            if position_size <= 0:
                return

            sl = price - stop_distance
            print(f"🚀🌙 LONG ENTRY — price: {price:.2f} | size: {position_size} | SL: {sl:.2f} | ATR: {atr_val:.2f}")
            self.buy(size=position_size, sl=sl)


bt = Backtest(data, VolatilityContraction, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)