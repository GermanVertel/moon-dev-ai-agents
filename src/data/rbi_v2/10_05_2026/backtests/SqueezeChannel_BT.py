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

print("🌙 Moon Dev SqueezeChannel Backtest Loading... ✨")
print(f"🚀 Data shape: {data.shape}")
print(f"📊 Data head:\n{data.head()}")


class SqueezeChannel(Strategy):
    # Strategy parameters
    bb_period = 20
    bb_std = 2.0
    bbw_lookback = 126  # 6-month lookback on daily; scaled for 15m data
    donchian_period = 20
    rsi_period = 14
    rsi_exit_level = 70
    atr_period = 14
    atr_mult = 2.0
    sma_regime = 200
    risk_pct = 0.02  # 2% risk per trade
    squeeze_window = 5  # squeeze must occur within last 5 bars
    time_stop_bars = 20

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low

        # Bollinger Bands
        self.bb_upper = self.I(talib.BBANDS, close, timeperiod=self.bb_period,
                               nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0,
                               name='BB_Upper', index=0)
        self.bb_middle = self.I(talib.BBANDS, close, timeperiod=self.bb_period,
                                nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0,
                                name='BB_Middle', index=1)
        self.bb_lower = self.I(talib.BBANDS, close, timeperiod=self.bb_period,
                               nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0,
                               name='BB_Lower', index=2)

        # BBW = (upper - lower) / middle
        bbw = (self.bb_upper - self.bb_lower) / self.bb_middle
        self.bbw = self.I(lambda x: x, bbw, name='BBW')

        # Rolling minimum BBW over lookback
        self.bbw_min = self.I(lambda x: pd.Series(x).rolling(self.bbw_lookback).min().values,
                              bbw, name='BBW_Min')

        # Donchian Channel
        self.dc_upper = self.I(talib.MAX, high, timeperiod=self.donchian_period, name='DC_Upper')
        self.dc_lower = self.I(talib.MIN, low, timeperiod=self.donchian_period, name='DC_Lower')
        self.dc_mid = self.I(lambda u, l: (u + l) / 2, self.dc_upper, self.dc_lower, name='DC_Mid')

        # RSI
        self.rsi = self.I(talib.RSI, close, timeperiod=self.rsi_period, name='RSI')

        # ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period, name='ATR')

        # Regime filter SMA
        self.sma200 = self.I(talib.SMA, close, timeperiod=self.sma_regime, name='SMA200')

        # State trackers
        self.entry_bar = None
        self.entry_price = None
        self.stop_price = None
        self.rsi_above_70 = False

    def next(self):
        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]

        # Track RSI above 70 for exit management
        if self.rsi[-1] >= self.rsi_exit_level:
            self.rsi_above_70 = True

        # ==================== ENTRY LOGIC ====================
        if not self.position:
            # Condition A: BBW squeeze within last N bars
            squeeze_recent = False
            for i in range(1, self.squeeze_window + 1):
                try:
                    if self.bbw[-i] <= self.bbw_min[-i]:
                        squeeze_recent = True
                        break
                except IndexError:
                    pass

            # Condition B: Close above upper Donchian
            breakout = price > self.dc_upper[-2] if len(self.dc_upper) > 2 else False

            # Regime filter: price above 200 SMA (optional)
            regime_ok = True
            if not np.isnan(self.sma200[-1]):
                regime_ok = price > self.sma200[-1]

            if squeeze_recent and breakout and regime_ok:
                # Risk-based position sizing
                stop = self.dc_mid[-1]
                # Fallback to ATR stop if mid too close
                atr_stop = price - self.atr_mult * self.atr[-1]
                stop = min(stop, atr_stop) if stop < price else atr_stop

                risk_per_unit = price - stop
                if risk_per_unit <= 0:
                    return

                equity = self.equity
                risk_amount = equity * self.risk_pct
                position_size = int(round(risk_amount / risk_per_unit))

                if position_size > 0:
                    self.buy(size=position_size)
                    self.entry_bar = len(self.data)
                    self.entry_price = price
                    self.stop_price = stop
                    self.rsi_above_70 = False
                    print(f"🌙✨ SQUEEZE BREAKOUT! Entry @ {price:.2f} | Size: {position_size} | Stop: {stop:.2f} | BBW: {self.bbw[-1]:.6f} 🚀")

        # ==================== EXIT LOGIC ====================
        else:
            # Primary exit: RSI crosses below 70 after being above
            rsi_cross_down = (self.rsi[-2] >= self.rsi_exit_level and
                              self.rsi[-1] < self.rsi_exit_level and
                              self.rsi_above_70)

            # Stop loss
            stop_hit = low <= self.stop_price

            # Time stop
            time_stop = (self.entry_bar is not None and
                         (len(self.data) - self.entry_bar) >= self.time_stop_bars)

            if rsi_cross_down:
                self.position.close()
                print(f"🌙 RSI Exit @ {price:.2f} | RSI: {self.rsi[-1]:.2f} | PnL: {self.position.pl:.2f} ✨")
                self.rsi_above_70 = False
            elif stop_hit:
                self.position.close()
                print(f"🛑 Stop Loss @ {self.stop_price:.2f} | Price: {price:.2f} | PnL: {self.position.pl:.2f}")
                self.rsi_above_70 = False
            elif time_stop:
                self.position.close()
                print(f"⏰ Time Stop @ {price:.2f} | Bars held: {len(self.data) - self.entry_bar} | PnL: {self.position.pl:.2f}")
                self.rsi_above_70 = False


print("🌙🚀 Initializing Moon Dev SqueezeChannel Backtest... ✨")
bt = Backtest(data, SqueezeChannel, cash=1_000_000, commission=0.001, exclusive_orders=True)
stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙✨ Backtest Complete! 🚀")