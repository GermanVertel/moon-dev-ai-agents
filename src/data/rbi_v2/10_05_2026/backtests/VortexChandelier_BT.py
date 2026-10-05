import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# 🌙 Moon Dev Data Loading 🚀
data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# 🧹 Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# 🗺️ Proper column mapping
data.columns = [col.capitalize() for col in data.columns]
data = data.rename(columns={'Datetime': 'Datetime'})
data = data.set_index('Datetime')
data.index = pd.to_datetime(data.index)

print("🌙✨ Data loaded successfully! Rows:", len(data))
print("📊 Columns:", list(data.columns))


class VortexChandelier(Strategy):
    # ⚙️ Strategy Parameters
    vi_period = 14
    atr_period = 14
    chandelier_lookback = 22
    atr_multiplier = 3.0
    vol_sma_period = 10
    vol_threshold = 0.8
    ema_period = 200
    risk_pct = 0.02

    def init(self):
        # 🌪️ Vortex Indicator
        high = self.data.High
        low = self.data.Low
        close = self.data.Close

        # VM+ = |High - Low_prev|, VM- = |Low - High_prev|
        prev_low = pd.Series(low).shift(1).values
        prev_high = pd.Series(high).shift(1).values

        vm_plus = np.abs(np.array(high) - prev_low)
        vm_minus = np.abs(np.array(low) - prev_high)

        tr = talib.TRANGE(high, low, close)
        atr_vi = talib.SMA(tr, timeperiod=self.vi_period)

        vm_plus_sum = talib.SMA(vm_plus, timeperiod=self.vi_period) * self.vi_period
        vm_minus_sum = talib.SMA(vm_minus, timeperiod=self.vi_period) * self.vi_period
        atr_sum = talib.SMA(tr, timeperiod=self.vi_period) * self.vi_period

        with np.errstate(divide='ignore', invalid='ignore'):
            vi_plus = np.where(atr_sum != 0, vm_plus_sum / atr_sum, 0)
            vi_minus = np.where(atr_sum != 0, vm_minus_sum / atr_sum, 0)

        self.vi_plus = self.I(lambda: vi_plus, name='VI+')
        self.vi_minus = self.I(lambda: vi_minus, name='VI-')

        # 📊 ATR for Chandelier Exit
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period, name='ATR')

        # 📈 Highest High / Lowest Low for Chandelier
        self.hh = self.I(talib.MAX, high, timeperiod=self.chandelier_lookback, name='HH')
        self.ll = self.I(talib.MIN, low, timeperiod=self.chandelier_lookback, name='LL')

        # 📉 Volume SMA
        self.vol_sma = self.I(talib.SMA, self.data.Volume, timeperiod=self.vol_sma_period, name='VolSMA')

        # 📏 200 EMA trend filter
        self.ema = self.I(talib.EMA, close, timeperiod=self.ema_period, name='EMA200')

        print("🌙✨ VortexChandelier indicators initialized! 🚀")

    def next(self):
        price = self.data.Close[-1]

        # Skip if indicators not ready
        if np.isnan(self.vi_plus[-1]) or np.isnan(self.vi_minus[-1]) or np.isnan(self.atr[-1]):
            return
        if np.isnan(self.ema[-1]) or np.isnan(self.vol_sma[-1]):
            return

        vi_plus = self.vi_plus[-1]
        vi_minus = self.vi_minus[-1]
        vi_plus_prev = self.vi_plus[-2]
        vi_minus_prev = self.vi_minus[-2]
        atr = self.atr[-1]
        hh = self.hh[-1]
        ll = self.ll[-1]
        vol = self.data.Volume[-1]
        vol_sma = self.vol_sma[-1]
        ema = self.ema[-1]

        # 🕯️ Chandelier Exit levels
        chandelier_long = hh - self.atr_multiplier * atr
        chandelier_short = ll + self.atr_multiplier * atr

        # 🔄 Crossover detection
        bull_cross = vi_plus_prev <= vi_minus_prev and vi_plus > vi_minus
        bear_cross = vi_minus_prev <= vi_plus_prev and vi_minus > vi_plus

        # 📉 Declining volume confirmation
        declining_vol = vol < self.vol_threshold * vol_sma

        # 🚫 If in position, manage exits
        if self.position:
            if self.position.is_long:
                if price < chandelier_long:
                    print(f"🌙🔴 LONG EXIT @ {price:.2f} | Chandelier: {chandelier_long:.2f}")
                    self.position.close()
            elif self.position.is_short:
                if price > chandelier_short:
                    print(f"🌙🔴 SHORT EXIT @ {price:.2f} | Chandelier: {chandelier_short:.2f}")
                    self.position.close()
            return

        # 🟢 LONG ENTRY
        if bull_cross and declining_vol and price > ema:
            stop_price = chandelier_long
            risk_per_unit = price - stop_price
            if risk_per_unit > 0:
                risk_amount = self.equity * self.risk_pct
                position_size = int(round(risk_amount / risk_per_unit))
                if position_size > 0:
                    print(f"🌙🟢 LONG ENTRY @ {price:.2f} | VI+={vi_plus:.3f} VI-={vi_minus:.3f} | Stop={stop_price:.2f} | Size={position_size}")
                    self.buy(size=position_size)

        # 🔴 SHORT ENTRY
        elif bear_cross and declining_vol and price < ema:
            stop_price = chandelier_short
            risk_per_unit = stop_price - price
            if risk_per_unit > 0:
                risk_amount = self.equity * self.risk_pct
                position_size = int(round(risk_amount / risk_per_unit))
                if position_size > 0:
                    print(f"🌙🔴 SHORT ENTRY @ {price:.2f} | VI+={vi_plus:.3f} VI-={vi_minus:.3f} | Stop={stop_price:.2f} | Size={position_size}")
                    self.sell(size=position_size)


# 🚀 Run Backtest
bt = Backtest(data, VortexChandelier, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)