import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

print("🌙✨ Moon Dev Backtest AI Initializing... MomentumFlow Divergence 🚀")

# Load data
data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

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
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')

print(f"🌙 Data loaded: {len(data)} rows ✨")
print(f"🚀 Columns: {list(data.columns)}")


class MomentumFlowDivergence(Strategy):
    cmf_period = 20
    cmf_ma_period = 20
    stoch_k = 14
    stoch_d = 3
    stoch_smooth = 3
    atr_period = 14
    atr_mult = 2.0
    risk_pct = 0.02

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        # Chaikin Money Flow
        def cmf_func(high, low, close, volume, period):
            mfm = ((close - low) - (high - close)) / (high - low)
            mfm = np.where((high - low) == 0, 0, mfm)
            mfv = mfm * volume
            cmf = pd.Series(mfv).rolling(period).sum() / pd.Series(volume).rolling(period).sum()
            return cmf.values

        self.cmf = self.I(cmf_func, high, low, close, volume, self.cmf_period, name="CMF")
        self.cmf_ma = self.I(talib.SMA, self.cmf, timeperiod=self.cmf_ma_period, name="CMF_MA")

        # Stochastic
        self.stoch_k_line, self.stoch_d_line = self.I(
            talib.STOCH, high, low, close,
            fastk_period=self.stoch_k,
            slowk_period=self.stoch_smooth,
            slowk_matype=0,
            slowd_period=self.stoch_d,
            slowd_matype=0,
            name="STOCH"
        )

        # ATR for trailing stop
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period, name="ATR")

        print("🌙 Indicators initialized ✨")

    def next(self):
        if len(self.data) < max(self.cmf_period, self.cmf_ma_period, self.stoch_k + self.stoch_smooth, self.stoch_d + self.stoch_smooth) + 5:
            return

        price = self.data.Close[-1]
        cmf = self.cmf[-1]
        cmf_ma = self.cmf_ma[-1]
        k = self.stoch_k_line[-1]
        d = self.stoch_d_line[-1]
        k_prev = self.stoch_k_line[-2]
        d_prev = self.stoch_d_line[-2]
        atr = self.atr[-1]

        # Long entry: Stoch declining/crossed below D, CMF > MA and CMF > 0, Stoch between 20-50
        stoch_bearish = (k < d and k_prev >= d_prev) or (k < 50 and k < k_prev)
        cmf_bullish = cmf > cmf_ma and cmf > 0
        stoch_zone_ok = 20 <= k <= 60

        # Short entry: Stoch rising/crossed above D, CMF < MA and CMF < 0, Stoch between 40-80
        stoch_bullish = (k > d and k_prev <= d_prev) or (k > 50 and k > k_prev)
        cmf_bearish = cmf < cmf_ma and cmf < 0
        stoch_zone_short = 40 <= k <= 80

        # Exit conditions
        long_exit = (k > 80 and k < k_prev) or (cmf < cmf_ma)
        short_exit = (k < 20 and k > k_prev) or (cmf > cmf_ma)

        if not self.position:
            if stoch_bearish and cmf_bullish and stoch_zone_ok:
                sl = price - atr * self.atr_mult
                risk_per_unit = price - sl
                if risk_per_unit > 0:
                    risk_amount = self.equity * self.risk_pct
                    size = int(round(risk_amount / risk_per_unit))
                    size = max(1, min(size, int(self.equity / price)))
                    tp = price + risk_per_unit * 2
                    self.buy(size=size, sl=sl, tp=tp)
                    print(f"🌙🚀 LONG ENTRY | Price: {price:.2f} | CMF: {cmf:.3f} > MA: {cmf_ma:.3f} | Stoch K: {k:.1f} D: {d:.1f} | SL: {sl:.2f} | TP: {tp:.2f} | Size: {size}")

            elif stoch_bullish and cmf_bearish and stoch_zone_short:
                sl = price + atr * self.atr_mult
                risk_per_unit = sl - price
                if risk_per_unit > 0:
                    risk_amount = self.equity * self.risk_pct
                    size = int(round(risk_amount / risk_per_unit))
                    size = max(1, min(size, int(self.equity / price)))
                    tp = price - risk_per_unit * 2
                    self.sell(size=size, sl=sl, tp=tp)
                    print(f"🌙🔻 SHORT ENTRY | Price: {price:.2f} | CMF: {cmf:.3f} < MA: {cmf_ma:.3f} | Stoch K: {k:.1f} D: {d:.1f} | SL: {sl:.2f} | TP: {tp:.2f} | Size: {size}")

        else:
            if self.position.is_long and long_exit:
                self.position.close()
                print(f"🌙✨ LONG EXIT | Price: {price:.2f} | Stoch K: {k:.1f} | CMF: {cmf:.3f}")
            elif self.position.is_short and short_exit:
                self.position.close()
                print(f"🌙✨ SHORT EXIT | Price: {price:.2f} | Stoch K: {k:.1f} | CMF: {cmf:.3f}")


bt = Backtest(data, MomentumFlowDivergence, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)