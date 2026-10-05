import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# ============================================================
# 🌙 Moon Dev's SqueezeMomentum Backtest 🌙
# ============================================================

print("🌙✨ Moon Dev Backtest Engine Warming Up... 🚀")

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

print(f"🌙 Data loaded: {len(data)} bars from {data.index[0]} to {data.index[-1]} ✨")


class SqueezeMomentum(Strategy):
    # Strategy parameters
    adx_period = 14
    adx_threshold = 20
    bb_period = 20
    bb_std = 2.0
    bbw_lookback = 100
    bbw_percentile = 20
    kc_period = 20
    kc_atr_mult = 1.5
    atr_period = 14
    risk_pct = 0.01  # 1% risk per trade

    def init(self):
        print("🌙 Initializing Moon Dev indicators... ✨")

        # ADX and DI
        self.adx = self.I(talib.ADX, self.data.High, self.data.Low, self.data.Close,
                          timeperiod=self.adx_period)
        self.plus_di = self.I(talib.PLUS_DI, self.data.High, self.data.Low, self.data.Close,
                              timeperiod=self.adx_period)
        self.minus_di = self.I(talib.MINUS_DI, self.data.High, self.data.Low, self.data.Close,
                               timeperiod=self.adx_period)

        # Bollinger Bands
        self.bb_upper = self.I(talib.BBANDS, self.data.Close, timeperiod=self.bb_period,
                               nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0, which=0)
        self.bb_middle = self.I(talib.BBANDS, self.data.Close, timeperiod=self.bb_period,
                                nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0, which=1)
        self.bb_lower = self.I(talib.BBANDS, self.data.Close, timeperiod=self.bb_period,
                               nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0, which=2)

        # ATR
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close,
                          timeperiod=self.atr_period)

        # Keltner Channels (EMA +/- mult * ATR)
        self.kc_middle = self.I(talib.EMA, self.data.Close, timeperiod=self.kc_period)
        self.kc_upper = self.kc_middle + self.kc_atr_mult * self.atr
        self.kc_lower = self.kc_middle - self.kc_atr_mult * self.atr

        print("🌙✨ All indicators initialized! 🚀")

    def next(self):
        # Skip if not enough bars
        if len(self.data) < max(self.bbw_lookback, self.bb_period, self.adx_period) + 5:
            return

        price = self.data.Close[-1]

        # Compute BBW
        bb_width = (self.bb_upper[-1] - self.bb_lower[-1]) / self.bb_middle[-1]

        # BBW percentile over lookback
        bbw_history = []
        for i in range(1, self.bbw_lookback + 1):
            try:
                u = self.bb_upper[-i]
                l = self.bb_lower[-i]
                m = self.bb_middle[-i]
                if m and m > 0:
                    bbw_history.append((u - l) / m)
            except Exception:
                pass

        if len(bbw_history) < 20:
            return

        bbw_threshold = np.percentile(bbw_history, self.bbw_percentile)
        squeeze_on = bb_width <= bbw_threshold

        adx_val = self.adx[-1]
        plus = self.plus_di[-1]
        minus = self.minus_di[-1]

        trend_strong = adx_val > self.adx_threshold
        bull_trend = plus > minus
        bear_trend = minus > plus

        # Entry conditions
        long_signal = (
            trend_strong and bull_trend and squeeze_on and
            price > self.bb_upper[-1]
        )
        short_signal = (
            trend_strong and bear_trend and squeeze_on and
            price < self.bb_lower[-1]
        )

        # Exit conditions via Keltner
        if self.position:
            if self.position.is_long:
                if price < self.kc_lower[-1]:
                    print(f"🌙💥 LONG EXIT @ {price:.2f} — Keltner breach! ✨")
                    self.position.close()
            elif self.position.is_short:
                if price > self.kc_upper[-1]:
                    print(f"🌙💥 SHORT EXIT @ {price:.2f} — Keltner breach! ✨")
                    self.position.close()

        # Entries
        if not self.position:
            if long_signal:
                # Position sizing based on risk
                stop_price = self.bb_lower[-1]
                risk_per_unit = price - stop_price
                if risk_per_unit > 0:
                    risk_amount = self.equity * self.risk_pct
                    size = int(round(risk_amount / risk_per_unit))
                    if size > 0:
                        print(f"🌙🚀 LONG ENTRY @ {price:.2f} | ADX={adx_val:.1f} | BBW={bb_width:.4f} <= {bbw_threshold:.4f} | size={size}")
                        self.buy(size=size)

            elif short_signal:
                stop_price = self.bb_upper[-1]
                risk_per_unit = stop_price - price
                if risk_per_unit > 0:
                    risk_amount = self.equity * self.risk_pct
                    size = int(round(risk_amount / risk_per_unit))
                    if size > 0:
                        print(f"🌙🔻 SHORT ENTRY @ {price:.2f} | ADX={adx_val:.1f} | BBW={bb_width:.4f} <= {bbw_threshold:.4f} | size={size}")
                        self.sell(size=size)


# ============================================================
# Run Backtest
# ============================================================
print("🌙✨ Launching Moon Dev Backtest... 🚀🚀🚀")
bt = Backtest(data, SqueezeMomentum, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙✨ Moon Dev Backtest Complete! 🚀")