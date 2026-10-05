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
data = data.rename(columns={
    'datetime': 'Datetime',
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

data['Datetime'] = pd.to_datetime(data['Datetime'])
data = data.set_index('Datetime')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

print("🌙✨ Moon Dev Data Loaded! Shape:", data.shape, "🚀")
print("🌙 First few rows:\n", data.head())


class SqueezeVolatility(Strategy):
    bb_period = 20
    bb_std = 2.0
    rsi_period = 14
    bbw_lookback = 100
    bbw_percentile = 0.20
    rsi_long_threshold = 60
    rsi_short_threshold = 40
    rsi_exit_long = 50
    rsi_exit_short = 50
    risk_pct = 0.02
    atr_period = 14
    atr_mult_sl = 2.0
    rr_ratio = 2.0

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low

        # Bollinger Bands
        self.bb_mid = self.I(talib.SMA, close, timeperiod=self.bb_period)
        self.bb_std_dev = self.I(talib.STDDEV, close, timeperiod=self.bb_period, nbdev=1)
        self.bb_upper = self.I(lambda c, m, s: m + self.bb_std * s,
                               close, self.bb_mid, self.bb_std_dev)
        self.bb_lower = self.I(lambda c, m, s: m - self.bb_std * s,
                               close, self.bb_mid, self.bb_std_dev)

        # Bollinger Band Width (guard against zero mid)
        self.bbw = self.I(lambda u, l, m: (u - l) / np.where(m == 0, np.nan, m),
                          self.bb_upper, self.bb_lower, self.bb_mid)

        # BBW percentile threshold over lookback
        self.bbw_threshold = self.I(
            lambda w: pd.Series(w).rolling(self.bbw_lookback).quantile(self.bbw_percentile).values,
            self.bbw
        )

        # RSI
        self.rsi = self.I(talib.RSI, close, timeperiod=self.rsi_period)

        # ATR for stop loss
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        # Volume SMA
        self.vol_sma = self.I(talib.SMA, self.data.Volume, timeperiod=20)

        print("🌙✨ Indicators initialized! Ready to squeeze! 🚀")

    def next(self):
        price = self.data.Close[-1]
        upper = self.bb_upper[-1]
        lower = self.bb_lower[-1]
        mid = self.bb_mid[-1]
        bbw_val = self.bbw[-1]
        bbw_thr = self.bbw_threshold[-1]
        rsi_val = self.rsi[-1]
        atr_val = self.atr[-1]
        vol = self.data.Volume[-1]
        vol_avg = self.vol_sma[-1]

        # Skip if indicators not ready
        if (np.isnan(bbw_thr) or np.isnan(rsi_val) or np.isnan(atr_val)
                or np.isnan(upper) or np.isnan(lower) or np.isnan(mid)
                or np.isnan(bbw_val) or np.isnan(vol_avg)):
            return

        # Manage open positions
        if self.position:
            if self.position.is_long:
                if price < mid or rsi_val < self.rsi_exit_long:
                    print(f"🌙 Long EXIT @ {price:.2f} | RSI={rsi_val:.2f} | Mid={mid:.2f} 💫")
                    self.position.close()
            elif self.position.is_short:
                if price > mid or rsi_val > self.rsi_exit_short:
                    print(f"🌙 Short EXIT @ {price:.2f} | RSI={rsi_val:.2f} | Mid={mid:.2f} 💫")
                    self.position.close()
            return

        # Squeeze condition
        squeeze = bbw_val <= bbw_thr
        volume_ok = vol > vol_avg

        if not squeeze or not volume_ok:
            return

        # Long entry
        if price > upper and rsi_val > self.rsi_long_threshold:
            sl = price - self.atr_mult_sl * atr_val
            tp = price + self.rr_ratio * (price - sl)
            risk_per_unit = price - sl
            if risk_per_unit <= 0:
                return
            risk_amount = self.equity * self.risk_pct
            size_units = risk_amount / risk_per_unit
            # Convert to fraction of equity for backtesting.py compatibility
            size_frac = (size_units * price) / self.equity
            size_frac = min(max(size_frac, 0.0001), 0.99)
            print(f"🚀🌙 LONG BREAKOUT! Price={price:.2f} Upper={upper:.2f} "
                  f"RSI={rsi_val:.2f} BBW={bbw_val:.4f} SL={sl:.2f} TP={tp:.2f} Size={size_frac:.4f}")
            self.buy(size=size_frac, sl=sl, tp=tp)

        # Short entry
        elif price < lower and rsi_val < self.rsi_short_threshold:
            sl = price + self.atr_mult_sl * atr_val
            tp = price - self.rr_ratio * (sl - price)
            risk_per_unit = sl - price
            if risk_per_unit <= 0:
                return
            risk_amount = self.equity * self.risk_pct
            size_units = risk_amount / risk_per_unit
            size_frac = (size_units * price) / self.equity
            size_frac = min(max(size_frac, 0.0001), 0.99)
            print(f"🚀🌙 SHORT BREAKOUT! Price={price:.2f} Lower={lower:.2f} "
                  f"RSI={rsi_val:.2f} BBW={bbw_val:.4f} SL={sl:.2f} TP={tp:.2f} Size={size_frac:.4f}")
            self.sell(size=size_frac, sl=sl, tp=tp)


bt = Backtest(data, SqueezeVolatility, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)