import pandas as pd
import numpy as np
import talib
from backtesting import Backtest, Strategy

# Load data
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')

# Clean columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Rename to proper case
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
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

# 🌙 Ensure all price/volume columns are float64 (talib requires double)
for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    data[col] = pd.to_numeric(data[col], errors='coerce').astype(np.float64)

data = data.dropna()

print("🌙 Moon Dev LiquidityDivergence Backtest Initializing... ✨")
print(f"📊 Data loaded: {len(data)} bars")
print(f"🚀 Data range: {data.index[0]} to {data.index[-1]}")


class LiquidityDivergence(Strategy):
    # Strategy parameters
    donchian_period = 20
    atr_period = 14
    atr_multiplier = 1.0        # channel widening
    vol_avg_period = 20         # for tick volume divergence
    atr_sl_mult = 1.0
    atr_tp_mult = 2.0
    risk_pct = 0.01
    low_liq_start = 22          # UTC hour
    low_liq_end = 2             # UTC hour (wraps)

    def init(self):
        print("🌙 Initializing indicators... ✨")
        # ATR
        self.atr = self.I(talib.ATR,
                          self.data.High.astype(np.float64),
                          self.data.Low.astype(np.float64),
                          self.data.Close.astype(np.float64),
                          timeperiod=self.atr_period)

        # Donchian base highs/lows
        self.dc_high = self.I(talib.MAX, self.data.High.astype(np.float64), timeperiod=self.donchian_period)
        self.dc_low = self.I(talib.MIN, self.data.Low.astype(np.float64), timeperiod=self.donchian_period)

        # Volume moving average — cast to float64 for talib
        self.vol_ma = self.I(talib.SMA, self.data.Volume.astype(np.float64), timeperiod=self.vol_avg_period)

        # RSI for momentum divergence confirmation
        self.rsi = self.I(talib.RSI, self.data.Close.astype(np.float64), timeperiod=14)

        print("🌙 Indicators ready! 🚀")

    def _in_low_liq_window(self, hour):
        if self.low_liq_start > self.low_liq_end:
            return hour >= self.low_liq_start or hour < self.low_liq_end
        else:
            return self.low_liq_start <= hour < self.low_liq_end

    def next(self):
        # Need enough bars
        if len(self.data) < max(self.donchian_period, self.vol_avg_period, self.atr_period) + 2:
            return

        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        vol = self.data.Volume[-1]
        atr = self.atr[-1]

        if np.isnan(atr) or atr <= 0:
            return

        # Volatility-adjusted Donchian bands
        upper_band = self.dc_high[-1] + atr * self.atr_multiplier
        lower_band = self.dc_low[-1] - atr * self.atr_multiplier

        vol_ma = self.vol_ma[-1]
        rsi = self.rsi[-1]

        # Get current UTC hour
        try:
            hour = self.data.index[-1].hour
        except Exception:
            hour = 0

        in_low_liq = self._in_low_liq_window(hour)

        # Skip if in position
        if self.position:
            # Time-based exit: if we leave low-liquidity window, close
            if not in_low_liq:
                print(f"🌙 Time-based exit — leaving low-liq window @ {price:.2f} ✨")
                self.position.close()
            return

        if not in_low_liq:
            return

        # Volume divergence: current bar volume lower than average
        vol_divergence = (not np.isnan(vol_ma)) and (vol < vol_ma)

        # Long entry: close above upper band + volume divergence + RSI not overbought
        if price > upper_band and vol_divergence and (not np.isnan(rsi)) and rsi < 70:
            sl = price - atr * self.atr_sl_mult
            tp = price + atr * self.atr_tp_mult
            risk_per_unit = price - sl
            if risk_per_unit <= 0:
                return
            equity = self.equity
            risk_amount = equity * self.risk_pct
            size = int(round(risk_amount / risk_per_unit))
            if size > 0:
                print(f"🚀 LONG signal @ {price:.2f} | upper_band={upper_band:.2f} | vol={vol:.2f} < volMA={vol_ma:.2f} | RSI={rsi:.2f} | SL={sl:.2f} TP={tp:.2f} | size={size} 🌙")
                self.buy(size=size, sl=sl, tp=tp)

        # Short entry: close below lower band + volume divergence + RSI not oversold
        elif price < lower_band and vol_divergence and (not np.isnan(rsi)) and rsi > 30:
            sl = price + atr * self.atr_sl_mult
            tp = price - atr * self.atr_tp_mult
            risk_per_unit = sl - price
            if risk_per_unit <= 0:
                return
            equity = self.equity
            risk_amount = equity * self.risk_pct
            size = int(round(risk_amount / risk_per_unit))
            if size > 0:
                print(f"🔻 SHORT signal @ {price:.2f} | lower_band={lower_band:.2f} | vol={vol:.2f} < volMA={vol_ma:.2f} | RSI={rsi:.2f} | SL={sl:.2f} TP={tp:.2f} | size={size} 🌙")
                self.sell(size=size, sl=sl, tp=tp)


# Run backtest
bt = Backtest(data, LiquidityDivergence, cash=1_000_000, commission=0.0005, exclusive_orders=False)

print("🌙 Running Moon Dev LiquidityDivergence backtest... 🚀✨")
stats = bt.run()
print(stats)
print(stats._strategy)