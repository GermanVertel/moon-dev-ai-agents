import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Load and prepare data
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={'datetime': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
data['Date'] = pd.to_datetime(data['Date'])
data = data.set_index('Date')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
print("🌙✨ Moon Dev data loaded successfully! Shape:", data.shape)


class SequentialReversal(Strategy):
    rsi_period = 14
    vol_sma_period = 20
    entry_retrace = 0.38
    exit_retrace = 0.26
    risk_pct = 0.02

    def init(self):
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=self.rsi_period)
        self.vol_sma = self.I(talib.SMA, self.data.Volume, timeperiod=self.vol_sma_period)
        print("🚀 Moon Dev indicators initialized! RSI + Volume SMA ready 🌙")

    def next(self):
        i = len(self.data) - 1
        if i < 5:
            return

        rsi = self.rsi

        r0 = rsi[-5]
        r1 = rsi[-4]
        r2 = rsi[-3]
        r3 = rsi[-2]  # reversal day RSI close
        r4 = rsi[-1]  # current

        # Long: bearish structure (lower highs and lows) then reversal day closes above prior RSI
        bearish_structure = (r1 < r0) and (r2 < r1)
        bullish_reversal = (r3 > r2)

        # Short: bullish structure then reversal day closes below prior RSI
        bullish_structure = (r1 > r0) and (r2 > r1)
        bearish_reversal = (r3 < r2)

        vol_confirm = self.data.Volume[-2] > self.vol_sma[-2]

        rev_high = self.data.High[-2]
        rev_low = self.data.Low[-2]
        rev_range = rev_high - rev_low

        if not self.position:
            if bearish_structure and bullish_reversal and vol_confirm and rev_range > 0:
                entry_price = rev_high - self.entry_retrace * rev_range
                stop_price = rev_low + self.exit_retrace * rev_range
                if self.data.Low[-1] <= entry_price <= self.data.High[-1]:
                    # Risk-based sizing
                    risk_per_unit = entry_price - stop_price
                    if risk_per_unit > 0:
                        equity = self.equity
                        risk_amount = equity * self.risk_pct
                        # Use fractional sizing for percentage of equity (valid 0<size<1)
                        risk_frac = self.risk_pct * (entry_price / risk_per_unit)
                        size = min(0.99, max(0.01, risk_frac))
                        if size > 0:
                            # Ensure SL < LIMIT < TP ordering for the broker
                            sl_price = stop_price
                            if sl_price >= entry_price:
                                sl_price = entry_price - max(1e-6, abs(entry_price) * 1e-4)
                            # TP must be strictly above entry_price (LIMIT)
                            tp_price = rev_high
                            if tp_price <= entry_price:
                                tp_price = entry_price + max(1e-6, abs(entry_price) * 1e-4)
                            # Final guard to guarantee valid ordering
                            if not (sl_price < entry_price < tp_price):
                                tp_price = entry_price + max(1e-6, abs(entry_price) * 1e-4)
                                sl_price = entry_price - max(1e-6, abs(entry_price) * 1e-4)
                            print(f"🌙✨ LONG signal! Entry={entry_price:.2f} Stop={sl_price:.2f} TP={tp_price:.2f} Size={size:.4f} 🚀")
                            self.buy(size=size, sl=sl_price, tp=tp_price)
        else:
            # Exit if close falls below exit retrace level
            if self.position.is_long:
                exit_level = rev_low + self.exit_retrace * rev_range
                if self.data.Close[-1] < exit_level:
                    print(f"🌙 Exit long at {self.data.Close[-1]:.2f} (below {exit_level:.2f})")
                    self.position.close()


bt = Backtest(data, SequentialReversal, cash=1_000_000, commission=0.002)
stats = bt.run()
print(stats)
print(stats._strategy)