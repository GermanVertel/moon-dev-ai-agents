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

        r0 = rsi[i-4]
        r1 = rsi[i-3]
        r2 = rsi[i-2]
        r3 = rsi[i-1]  # reversal day RSI close
        r4 = rsi[i]    # current

        # Long: bearish structure (lower highs and lows) then reversal day closes above prior RSI
        bearish_structure = (r1 < r0) and (r2 < r1)
        bullish_reversal = (r3 > r2)

        # Short: bullish structure then reversal day closes below prior RSI
        bullish_structure = (r1 > r0) and (r2 > r1)
        bearish_reversal = (r3 < r2)

        vol_confirm = self.data.Volume[i-1] > self.vol_sma[i-1]

        rev_high = self.data.High[i-1]
        rev_low = self.data.Low[i-1]
        rev_range = rev_high - rev_low

        if not self.position:
            if bearish_structure and bullish_reversal and vol_confirm and rev_range > 0:
                entry_price = rev_high - self.entry_retrace * rev_range
                stop_price = rev_low + self.exit_retrace * rev_range
                if self.data.Low[i] <= entry_price <= self.data.High[i]:
                    # Risk-based sizing
                    risk_per_unit = entry_price - stop_price
                    if risk_per_unit > 0:
                        equity = self.equity
                        risk_amount = equity * self.risk_pct
                        size = int(round(risk_amount / risk_per_unit))
                        if size > 0:
                            # Ensure SL < LIMIT < TP ordering for the broker
                            sl_price = min(stop_price, entry_price - 1e-6)
                            # TP must be strictly above entry_price (LIMIT)
                            tp_price = max(rev_high, entry_price + 1e-6)
                            # Final guard to guarantee valid ordering
                            if not (sl_price < entry_price < tp_price):
                                tp_price = entry_price + max(1e-6, abs(entry_price) * 1e-6)
                                sl_price = entry_price - max(1e-6, abs(entry_price) * 1e-6)
                            print(f"🌙✨ LONG signal! Entry={entry_price:.2f} Stop={sl_price:.2f} TP={tp_price:.2f} Size={size} 🚀")
                            self.buy(size=size, sl=sl_price, tp=tp_price)
        else:
            # Exit if close falls below exit retrace level
            if self.position.is_long:
                exit_level = rev_low + self.exit_retrace * rev_range
                if self.data.Close[i] < exit_level:
                    print(f"🌙 Exit long at {self.data.Close[i]:.2f} (below {exit_level:.2f})")
                    self.position.close()


bt = Backtest(data, SequentialReversal, cash=1_000_000, commission=0.002)
stats = bt.run()
print(stats)
print(stats._strategy)