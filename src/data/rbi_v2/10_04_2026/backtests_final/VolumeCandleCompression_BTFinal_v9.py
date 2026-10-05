import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy


# ============================================================
# 🌙 Moon Dev's VolumeCandle Compression Strategy 🌙
# ============================================================

def load_data(path):
    print("🌙 Loading data from Moon Dev's secret vault...")
    data = pd.read_csv(path)
    data.columns = data.columns.str.strip().str.lower()
    data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
    # Map to required columns
    data = data.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    })
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    print(f"✨ Data loaded: {len(data)} bars from {data.index[0]} to {data.index[-1]}")
    return data


class VolumeCandleCompression(Strategy):
    # Strategy Parameters
    vol_sma_period = 20
    atr_period = 14
    rvol_threshold = 1.0          # consolidation: RVOL < 1.0
    consolidation_bars = 6        # N consecutive low-RVOL bars
    range_width_pct = 0.02        # range width < 2% of price
    acv_mult_exit = 0.5           # exit when vol < 0.5*ACV
    acv_mult_entry = 1.0          # entry requires vol > ACV
    stop_atr_buffer = 0.5         # stop = band ± 0.5*ATR
    risk_pct = 0.01               # 1% risk per trade
    max_bars_in_trade = 40        # time-based exit

    def init(self):
        print("🌙 Initializing Moon Dev's VolumeCandle Compression Strategy...")
        self.vol_sma = self.I(talib.SMA, self.data.Volume, timeperiod=self.vol_sma_period)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period)
        self.rvol = self.I(lambda v, s: v / np.where(s == 0, np.nan, s), self.data.Volume, self.vol_sma)

        # Rolling highest high / lowest low over consolidation window
        self.hh = self.I(talib.MAX, self.data.High, timeperiod=self.consolidation_bars)
        self.ll = self.I(talib.MIN, self.data.Low, timeperiod=self.consolidation_bars)

        # Track consolidation state
        self.consolidation_high = np.nan
        self.consolidation_low = np.nan
        self.acv = np.nan
        self.in_consolidation = False
        self.entry_bar = None

        print("✨ Indicators ready: RVOL, ATR, HH/LL, VolSMA")

    def next(self):
        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        vol = self.data.Volume[-1]

        # Skip if indicators not ready
        if (np.isnan(self.rvol[-1]) or np.isnan(self.atr[-1])
                or np.isnan(self.hh[-1]) or np.isnan(self.ll[-1])):
            return

        # =====================================================
        # 🌙 Consolidation Detection
        # =====================================================
        if not self.in_consolidation and len(self.data) >= self.consolidation_bars + 1:
            window_rvol = self.rvol[-self.consolidation_bars:]
            window_high = self.hh[-1]
            window_low = self.ll[-1]
            range_width = (window_high - window_low) / window_low if window_low > 0 else 999

            low_rvol_count = np.sum(window_rvol < self.rvol_threshold)
            if low_rvol_count >= self.consolidation_bars - 1 and range_width < self.range_width_pct:
                self.in_consolidation = True
                self.consolidation_high = window_high
                self.consolidation_low = window_low
                # ACV = mean volume over consolidation window
                self.acv = np.mean(self.data.Volume[-self.consolidation_bars:])
                print(f"🌙✨ CONSOLIDATION DETECTED! High={self.consolidation_high:.2f} "
                      f"Low={self.consolidation_low:.2f} ACV={self.acv:.2f}")

        # =====================================================
        # 🌙 Trade Management
        # =====================================================
        if self.position:
            # Time-based exit
            if self.entry_bar is not None and (len(self.data) - self.entry_bar) >= self.max_bars_in_trade:
                print(f"⏰ Time-based exit after {self.max_bars_in_trade} bars")
                self.position.close()
                self.in_consolidation = False
                self.entry_bar = None
                return

            # Primary exit: volume collapse below 0.5 * ACV
            if not np.isnan(self.acv) and vol < self.acv_mult_exit * self.acv:
                print(f"💨 Volume collapse exit! vol={vol:.2f} ACV={self.acv:.2f}")
                self.position.close()
                self.in_consolidation = False
                self.entry_bar = None
                return

            # Take profit at opposite band
            if self.position.is_long and not np.isnan(self.consolidation_high) and high >= self.consolidation_high:
                print(f"🎯 Long TP hit at upper band {self.consolidation_high:.2f}")
                self.position.close()
                self.in_consolidation = False
                self.entry_bar = None
                return
            if self.position.is_short and not np.isnan(self.consolidation_low) and low <= self.consolidation_low:
                print(f"🎯 Short TP hit at lower band {self.consolidation_low:.2f}")
                self.position.close()
                self.in_consolidation = False
                self.entry_bar = None
                return
            return

        # =====================================================
        # 🌙 Entry Logic (only during consolidation)
        # =====================================================
        if not self.in_consolidation or np.isnan(self.acv):
            return

        # Volume confirmation: vol > ACV
        vol_confirmed = vol > self.acv_mult_entry * self.acv
        if not vol_confirmed:
            return

        # Bullish / Bearish rejection candles
        body = abs(self.data.Close[-1] - self.data.Open[-1])
        lower_wick = min(self.data.Open[-1], self.data.Close[-1]) - low
        upper_wick = high - max(self.data.Open[-1], self.data.Close[-1])
        bullish_rejection = (self.data.Close[-1] > self.data.Open[-1]) and (lower_wick > body)
        bearish_rejection = (self.data.Close[-1] < self.data.Open[-1]) and (upper_wick > body)

        # LONG entry: touch lower band + bullish rejection
        if low <= self.consolidation_low * 1.001 and bullish_rejection:
            stop = self.consolidation_low - self.stop_atr_buffer * self.atr[-1]
            risk_per_unit = price - stop
            if risk_per_unit > 0:
                risk_amount = self.equity * self.risk_pct
                size = int(round(risk_amount / risk_per_unit))
                max_size = int(self.equity * 0.95 / price)
                if size > max_size:
                    size = max_size
                if size >= 1:
                    print(f"🚀 LONG ENTRY! Price={price:.2f} Stop={stop:.2f} Size={size} 🌙")
                    self.buy(size=size, sl=stop)
                    self.entry_bar = len(self.data)

        # SHORT entry: touch upper band + bearish rejection
        elif high >= self.consolidation_high * 0.999 and bearish_rejection:
            stop = self.consolidation_high + self.stop_atr_buffer * self.atr[-1]
            risk_per_unit = stop - price
            if risk_per_unit > 0:
                risk_amount = self.equity * self.risk_pct
                size = int(round(risk_amount / risk_per_unit))
                max_size = int(self.equity * 0.95 / price)
                if size > max_size:
                    size = max_size
                if size >= 1:
                    print(f"🔻 SHORT ENTRY! Price={price:.2f} Stop={stop:.2f} Size={size} 🌙")
                    self.sell(size=size, sl=stop)
                    self.entry_bar = len(self.data)


# ============================================================
# 🌙 Run Backtest
# ============================================================
if __name__ == "__main__":
    data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
    data = load_data(data_path)

    bt = Backtest(
        data,
        VolumeCandleCompression,
        cash=1_000_000,
        commission=0.001,
        exclusive_orders=True
    )

    print("🌙✨🚀 Running Moon Dev's VolumeCandle Compression Backtest...")
    stats = bt.run()
    print(stats)
    print(stats._strategy)