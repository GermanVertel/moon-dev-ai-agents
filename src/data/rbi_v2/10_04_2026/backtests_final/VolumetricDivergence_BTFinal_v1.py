import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy


print("🌙✨ Starting Moon Dev's Volumetric Divergence Backtest! ✨🌙")


def load_data(path):
    data = pd.read_csv(path)
    data.columns = data.columns.str.strip().str.lower()
    data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
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
    return data


class VolumetricDivergence(Strategy):
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    vol_sma_period = 20
    bb_period = 20
    bb_dev = 2
    atr_period = 14
    swing_lookback = 5
    divergence_window = 40
    vol_mult = 1.5
    atr_tp_mult = 2.5
    atr_sl_mult = 1.5
    bb_expansion = 1.20
    risk_pct = 0.02

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        self.macd, self.macd_signal, self.macd_hist = self.I(
            talib.MACD, close,
            fastperiod=self.macd_fast,
            slowperiod=self.macd_slow,
            signalperiod=self.macd_signal
        )
        self.vol_sma = self.I(talib.SMA, volume, timeperiod=self.vol_sma_period)
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS, close,
            timeperiod=self.bb_period,
            nbdevup=self.bb_dev,
            nbdevdn=self.bb_dev
        )
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        self.swing_low = self.I(talib.MIN, low, timeperiod=self.swing_lookback)

        # 🌙 Track trade metadata in dicts keyed by trade entry bar
        self.atr_at_entry = {}
        self.bb_width_at_entry = {}

        print("🌙 Moon Dev indicators initialized! MACD, BB, ATR, Vol SMA ready 🚀")

    def next(self):
        price = self.data.Close[-1]

        if len(self.data) < self.divergence_window + 5:
            return

        if self.position:
            # 🌙 Safely get entry info from last trade
            if len(self.trades) == 0:
                return
            last_trade = self.trades[-1]
            entry_price = last_trade.entry_price
            entry_bar = last_trade.entry_bar
            entry_atr = self.atr_at_entry.get(entry_bar, self.atr[-1])
            entry_bb_width = self.bb_width_at_entry.get(entry_bar, None)

            tp_level = entry_price + entry_atr * self.atr_tp_mult

            current_bb_width = (self.bb_upper[-1] - self.bb_lower[-1]) / self.bb_middle[-1]

            if price >= tp_level:
                print(f"🌙✨ TP HIT! Price {price:.2f} >= {tp_level:.2f}. Closing long 🚀")
                self.position.close()
                return

            if entry_bb_width and current_bb_width >= entry_bb_width * self.bb_expansion:
                print(f"🌙✨ BB EXPANSION! Width {current_bb_width:.4f} >= {entry_bb_width * self.bb_expansion:.4f}. Exiting 🚀")
                self.position.close()
                return

            return

        if self.macd_hist[-1] is None or np.isnan(self.macd_hist[-1]):
            return

        if self.vol_sma[-1] is None or np.isnan(self.vol_sma[-1]):
            return

        vol_spike = self.data.Volume[-1] > self.vol_sma[-1] * self.vol_mult
        if not vol_spike:
            return

        lookback = self.divergence_window
        recent_lows = self.data.Low[-lookback:]
        recent_macd = self.macd[-lookback:]

        if len(recent_lows) < lookback:
            return

        price_lows_idx = []
        for i in range(2, len(recent_lows) - 2):
            if (recent_lows[i] < recent_lows[i-1] and recent_lows[i] < recent_lows[i-2] and
                recent_lows[i] < recent_lows[i+1] and recent_lows[i] < recent_lows[i+2]):
                price_lows_idx.append(i)

        if len(price_lows_idx) < 2:
            return

        last_low_idx = price_lows_idx[-1]
        prev_low_idx = price_lows_idx[-2]

        price_lower_low = recent_lows[last_low_idx] < recent_lows[prev_low_idx]
        macd_higher_low = recent_macd[last_low_idx] > recent_macd[prev_low_idx]

        if not (price_lower_low and macd_higher_low):
            return

        if last_low_idx < len(recent_lows) - 5:
            return

        near_lower_bb = price <= self.bb_lower[-1] * 1.005
        if not near_lower_bb:
            return

        atr_val = self.atr[-1]
        if np.isnan(atr_val) or atr_val <= 0:
            return

        current_atr = atr_val
        avg_atr = np.nanmean(self.atr[-50:]) if len(self.atr) >= 50 else current_atr
        if avg_atr > 0:
            if current_atr < avg_atr * 0.3 or current_atr > avg_atr * 3.0:
                print(f"🌙 Skipping - ATR regime off: {current_atr:.2f} vs avg {avg_atr:.2f}")
                return

        stop_loss = self.swing_low[-1] - atr_val
        risk = price - stop_loss
        if risk <= 0:
            return

        risk_amount = self.equity * self.risk_pct
        position_size = int(round(risk_amount / risk))
        if position_size < 1:
            position_size = 1

        max_size = int(self.equity / price)
        if position_size > max_size:
            position_size = max_size

        if position_size < 1:
            return

        bb_width_entry = (self.bb_upper[-1] - self.bb_lower[-1]) / self.bb_middle[-1]

        print(f"🌙🚀 LONG SIGNAL! Divergence + Volume Spike. Price: {price:.2f}, Size: {position_size}, SL: {stop_loss:.2f}, ATR: {atr_val:.2f}")

        # 🌙 Record metadata BEFORE placing trade so entry_bar matches
        entry_bar = len(self.data) - 1
        self.atr_at_entry[entry_bar] = atr_val
        self.bb_width_at_entry[entry_bar] = bb_width_entry

        self.buy(size=position_size, sl=stop_loss)


data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = load_data(data_path)
print(f"🌙 Data loaded: {len(data)} bars ✨")

bt = Backtest(data, VolumetricDivergence, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙✨ Moon Dev backtest complete! 🚀")