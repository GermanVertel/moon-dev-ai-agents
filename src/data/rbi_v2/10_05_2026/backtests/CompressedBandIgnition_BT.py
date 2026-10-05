import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

print("🌙 Moon Dev's CompressedBand Ignition Backtest Loading... ✨")

data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')
print(f"🚀 Data loaded: {len(data)} bars")
print(f"🌙 Columns: {list(data.columns)}")


class CompressedBandIgnition(Strategy):
    bb_period = 20
    bb_std = 2.0
    bbw_lookback = 20
    adx_period = 14
    adx_squeeze = 20
    adx_exhaust = 40
    vol_mult = 1.5
    vol_period = 20
    setup_window = 10
    atr_period = 14
    atr_stop_mult = 2.0
    risk_pct = 0.01
    time_stop = 20

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS, close, timeperiod=self.bb_period,
            nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0
        )
        self.bbw = self.I(lambda u, l, m: (u - l) / m, self.bb_upper, self.bb_lower, self.bb_middle)
        self.bbw_prev = self.I(lambda x: np.concatenate([[np.nan], x[:-1]]), self.bbw)

        self.adx = self.I(talib.ADX, high, low, close, timeperiod=self.adx_period)
        self.pdi = self.I(talib.PLUS_DI, high, low, close, timeperiod=self.adx_period)
        self.mdi = self.I(talib.MINUS_DI, high, low, close, timeperiod=self.adx_period)

        self.vol_sma = self.I(talib.SMA, volume, timeperiod=self.vol_period)
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        # Rolling min of prior BBW values (strictly prior bars)
        self.bbw_min = self.I(
            lambda x: pd.Series(x).rolling(self.bbw_lookback).min().shift(1).values,
            self.bbw
        )

        self.squeeze_active = False
        self.squeeze_bar = -1
        self.trailing_stop = None
        self.entry_bar = -1
        self.entry_price = None
        self.trade_dir = None

        print("✨ Indicators initialized: BB, BBW, ADX, DI, VolSMA, ATR 🌙")

    def next(self):
        i = len(self.data) - 1
        if i < self.bb_period + self.bbw_lookback + 5:
            return

        close = self.data.Close[-1]
        upper = self.bb_upper[-1]
        lower = self.bb_lower[-1]
        bbw = self.bbw[-1]
        bbw_prev = self.bbw_prev[-1]
        bbw_min = self.bbw_min[-1]
        adx = self.adx[-1]
        pdi = self.pdi[-1]
        mdi = self.mdi[-1]
        vol = self.data.Volume[-1]
        vol_sma = self.vol_sma[-1]
        atr = self.atr[-1]

        if any(np.isnan([bbw, bbw_min, adx, pdi, mdi, vol_sma, atr])):
            return

        # --- Squeeze detection ---
        if bbw <= bbw_min and adx < self.adx_squeeze:
            if not self.squeeze_active:
                print(f"🌙 SQUEEZE DETECTED at bar {i} | BBW={bbw:.4f} ADX={adx:.2f} ✨")
            self.squeeze_active = True
            self.squeeze_bar = i

        # --- Manage open position ---
        if self.position:
            # Update trailing stop
            if self.trade_dir == 'long':
                new_stop = close - self.atr_stop_mult * atr
                if self.trailing_stop is None or new_stop > self.trailing_stop:
                    self.trailing_stop = new_stop
                stop_hit = close <= self.trailing_stop
                band_exit = close < upper
                adx_exhaust = adx > self.adx_exhaust
                time_exit = (i - self.entry_bar) >= self.time_stop
                if stop_hit or band_exit or adx_exhaust or time_exit:
                    reason = 'stop' if stop_hit else 'band' if band_exit else 'adx' if adx_exhaust else 'time'
                    self.position.close()
                    print(f"🚀 EXIT LONG at {close:.2f} | reason={reason} 🌙")
                    self.squeeze_active = False
                    self.trailing_stop = None
                    self.trade_dir = None
            else:
                new_stop = close + self.atr_stop_mult * atr
                if self.trailing_stop is None or new_stop < self.trailing_stop:
                    self.trailing_stop = new_stop
                stop_hit = close >= self.trailing_stop
                band_exit = close > lower
                adx_exhaust = adx > self.adx_exhaust
                time_exit = (i - self.entry_bar) >= self.time_stop
                if stop_hit or band_exit or adx_exhaust or time_exit:
                    reason = 'stop' if stop_hit else 'band' if band_exit else 'adx' if adx_exhaust else 'time'
                    self.position.close()
                    print(f"🚀 EXIT SHORT at {close:.2f} | reason={reason} 🌙")
                    self.squeeze_active = False
                    self.trailing_stop = None
                    self.trade_dir = None
            return

        # --- Trigger check within setup window ---
        if self.squeeze_active and (i - self.squeeze_bar) <= self.setup_window:
            expanding = bbw > bbw_prev
            vol_ok = vol > self.vol_mult * vol_sma
            long_trigger = expanding and close > upper and vol_ok and pdi > mdi
            short_trigger = expanding and close < lower and vol_ok and mdi > pdi

            if long_trigger or short_trigger:
                # Position sizing: risk 1% of equity, stop = 2*ATR
                stop_dist = self.atr_stop_mult * atr
                if stop_dist <= 0:
                    return
                risk_amount = self.equity * self.risk_pct
                size = int(round(risk_amount / stop_dist))
                if size < 1:
                    return

                if long_trigger:
                    self.buy(size=size)
                    self.trailing_stop = close - stop_dist
                    self.trade_dir = 'long'
                    self.entry_bar = i
                    self.entry_price = close
                    print(f"🚀 LONG IGNITION at {close:.2f} | size={size} BBW={bbw:.4f} ADX={adx:.2f} Vol={vol:.0f} 🌙")
                else:
                    self.sell(size=size)
                    self.trailing_stop = close + stop_dist
                    self.trade_dir = 'short'
                    self.entry_bar = i
                    self.entry_price = close
                    print(f"🚀 SHORT IGNITION at {close:.2f} | size={size} BBW={bbw:.4f} ADX={adx:.2f} Vol={vol:.0f} 🌙")

                self.squeeze_active = False
        else:
            if self.squeeze_active and (i - self.squeeze_bar) > self.setup_window:
                self.squeeze_active = False


bt = Backtest(data, CompressedBandIgnition, cash=1_000_000, commission=0.0002)
stats = bt.run()
print(stats)
print(stats._strategy)