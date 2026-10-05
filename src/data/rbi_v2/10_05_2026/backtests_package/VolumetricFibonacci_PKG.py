import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# 🌙 Moon Dev's VolumetricFibonacci Backtest 🌙

def load_data(path):
    print("🌙 Loading data from Moon Dev's secret vault...")
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
    print(f"✨ Data loaded: {len(data)} bars of cosmic price action 🚀")
    return data


class VolumetricFibonacci(Strategy):
    # Strategy parameters
    rsi_period = 14
    rsi_overbought = 70
    rsi_oversold = 30
    bb_period = 20
    bb_std = 2.0
    vol_ma_period = 20
    vol_mult = 1.5
    bbw_lookback = 100
    bbw_percentile = 85
    swing_lookback = 20
    atr_period = 14
    atr_buffer = 0.5
    fib_target = 0.382
    time_stop_bars = 30
    risk_pct = 0.02

    def init(self):
        print("🌙 Initializing Moon Dev's VolumetricFibonacci indicators...")
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        # RSI
        self.rsi = self.I(talib.RSI, close, timeperiod=self.rsi_period, name='RSI')

        # Bollinger Bands
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS, close, timeperiod=self.bb_period,
            nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0,
            name='BB'
        )

        # Bollinger Bandwidth
        def bbw_calc(upper, middle, lower):
            return (upper - lower) / middle
        self.bbw = self.I(bbw_calc, self.bb_upper, self.bb_middle, self.bb_lower, name='BBW')

        # BBW rolling threshold (percentile)
        def bbw_threshold(bbw_series):
            return pd.Series(bbw_series).rolling(self.bbw_lookback).quantile(self.bbw_percentile / 100.0).values
        self.bbw_thresh = self.I(bbw_threshold, self.bbw, name='BBW_Thresh')

        # Volume MA
        self.vol_ma = self.I(talib.SMA, volume, timeperiod=self.vol_ma_period, name='VolMA')

        # ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period, name='ATR')

        # Swing highs/lows
        self.swing_high = self.I(talib.MAX, high, timeperiod=self.swing_lookback, name='SwingHigh')
        self.swing_low = self.I(talib.MIN, low, timeperiod=self.swing_lookback, name='SwingLow')

        self.entry_bar = None
        print("✨ Indicators ready! May the Moon guide our trades 🌙")

    def next(self):
        price = self.data.Close[-1]

        # Handle open position exits
        if self.position:
            bars_held = len(self.data) - self.entry_bar

            # Time stop
            if bars_held >= self.time_stop_bars:
                print(f"⏰ Moon Dev time stop hit at {price:.2f} after {bars_held} bars")
                self.position.close()
                return

            # Check TP / SL via bracket orders (handled automatically), but also manual fib check
            if self.position.is_short:
                # Fib retracement target from swing low to swing high
                sh = self.swing_high[-1]
                sl = self.swing_low[-1]
                fib_target_price = sh - (sh - sl) * self.fib_target
                if price <= fib_target_price:
                    print(f"🎯 SHORT fib target hit at {price:.2f} (target {fib_target_price:.2f})")
                    self.position.close()
                    return
            elif self.position.is_long:
                sh = self.swing_high[-1]
                sl = self.swing_low[-1]
                fib_target_price = sl + (sh - sl) * self.fib_target
                if price >= fib_target_price:
                    print(f"🎯 LONG fib target hit at {price:.2f} (target {fib_target_price:.2f})")
                    self.position.close()
                    return
            return

        # Entry logic
        if len(self.data) < max(self.bbw_lookback, self.swing_lookback, self.vol_ma_period) + 5:
            return

        rsi = self.rsi[-1]
        vol = self.data.Volume[-1]
        vol_ma = self.vol_ma[-1]
        bbw = self.bbw[-1]
        bbw_thresh = self.bbw_thresh[-1]

        if np.isnan(rsi) or np.isnan(vol_ma) or np.isnan(bbw) or np.isnan(bbw_thresh):
            return

        vol_surge = vol > (self.vol_mult * vol_ma)
        bbw_expansion = bbw > bbw_thresh

        if not (vol_surge and bbw_expansion):
            return

        atr = self.atr[-1]
        if np.isnan(atr) or atr <= 0:
            return

        sh = self.swing_high[-1]
        sl = self.swing_low[-1]

        # Risk sizing: 1M units base
        size = 1000000

        # SHORT: overbought exhaustion
        if rsi > self.rsi_overbought:
            stop_price = sh + self.atr_buffer * atr
            risk_per_unit = stop_price - price
            if risk_per_unit <= 0:
                return
            print(f"🌙 SHORT signal! RSI={rsi:.1f} Vol={vol:.0f} BBW={bbw:.4f} > {bbw_thresh:.4f} | Entry={price:.2f} SL={stop_price:.2f}")
            self.sell(size=size, sl=stop_price)
            self.entry_bar = len(self.data)

        # LONG: oversold exhaustion
        elif rsi < self.rsi_oversold:
            stop_price = sl - self.atr_buffer * atr
            risk_per_unit = price - stop_price
            if risk_per_unit <= 0:
                return
            print(f"🌙 LONG signal! RSI={rsi:.1f} Vol={vol:.0f} BBW={bbw:.4f} > {bbw_thresh:.4f} | Entry={price:.2f} SL={stop_price:.2f}")
            self.buy(size=size, sl=stop_price)
            self.entry_bar = len(self.data)


if __name__ == '__main__':
    data_path = '/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv'
    data = load_data(data_path)

    print("🚀 Launching Moon Dev's VolumetricFibonacci backtest...")
    bt = Backtest(
        data,
        VolumetricFibonacci,
        cash=1000000,
        commission=0.0002,
        exclusive=False
    )

    stats = bt.run()
    print(stats)
    print(stats._strategy)
    print("🌙✨ Backtest complete! Moon Dev out 🚀")