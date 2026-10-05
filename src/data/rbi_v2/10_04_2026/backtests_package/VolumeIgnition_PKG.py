import pandas as pd
import numpy as np
import talib
from backtesting import Backtest, Strategy

# Load data
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper column mapping
data.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')
data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

print("🌙 Moon Dev VolumeIgnition Backtest Initializing... ✨")
print(f"📊 Data loaded: {len(data)} bars")
print(f"🚀 Starting backtest...")

class VolumeIgnition(Strategy):
    bb_period = 20
    bb_std = 2.0
    vol_period = 50
    vol_threshold = 1.5  # 150% of average volume
    atr_period = 14
    risk_pct = 0.01  # 1% risk per trade
    max_hold_mult = 2.0  # 2 * ATR bars max holding

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        # Bollinger Bands
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS, close, timeperiod=self.bb_period, nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0
        )

        # Volume average
        self.vol_avg = self.I(talib.SMA, volume, timeperiod=self.vol_period)

        # ATR
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period)

        # Track entry info
        self.entry_bar = None
        self.entry_price = None
        self.stop_price = None
        self.target_price = None
        self.max_hold_bars = None
        self.breakeven_moved = False

        print("🌙 Indicators initialized: BB, Volume SMA, ATR ✨")

    def next(self):
        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        volume = self.data.Volume[-1]

        ub = self.bb_upper[-1]
        mb = self.bb_middle[-1]
        lb = self.bb_lower[-1]
        vol_avg = self.vol_avg[-1]
        atr = self.atr[-1]

        # Skip if indicators not ready
        if np.isnan(ub) or np.isnan(vol_avg) or np.isnan(atr) or atr <= 0:
            return

        # ========== EXIT LOGIC ==========
        if self.position:
            bars_held = len(self.data) - 1 - self.entry_bar

            # Time stop: max holding period
            if bars_held >= self.max_hold_bars:
                print(f"⏰ Time stop hit at bar {len(self.data)} | Held {bars_held} bars | Exit @ {price:.2f}")
                self.position.close()
                return

            # Primary exit: close <= lower band
            if price <= lb:
                print(f"🔻 Lower band reversion exit @ {price:.2f} | LB={lb:.2f} | PnL: {self.position.pl:.2f}")
                self.position.close()
                return

            # Trailing stop: move to breakeven after 1 ATR profit
            if not self.breakeven_moved and price >= self.entry_price + atr:
                self.stop_price = max(self.stop_price, self.entry_price)
                self.breakeven_moved = True
                print(f"🛡️ Stop moved to breakeven @ {self.entry_price:.2f}")

            # Trail by middle band
            if self.breakeven_moved:
                self.stop_price = max(self.stop_price, mb)

            # Stop loss hit
            if low <= self.stop_price:
                print(f"🛑 Stop loss hit @ {self.stop_price:.2f} | PnL: {self.position.pl:.2f}")
                self.position.close()
                return

            # Optional profit target: 1 ATR above entry
            if self.target_price and high >= self.target_price:
                print(f"🎯 Profit target hit @ {self.target_price:.2f}")
                self.position.close()
                return

        # ========== ENTRY LOGIC ==========
        else:
            # Entry: close > upper band AND volume > 150% of avg
            high_volume = volume >= self.vol_threshold * vol_avg
            band_breakout = price > ub

            # Bullish candle confirmation: close in upper 25% of range
            bar_range = high - low
            bullish_body = (price - low) >= 0.75 * bar_range if bar_range > 0 else False

            if band_breakout and high_volume and bullish_body:
                # Risk-based position sizing
                stop = min(mb, price - 1.5 * atr)
                risk_per_unit = price - stop
                if risk_per_unit <= 0:
                    return

                risk_amount = self.equity * self.risk_pct
                position_size = int(round(risk_amount / risk_per_unit))

                if position_size < 1:
                    position_size = 1

                # Cap at available equity
                max_size = int(self.equity / price)
                position_size = min(position_size, max_size)

                if position_size < 1:
                    return

                self.entry_bar = len(self.data) - 1
                self.entry_price = price
                self.stop_price = stop
                self.target_price = price + atr
                self.max_hold_bars = max(1, int(round(self.max_hold_mult * atr)))
                self.breakeven_moved = False

                self.buy(size=position_size)
                print(f"🚀 VOLUME IGNITION! Entry @ {price:.2f} | Size: {position_size} | Stop: {stop:.2f} | Target: {self.target_price:.2f} | MaxHold: {self.max_hold_bars} bars")


# Run backtest
bt = Backtest(data, VolumeIgnition, cash=1_000_000, commission=0.001)
stats = bt.run()
print(stats)
print(stats._strategy)