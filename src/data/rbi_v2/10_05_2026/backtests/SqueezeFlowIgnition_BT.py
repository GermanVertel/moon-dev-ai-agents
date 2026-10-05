import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Load data
data = pd.read_csv('/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv')

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

# Set datetime index
data['Datetime'] = pd.to_datetime(data['Datetime'])
data = data.set_index('Datetime')

print("🌙✨ Moon Dev SqueezeFlow Ignition Backtest Loading... 🚀")
print(f"📊 Data shape: {data.shape}")
print(f"📅 Date range: {data.index[0]} to {data.index[-1]}")
print(f"💰 Price range: {data['Close'].min():.2f} - {data['Close'].max():.2f}")
print("=" * 60)


class SqueezeFlowIgnition(Strategy):
    """
    SqueezeFlow Ignition Strategy 🌙
    Combines Bollinger Band squeeze filter with CMF zero-cross trigger.
    """
    # Parameters
    bb_period = 20
    bb_std = 2.0
    bw_median_period = 100
    squeeze_threshold = 0.5
    cmf_period = 20
    atr_period = 14
    atr_stop_mult = 2.0
    risk_pct = 0.02
    time_stop_bars = 10

    def init(self):
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        # Bollinger Bands
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS, close,
            timeperiod=self.bb_period,
            nbdevup=self.bb_std,
            nbdevdn=self.bb_std,
            matype=0,
            name='BB'
        )

        # Bandwidth = (Upper - Lower) / Middle
        def bandwidth(upper, lower, middle):
            return (upper - lower) / middle

        self.bandwidth = self.I(
            bandwidth,
            self.bb_upper, self.bb_lower, self.bb_middle,
            name='Bandwidth'
        )

        # 100-period rolling median of bandwidth
        def rolling_median(arr, period):
            return pd.Series(arr).rolling(period).median().values

        self.bw_median = self.I(
            rolling_median,
            self.bandwidth, self.bw_median_period,
            name='BW_Median'
        )

        # Chaikin Money Flow
        def cmf(high, low, close, volume, period):
            mfm = ((close - low) - (high - close)) / (high - low)
            mfm = np.where((high - low) == 0, 0, mfm)
            mfv = mfm * volume
            cmf_val = pd.Series(mfv).rolling(period).sum() / pd.Series(volume).rolling(period).sum()
            return cmf_val.values

        self.cmf = self.I(
            cmf,
            high, low, close, volume, self.cmf_period,
            name='CMF'
        )

        # ATR for stop sizing
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period, name='ATR')

        # Trend filter: 50 SMA
        self.sma50 = self.I(talib.SMA, close, timeperiod=50, name='SMA50')

        # State tracking
        self.entry_bar = None
        self.stop_price = None

        print("🌙 Indicators initialized successfully ✨")

    def next(self):
        if len(self.data) < max(self.bw_median_period, self.cmf_period) + 5:
            return

        price = self.data.Close[-1]
        upper = self.bb_upper[-1]
        lower = self.bb_lower[-1]
        middle = self.bb_middle[-1]
        bw = self.bandwidth[-1]
        bw_med = self.bw_median[-1]
        cmf_now = self.cmf[-1]
        cmf_prev = self.cmf[-2]
        atr_now = self.atr[-1]

        # Handle NaN
        if any(np.isnan([upper, lower, middle, bw, bw_med, cmf_now, cmf_prev, atr_now])):
            return

        # =========================
        # EXITS
        # =========================
        if self.position:
            # Primary exit: CMF crosses below zero
            if cmf_prev > 0 and cmf_now < 0:
                print(f"🌙 CMF reversal exit at {price:.2f} | CMF: {cmf_now:.4f} 🔻")
                self.position.close()
                self.entry_bar = None
                self.stop_price = None
                return

            # Secondary exit: overextension above upper band
            if price > upper + 0.5 * (upper - lower):
                print(f"🚀 Overextension exit at {price:.2f} | Upper: {upper:.2f} 💰")
                self.position.close()
                self.entry_bar = None
                self.stop_price = None
                return

            # Stop loss
            if self.stop_price is not None and price <= self.stop_price:
                print(f"🛑 Stop loss hit at {price:.2f} | Stop: {self.stop_price:.2f}")
                self.position.close()
                self.entry_bar = None
                self.stop_price = None
                return

            # Time stop
            if self.entry_bar is not None and (len(self.data) - self.entry_bar) >= self.time_stop_bars:
                print(f"⏰ Time stop exit at {price:.2f} after {self.time_stop_bars} bars")
                self.position.close()
                self.entry_bar = None
                self.stop_price = None
                return

        # =========================
        # ENTRY
        # =========================
        if not self.position:
            # Squeeze filter
            squeeze_active = bw < self.squeeze_threshold * bw_med

            # CMF cross above zero
            cmf_cross_up = cmf_prev <= 0 and cmf_now > 0

            # Trend filter
            trend_ok = price > middle and price > self.sma50[-1]

            if squeeze_active and cmf_cross_up and trend_ok:
                # Position sizing: risk-based using ATR
                risk_per_unit = self.atr_stop_mult * atr_now
                if risk_per_unit <= 0:
                    return

                equity = self.equity
                risk_amount = equity * self.risk_pct
                position_size = risk_amount / risk_per_unit
                position_size = int(round(position_size))

                if position_size < 1:
                    position_size = 1

                self.stop_price = price - risk_per_unit

                print(f"🌙✨ SQUEEZE IGNITION! Entry at {price:.2f} 🚀")
                print(f"   📊 BW: {bw:.4f} < {self.squeeze_threshold * bw_med:.4f} (median)")
                print(f"   💰 CMF: {cmf_prev:.4f} → {cmf_now:.4f} (crossed zero)")
                print(f"   📈 SMA50: {self.sma50[-1]:.2f} | Middle BB: {middle:.2f}")
                print(f"   🛑 Stop: {self.stop_price:.2f} | Size: {position_size}")

                self.buy(size=position_size)
                self.entry_bar = len(self.data)


# Run backtest
print("🌙 Starting Moon Dev SqueezeFlow Ignition Backtest... 🚀")
bt = Backtest(
    data,
    SqueezeFlowIgnition,
    cash=1_000_000,
    commission=0.002,
    exclusive_orders=True
)

stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙✨ Backtest complete! 🚀")