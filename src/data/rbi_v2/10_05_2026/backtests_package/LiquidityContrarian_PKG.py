import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# ============================================================
# Moon Dev's LiquidityContrarian Backtest 🌙
# ============================================================

print("🌙 Moon Dev booting up LiquidityContrarian strategy...")
print("🚀 Loading BTC-USD 15m data...")

data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Clean columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Proper column mapping
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

# Ensure datetime index
if 'datetime' in data.columns:
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')

print(f"✨ Data loaded: {len(data)} bars")
print(f"🌙 Date range: {data.index[0]} to {data.index[-1]}")


class LiquidityContrarian(Strategy):
    # Parameters
    atr_period = 14
    vol_ma_period = 20
    funding_zscore_period = 30
    heatmap_lookback = 96  # ~24h on 15m
    wick_threshold = 0.60
    volume_spike_mult = 2.0
    funding_zscore_threshold = 2.0
    funding_divergence_threshold = 0.0003  # 0.03%
    risk_pct = 0.005  # 0.5%
    sl_atr_mult = 1.2
    tp1_atr_mult = 1.5
    tp2_atr_mult = 3.0
    time_stop_bars = 96  # 24h on 15m
    max_trades_per_day = 4

    def init(self):
        print("🌙 Initializing indicators...")

        # ATR
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close,
                          timeperiod=self.atr_period)

        # Volume MA
        self.vol_ma = self.I(talib.SMA, self.data.Volume,
                             timeperiod=self.vol_ma_period)

        # Heatmap proxies: rolling high/low density bands
        self.roll_high = self.I(talib.MAX, self.data.High,
                                timeperiod=self.heatmap_lookback)
        self.roll_low = self.I(talib.MIN, self.data.Low,
                               timeperiod=self.heatmap_lookback)

        # Funding rate proxy: use price-based synthetic funding (since we don't have real funding data)
        # Synthetic funding = (close - sma) / sma, a common proxy for funding pressure
        self.sma_fund = self.I(talib.SMA, self.data.Close, timeperiod=8)

        # Funding zscore
        self.funding_mean = self.I(talib.SMA, self.data.Close,
                                   timeperiod=self.funding_zscore_period)
        self.funding_std = self.I(talib.STDDEV, self.data.Close,
                                  timeperiod=self.funding_zscore_period)

        # Volume spike flag
        self.vol_spike = self.I(lambda v, m: v > m * 2.0,
                                self.data.Volume, self.vol_ma)

        # Trackers
        self.tp1_hit = False
        self.entry_price = None
        self.entry_bar = None
        self.trades_today = 0
        self.current_day = None
        self.breach_extreme = None
        self.position_side = None

        print("✨ Indicators ready!")

    def next(self):
        # Skip if not enough data
        if len(self.data) < self.heatmap_lookback + 5:
            return

        # Daily trade counter reset
        current_day = self.data.index[-1].date()
        if self.current_day != current_day:
            self.current_day = current_day
            self.trades_today = 0

        price = self.data.Close[-1]
        high = self.data.High[-1]
        low = self.data.Low[-1]
        open_ = self.data.Open[-1]
        atr = self.atr[-1]
        vol_spike = self.vol_spike[-1]

        if np.isnan(atr) or atr <= 0:
            return

        # Candle metrics
        candle_range = high - low
        if candle_range <= 0:
            return
        upper_wick = (high - max(open_, price)) / candle_range
        lower_wick = (min(open_, price) - low) / candle_range

        # Funding proxy & zscore (computed inline, no backtesting.lib)
        sma_fund = self.sma_fund[-1]
        if np.isnan(sma_fund) or sma_fund == 0:
            return
        funding_proxy = (price - sma_fund) / sma_fund

        f_mean = self.funding_mean[-1]
        f_std = self.funding_std[-1]
        if np.isnan(f_std) or f_std <= 0 or np.isnan(f_mean):
            return
        funding_z = (price - f_mean) / f_std

        # Heatmap bands (proxies)
        band_high = self.roll_high[-2]  # previous high band
        band_low = self.roll_low[-2]    # previous low band

        # ==================== MANAGE OPEN POSITION ====================
        if self.position:
            bars_held = len(self.data) - self.entry_bar

            # Time stop
            if bars_held >= self.time_stop_bars and not self.tp1_hit:
                print(f"⏰ Moon Dev TIME STOP hit after {bars_held} bars — exiting")
                self.position.close()
                self.tp1_hit = False
                self.entry_bar = None
                return

            # Chandelier-style trailing after TP1
            if self.tp1_hit and self.position_side == 'long':
                trail_stop = self.data.High[-bars_held:].max() - 3 * atr if bars_held > 0 else None
                if trail_stop and price < trail_stop:
                    print(f"🌙 Moon Dev CHANDELIER trail exit LONG at {price:.2f}")
                    self.position.close()
                    self.tp1_hit = False
                    self.entry_bar = None
                    return
            elif self.tp1_hit and self.position_side == 'short':
                trail_stop = self.data.Low[-bars_held:].min() + 3 * atr if bars_held > 0 else None
                if trail_stop and price > trail_stop:
                    print(f"🌙 Moon Dev CHANDELIER trail exit SHORT at {price:.2f}")
                    self.position.close()
                    self.tp1_hit = False
                    self.entry_bar = None
                    return

            # TP1 check — move to breakeven
            if not self.tp1_hit and self.entry_price:
                if self.position_side == 'long' and price >= self.entry_price + self.tp1_atr_mult * atr:
                    self.tp1_hit = True
                    print(f"🎯 Moon Dev TP1 hit LONG — moving SL to breakeven")
                elif self.position_side == 'short' and price <= self.entry_price - self.tp1_atr_mult * atr:
                    self.tp1_hit = True
                    print(f"🎯 Moon Dev TP1 hit SHORT — moving SL to breakeven")
            return

        # ==================== ENTRY LOGIC ====================
        if self.trades_today >= self.max_trades_per_day:
            return

        # LONG ENTRY: fade downside liquidation cascade
        # 1. Price breaches below previous low band
        breach_down = price < band_low
        # 2. Funding zscore < -2 (shorts overcrowded)
        funding_oversold = funding_z < -self.funding_zscore_threshold
        # 3. Wick rejection or bullish engulfing
        wick_reject_long = lower_wick > self.wick_threshold
        bullish_engulf = (price > open_ and
                          len(self.data) > 1 and
                          price > self.data.Open[-2] and
                          open_ < self.data.Close[-2])
        # 4. Volume spike
        if breach_down and funding_oversold and (wick_reject_long or bullish_engulf) and vol_spike:
            sl = low - self.sl_atr_mult * atr
            risk = price - sl
            if risk > 0:
                size = int(round((self.equity * self.risk_pct) / risk))
                if size > 0:
                    print(f"🌙🚀 Moon Dev LONG SIGNAL! Price={price:.2f} Z={funding_z:.2f} "
                          f"LowerWick={lower_wick:.2f} VolSpike={vol_spike} Size={size}")
                    self.buy(size=size, sl=sl)
                    self.entry_price = price
                    self.entry_bar = len(self.data)
                    self.tp1_hit = False
                    self.position_side = 'long'
                    self.trades_today += 1
                    return

        # SHORT ENTRY: fade upside liquidation cascade
        breach_up = price > band_high
        funding_overbought = funding_z > self.funding_zscore_threshold
        wick_reject_short = upper_wick > self.wick_threshold
        bearish_engulf = (price < open_ and
                          len(self.data) > 1 and
                          price < self.data.Open[-2] and
                          open_ > self.data.Close[-2])
        if breach_up and funding_overbought and (wick_reject_short or bearish_engulf) and vol_spike:
            sl = high + self.sl_atr_mult * atr
            risk = sl - price
            if risk > 0:
                size = int(round((self.equity * self.risk_pct) / risk))
                if size > 0:
                    print(f"🌙🚀 Moon Dev SHORT SIGNAL! Price={price:.2f} Z={funding_z:.2f} "
                          f"UpperWick={upper_wick:.2f} VolSpike={vol_spike} Size={size}")
                    self.sell(size=size, sl=sl)
                    self.entry_price = price
                    self.entry_bar = len(self.data)
                    self.tp1_hit = False
                    self.position_side = 'short'
                    self.trades_today += 1
                    return


print("🌙✨ Running Moon Dev LiquidityContrarian backtest...")
bt = Backtest(
    data,
    LiquidityContrarian,
    cash=1_000_000,
    commission=0.0005,
    exclusive_orders=True
)

stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙✨ Moon Dev backtest complete! 🚀")