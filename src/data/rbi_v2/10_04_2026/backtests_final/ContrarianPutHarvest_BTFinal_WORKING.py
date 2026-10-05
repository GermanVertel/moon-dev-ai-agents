import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# ============================================================
# 🌙 Moon Dev's ContrarianPutHarvest Backtest 🌙
# ============================================================

print("🌙✨ Moon Dev Backtest Engine Booting Up... ✨🌙")

# Load data
data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
print(f"🚀 Loading data from: {data_path}")
data = pd.read_csv(data_path)

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Ensure proper column mapping
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

# Set datetime index
if 'datetime' in data.columns:
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')

print(f"✅ Data loaded: {len(data)} bars")
print(f"📊 Columns: {list(data.columns)}")


class ContrarianPutHarvest(Strategy):
    """
    🌙 ContrarianPutHarvest Strategy 🌙
    
    Proxy implementation using futures data:
    - Sell puts proxy = go LONG when oversold + high volume (OI proxy)
    - Exit when volume inflection (OI proxy) rises / momentum shifts
    - Trend filter: price below EMA = downtrend = premium rich environment
    """

    # Parameters
    ema_period = 20
    oi_lookback = 60
    oi_percentile = 90
    iv_rank_threshold = 50
    profit_target_pct = 0.60  # 60% of premium
    stop_loss_pct = 0.30      # Stop out
    time_exit_bars = 20       # ~5 hours on 15m = time exit proxy

    def init(self):
        print("🌙 Initializing indicators...")
        self.ema20 = self.I(talib.EMA, self.data.Close, timeperiod=self.ema_period)
        self.ema50 = self.I(talib.EMA, self.data.Close, timeperiod=50)
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=14)
        # Open Interest proxy = Volume
        self.vol_sma = self.I(talib.SMA, self.data.Volume, timeperiod=20)
        self.vol_max = self.I(talib.MAX, self.data.Volume, timeperiod=self.oi_lookback)
        self.vol_min = self.I(talib.MIN, self.data.Volume, timeperiod=self.oi_lookback)
        # Volatility proxy for IV rank
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14)
        self.atr_sma = self.I(talib.SMA, self.atr, timeperiod=50)
        print("✅ Indicators ready!")

    def next(self):
        price = self.data.Close[-1]
        ema20 = self.ema20[-1]
        ema50 = self.ema50[-1]
        rsi = self.rsi[-1]
        vol = self.data.Volume[-1]
        vol_sma = self.vol_sma[-1]
        vol_max = self.vol_max[-1]
        vol_min = self.vol_min[-1]
        atr = self.atr[-1]
        atr_sma = self.atr_sma[-1]

        # Avoid NaN
        if np.isnan(ema20) or np.isnan(vol_max) or np.isnan(atr_sma) or vol_max == vol_min:
            return

        # OI percentile proxy
        vol_pct = (vol - vol_min) / (vol_max - vol_min) * 100

        # IV rank proxy (ATR relative to its average)
        iv_rank = (atr / atr_sma) * 50 if atr_sma > 0 else 0

        # === ENTRY CONDITIONS ===
        downtrend = price < ema20 and ema20 < ema50
        high_oi = vol_pct > self.oi_percentile
        premium_rich = iv_rank > self.iv_rank_threshold
        oversold = rsi < 40

        if not self.position:
            if downtrend and high_oi and premium_rich and oversold:
                print(f"🌙✨ ENTRY SIGNAL @ {price:.2f} | RSI={rsi:.1f} | VolPct={vol_pct:.1f}% | IVR={iv_rank:.1f}")
                # Position size as fraction of equity (0 < size < 1)
                self.buy(size=0.95)
                self.entry_price = price
                self.entry_bar = len(self.data)

        else:
            # === EXIT CONDITIONS ===
            pnl_pct = (price - self.entry_price) / self.entry_price
            bars_held = len(self.data) - self.entry_bar

            # OI inflection exit: volume rising above SMA after plateau
            oi_inflection = vol > vol_sma * 1.3 and vol_pct > 50

            # Take profit at 60% of "premium" (proxy = price move)
            take_profit = pnl_pct >= self.profit_target_pct * 0.05

            # Stop loss
            stop_loss = pnl_pct <= -self.stop_loss_pct * 0.05

            # Time exit
            time_exit = bars_held >= self.time_exit_bars

            if oi_inflection:
                print(f"🌙 OI INFLECTION EXIT @ {price:.2f} | PnL={pnl_pct*100:.2f}% | VolPct={vol_pct:.1f}%")
                self.position.close()
            elif take_profit:
                print(f"🎯 PROFIT TARGET HIT @ {price:.2f} | PnL={pnl_pct*100:.2f}%")
                self.position.close()
            elif stop_loss:
                print(f"🛑 STOP LOSS @ {price:.2f} | PnL={pnl_pct*100:.2f}%")
                self.position.close()
            elif time_exit:
                print(f"⏰ TIME EXIT @ {price:.2f} | PnL={pnl_pct*100:.2f}% | Bars={bars_held}")
                self.position.close()


print("🌙 Running ContrarianPutHarvest Backtest...")
bt = Backtest(
    data,
    ContrarianPutHarvest,
    cash=1000000,
    commission=0.002,
)

stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙✨ Backtest Complete! ✨🌙")