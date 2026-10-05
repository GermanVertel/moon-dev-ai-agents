import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# 🌙 Moon Dev's VolatilityCliffHarvester 🌙
# Data path
DATA_PATH = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"

# Load and clean data
data = pd.read_csv(DATA_PATH)
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
# Map to proper case
data.columns = [c.capitalize() for c in data.columns]
# Ensure datetime index
if 'Datetime' in data.columns:
    data['Datetime'] = pd.to_datetime(data['Datetime'])
    data.set_index('Datetime', inplace=True)
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
data.dropna(inplace=True)

print(f"🌙 Data loaded: {len(data)} bars | columns: {list(data.columns)}")


class VolatilityCliffHarvester(Strategy):
    # 🌙 Strategy parameters
    rv_short = 5          # short realized vol window (bars)
    rv_long = 20          # long realized vol window (bars)
    vol_of_vol_window = 20
    cliff_threshold = 0.50   # |ΔRV|/RV_prior > 50%
    spread_z_threshold = 2.0
    exit_z_threshold = 0.5
    stop_z_threshold = 1.5
    time_stop_bars = 40      # ~10 hours on 15m bars
    risk_per_trade = 0.01    # 1% of equity

    def init(self):
        # 🌙 Compute log returns
        close = pd.Series(self.data.Close)
        log_ret = np.log(close / close.shift(1)).fillna(0)

        # 🌙 Realized volatility (short & long horizons)
        self.rv_short_ind = self.I(
            lambda x: pd.Series(x).rolling(self.rv_short).std() * np.sqrt(252 * 96),
            log_ret
        )
        self.rv_long_ind = self.I(
            lambda x: pd.Series(x).rolling(self.rv_long).std() * np.sqrt(252 * 96),
            log_ret
        )

        # 🌙 IV proxy: use ATR-based annualized vol as stand-in for implied vol
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14)
        self.iv_proxy = self.I(
            lambda a, c: (a / c) * np.sqrt(252 * 96),
            self.atr, self.data.Close
        )

        # 🌙 IV-RV spread
        self.spread = self.I(lambda iv, rv: iv - rv, self.iv_proxy, self.rv_short_ind)

        # 🌙 Rolling stats of spread
        self.spread_mean = self.I(
            lambda s: pd.Series(s).rolling(self.rv_long).mean(), self.spread
        )
        self.spread_std = self.I(
            lambda s: pd.Series(s).rolling(self.rv_long).std(), self.spread
        )

        # 🌙 Vol-of-vol: rolling std of IV changes
        iv_series = pd.Series(self.iv_proxy)
        iv_chg = iv_series.diff().fillna(0)
        self.vov = self.I(
            lambda x: pd.Series(x).rolling(self.vol_of_vol_window).std(),
            iv_chg
        )

        # 🌙 CUSUM changepoint detector on RV
        rv_series = pd.Series(self.rv_short_ind)
        rv_mean = rv_series.rolling(self.rv_long).mean()
        rv_std = rv_series.rolling(self.rv_long).std().replace(0, np.nan)
        z = ((rv_series - rv_mean) / rv_std).fillna(0)
        self.cusum = self.I(
            lambda zz: pd.Series(zz).cumsum(), z
        )

        # 🌙 Delta RV for cliff detection
        self.delta_rv = self.I(
            lambda rv: pd.Series(rv).diff().abs() / pd.Series(rv).shift(1).replace(0, np.nan),
            self.rv_short_ind
        )

        # 🌙 Track entry bar for time stop
        self.entry_bar = None
        self.entry_spread = None

        print("🌙✨ VolatilityCliffHarvester initialized! Ready to hunt cliffs 🚀")

    def next(self):
        # Need enough bars
        if len(self.data) < self.rv_long + 5:
            return

        price = self.data.Close[-1]
        rv_s = self.rv_short_ind[-1]
        rv_l = self.rv_long_ind[-1]
        iv = self.iv_proxy[-1]
        spread = self.spread[-1]
        s_mean = self.spread_mean[-1]
        s_std = self.spread_std[-1]
        vov = self.vov[-1]
        delta_rv = self.delta_rv[-1]

        if np.isnan(spread) or np.isnan(s_mean) or np.isnan(s_std) or s_std == 0:
            return

        spread_z = (spread - s_mean) / s_std

        # 🌙 Cliff detection
        cliff_fired = (not np.isnan(delta_rv)) and (delta_rv > self.cliff_threshold)

        # 🌙 Vol-of-vol regime filter
        vov_arr = np.array(self.vov)
        vov_valid = vov_arr[~np.isnan(vov_arr)]
        if len(vov_valid) > 50:
            vov_ok = (not np.isnan(vov)) and (vov < np.nanpercentile(vov_valid[-200:], 90))
        else:
            vov_ok = True

        # =====================
        # 🌙 EXIT LOGIC
        # =====================
        if self.position:
            bars_held = len(self.data) - self.entry_bar if self.entry_bar else 0

            # Convergence exit
            if abs(spread_z) <= self.exit_z_threshold:
                print(f"🌙✅ CONVERGENCE EXIT | spread_z={spread_z:.2f} | price={price:.2f}")
                self.position.close()
                self.entry_bar = None
                return

            # Hard stop: spread widens further
            if self.entry_spread is not None:
                if abs(spread_z) >= self.stop_z_threshold and np.sign(spread_z) == np.sign(self.entry_spread):
                    print(f"🌙🛑 HARD STOP | spread_z={spread_z:.2f} | price={price:.2f}")
                    self.position.close()
                    self.entry_bar = None
                    return

            # Time stop
            if bars_held >= self.time_stop_bars:
                print(f"🌙⏰ TIME STOP | bars={bars_held} | price={price:.2f}")
                self.position.close()
                self.entry_bar = None
                return

            # Vol-of-vol exit
            if not vov_ok:
                print(f"🌙⚠️ VOL-OF-VOL EXIT | vov={vov:.4f}")
                self.position.close()
                self.entry_bar = None
                return

            return

        # =====================
        # 🌙 ENTRY LOGIC
        # =====================
        if not cliff_fired:
            return
        if not vov_ok:
            return

        # 🌙 Risk-based position sizing
        equity = self.equity
        risk_amount = equity * self.risk_per_trade

        # Stop distance in price terms: approximate using ATR
        atr_val = self.atr[-1]
        if np.isnan(atr_val) or atr_val <= 0:
            return
        stop_distance = atr_val * 1.5
        position_size = int(round(risk_amount / stop_distance))
        if position_size < 1:
            position_size = 1

        # 🌙 Long vol (underpriced): spread strongly negative → long the fast repriced side
        if spread_z <= -self.spread_z_threshold:
            print(f"🌙🚀 LONG VOL ENTRY | spread_z={spread_z:.2f} | ΔRV={delta_rv:.2f} | size={position_size} | price={price:.2f}")
            self.buy(size=position_size)
            self.entry_bar = len(self.data)
            self.entry_spread = spread_z

        # 🌙 Short vol (overpriced): spread strongly positive → short the lagging side
        elif spread_z >= self.spread_z_threshold:
            print(f"🌙🔻 SHORT VOL ENTRY | spread_z={spread_z:.2f} | ΔRV={delta_rv:.2f} | size={position_size} | price={price:.2f}")
            self.sell(size=position_size)
            self.entry_bar = len(self.data)
            self.entry_spread = spread_z


# 🌙 Run backtest
bt = Backtest(
    data,
    VolatilityCliffHarvester,
    cash=1_000_000,
    commission=0.0002,
    exclusive_orders=True
)

stats = bt.run()
print(stats)
print(stats._strategy)