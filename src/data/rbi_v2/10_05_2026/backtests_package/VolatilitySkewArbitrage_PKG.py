import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Moon Dev's VolatilitySkewArbitrage Backtest 🌙✨🚀

def load_data(path):
    print("🌙 Loading Moon Dev data from the cosmic archives...")
    data = pd.read_csv(path)
    data.columns = data.columns.str.strip().str.lower()
    data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
    # Map to proper case
    data = data.rename(columns={
        'datetime': 'Date',
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    })
    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'])
        data = data.set_index('Date')
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    print(f"✨ Loaded {len(data)} bars of lunar price action!")
    return data


class VolatilitySkewArbitrage(Strategy):
    # Strategy parameters
    vix_high = 50
    vix_low = 20
    vix_high_exit = 40
    vix_low_exit = 30
    vix_high_confirm = 2
    vix_low_confirm = 3
    vix_proxy_window = 20  # rolling window for realized vol proxy
    iv_dn_window = 252
    risk_pct = 0.02
    atr_period = 14
    atr_mult = 2.0

    def init(self):
        print("🌙 Initializing VolatilitySkewArbitrage strategy...")
        close = self.data.Close
        high = self.data.High
        low = self.data.Low

        # Realized volatility proxy for VIX (annualized std of returns)
        def realized_vol(close, window):
            ret = np.log(close / np.roll(close, 1))
            ret[0] = 0.0
            return pd.Series(ret).rolling(window).std().values * np.sqrt(252 * 96)

        self.vix = self.I(realized_vol, close, self.vix_proxy_window, name="VIX_Proxy")

        # Delta-neutral IV proxy: use realized vol / |z-score of close| as proxy
        # Since we don't have options chain, we approximate IV_dn with realized vol normalized by delta proxy
        def iv_dn_proxy(close, window):
            ret = np.log(close / np.roll(close, 1))
            ret[0] = 0.0
            rv = pd.Series(ret).rolling(window).std().values * np.sqrt(252 * 96)
            # delta proxy: distance from rolling mean normalized
            ma = pd.Series(close).rolling(window).mean().values
            sd = pd.Series(close).rolling(window).std().values
            delta_proxy = np.abs((close - ma) / np.where(sd == 0, 1, sd))
            delta_proxy = np.where(delta_proxy < 0.01, 0.01, delta_proxy)
            return rv / delta_proxy

        self.iv_dn = self.I(iv_dn_proxy, close, 20, name="IV_DN_Proxy")

        # Rolling percentile of IV_dn
        def pct_rank(arr, window):
            s = pd.Series(arr)
            return s.rolling(window).apply(lambda x: (x.iloc[-1] > x.iloc[:-1]).mean() if len(x) > 1 else 0.5, raw=False).values

        self.iv_dn_pct = self.I(pct_rank, self.iv_dn, self.iv_dn_window, name="IV_DN_Pct")

        # ATR for risk sizing
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period, name="ATR")

        # Prior day high/low
        def prior_high(high):
            return pd.Series(high).shift(1).values
        def prior_low(low):
            return pd.Series(low).shift(1).values

        self.prior_high = self.I(prior_high, high, name="PriorHigh")
        self.prior_low = self.I(prior_low, low, name="PriorLow")

        # Track state
        self.entry_strike = None  # For trailing stop at original strike
        self.entry_price = None
        self.trade_type = None  # 'call' or 'put'
        self.vix_high_count = 0
        self.vix_low_count = 0

    def next(self):
        price = self.data.Close[-1]
        vix = self.vix[-1]
        iv_dn_pct = self.iv_dn_pct[-1]
        atr = self.atr[-1]

        if np.isnan(vix) or np.isnan(iv_dn_pct) or np.isnan(atr) or atr <= 0:
            return

        # Track consecutive VIX regime closes
        if vix > self.vix_high:
            self.vix_high_count += 1
        else:
            self.vix_high_count = 0

        if vix < self.vix_low:
            self.vix_low_count += 1
        else:
            self.vix_low_count = 0

        # ============ EXIT LOGIC ============
        if self.position:
            # Trailing stop at original strike
            if self.trade_type == 'call':
                # Stop at original strike (entry price proxy since no options)
                stop_price = max(self.entry_strike, self.entry_strike)
                # Trail upward but never below original strike
                if price < stop_price:
                    print(f"🌙 Exit CALL — price {price:.2f} breached original strike {self.entry_strike:.2f} 🛑")
                    self.position.close()
                    self._reset_state()
                    return
                # Regime exit: VIX crosses below 40
                if vix < self.vix_high_exit:
                    print(f"🌙 Regime exit CALL — VIX {vix:.2f} < {self.vix_high_exit} 🌊")
                    self.position.close()
                    self._reset_state()
                    return
                # Vol collapse exit: IV_dn reverts to median
                if iv_dn_pct < 0.5:
                    print(f"🌙 Vol collapse exit CALL — IV_dn pct {iv_dn_pct:.2f} 📉")
                    self.position.close()
                    self._reset_state()
                    return

            elif self.trade_type == 'put':
                stop_price = self.entry_strike
                if price > stop_price:
                    print(f"🌙 Exit PUT — price {price:.2f} breached original strike {self.entry_strike:.2f} 🛑")
                    self.position.close()
                    self._reset_state()
                    return
                if vix > self.vix_low_exit:
                    print(f"🌙 Regime exit PUT — VIX {vix:.2f} > {self.vix_low_exit} 🌊")
                    self.position.close()
                    self._reset_state()
                    return
                if iv_dn_pct > 0.5:
                    print(f"🌙 Vol collapse exit PUT — IV_dn pct {iv_dn_pct:.2f} 📉")
                    self.position.close()
                    self._reset_state()
                    return

        # ============ ENTRY LOGIC ============
        if not self.position:
            # Long Call Entry: VIX > 50 for 2 consecutive sessions, pullback day
            if (self.vix_high_count >= self.vix_high_confirm and
                    iv_dn_pct >= 0.90 and
                    price < self.prior_low[-1]):
                # Risk sizing: 2% of equity
                risk_amount = self.equity * self.risk_pct
                stop_distance = atr * self.atr_mult
                if stop_distance > 0:
                    size = int(round(risk_amount / stop_distance))
                    size = max(1, min(size, int(self.equity / price)))
                    if size > 0:
                        print(f"🚀🌙 MOON DEV LONG CALL SIGNAL! VIX={vix:.2f}, IV_dn_pct={iv_dn_pct:.2f}, Pullback confirmed. Size={size}")
                        self.buy(size=size)
                        self.entry_strike = price
                        self.entry_price = price
                        self.trade_type = 'call'

            # Short Put Entry: VIX < 20 for 3 consecutive sessions, rally day
            elif (self.vix_low_count >= self.vix_low_confirm and
                  iv_dn_pct <= 0.10 and
                  price > self.prior_high[-1]):
                risk_amount = self.equity * self.risk_pct
                stop_distance = atr * self.atr_mult
                if stop_distance > 0:
                    size = int(round(risk_amount / stop_distance))
                    size = max(1, min(size, int(self.equity / price)))
                    if size > 0:
                        print(f"🚀🌙 MOON DEV SHORT PUT SIGNAL! VIX={vix:.2f}, IV_dn_pct={iv_dn_pct:.2f}, Rally confirmed. Size={size}")
                        self.sell(size=size)
                        self.entry_strike = price
                        self.entry_price = price
                        self.trade_type = 'put'

    def _reset_state(self):
        self.entry_strike = None
        self.entry_price = None
        self.trade_type = None


if __name__ == "__main__":
    print("🌙✨🚀 Moon Dev's VolatilitySkewArbitrage Backtest Starting...")
    data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
    data = load_data(data_path)

    bt = Backtest(
        data,
        VolatilitySkewArbitrage,
        cash=1_000_000,
        commission=0.002,
        exclusive=False
    )

    print("🌙 Running backtest with 1,000,000 lunar credits...")
    stats = bt.run()
    print(stats)
    print(stats._strategy)
    print("✨🌙 Backtest complete! May the volatility premium be with you! 🚀")