import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

print("🌙 Moon Dev's VolatilityContrarian Backtest Initializing... ✨")

data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"

data = pd.read_csv(data_path)
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

if 'datetime' in data.columns:
    data['datetime'] = pd.to_datetime(data['datetime'])
    data = data.set_index('datetime')

data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
print(f"🚀 Data loaded: {len(data)} rows from {data.index[0]} to {data.index[-1]}")


class VolatilityContrarian(Strategy):
    """
    🌙 VolatilityContrarian Strategy
    """
    
    # Parameters
    oiv_lookback = 252
    entry_z_high = 1.5
    entry_z_low = -1.5
    exit_z_threshold = 0.5
    atr_period = 14
    atr_stop_mult = 2.0
    risk_pct = 0.02
    vol_window = 30
    
    def init(self):
        print("✨ Initializing VolatilityContrarian indicators... 🌙")
        
        close = pd.Series(self.data.Close)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)
        
        # Proxy OIV: 30-day realized volatility (annualized)
        log_ret = np.log(close / close.shift(1))
        realized_vol = log_ret.rolling(self.vol_window).std() * np.sqrt(252) * 100
        
        # OIV z-score (252-day rolling)
        oiv_mean = realized_vol.rolling(self.oiv_lookback).mean()
        oiv_std = realized_vol.rolling(self.oiv_lookback).std()
        oiv_z = (realized_vol - oiv_mean) / oiv_std
        
        # ATR for stops
        atr = talib.ATR(high.values, low.values, close.values, timeperiod=self.atr_period)
        
        # Momentum (for deceleration confirmation)
        mom = close.pct_change(20)
        mom_prev = mom.shift(5)
        
        self.oiv_z = self.I(lambda: oiv_z.values, name='OIV_z')
        self.realized_vol = self.I(lambda: realized_vol.values, name='RealizedVol')
        self.atr = self.I(lambda: atr, name='ATR')
        self.mom = self.I(lambda: mom.values, name='Momentum')
        self.mom_prev = self.I(lambda: mom_prev.values, name='Momentum_prev')
        
        print("🌙 Indicators ready! OIV_z, RealizedVol, ATR, Momentum loaded ✨")
    
    def next(self):
        if len(self.data) < self.oiv_lookback + self.vol_window + 10:
            return
        
        price = self.data.Close[-1]
        z = self.oiv_z[-1]
        atr = self.atr[-1]
        mom = self.mom[-1]
        mom_prev = self.mom_prev[-1]
        
        if np.isnan(z) or np.isnan(atr) or atr <= 0:
            return
        
        # --- Exit logic ---
        if self.position:
            if self.position.is_long and abs(z) < self.exit_z_threshold:
                print(f"🌙 EXIT LONG | OIV_z reverted to {z:.2f} | Price: {price:.2f} ✨")
                self.position.close()
                return
            if self.position.is_short and abs(z) < self.exit_z_threshold:
                print(f"🌙 EXIT SHORT | OIV_z reverted to {z:.2f} | Price: {price:.2f} ✨")
                self.position.close()
                return
        
        # --- Entry logic ---
        if not self.position:
            # Contrarian Long: extreme high OIV_z (fear) + momentum decelerating
            if z > self.entry_z_high:
                decelerating = (not np.isnan(mom_prev)) and (mom < mom_prev)
                if decelerating:
                    stop_price = price - self.atr_stop_mult * atr
                    risk_per_unit = price - stop_price
                    if risk_per_unit > 0:
                        equity = self.equity
                        risk_amount = equity * self.risk_pct
                        size = int(round(risk_amount / risk_per_unit))
                        if size > 0:
                            print(f"🚀 LONG ENTRY | OIV_z={z:.2f} (fear) | Price={price:.2f} | Size={size} | Stop={stop_price:.2f} 🌙")
                            self.buy(size=size, sl=stop_price)
                            return
            
            # Contrarian Short: extreme low OIV_z (complacency) + momentum decelerating
            if z < self.entry_z_low:
                decelerating = (not np.isnan(mom_prev)) and (mom > mom_prev)
                if decelerating:
                    stop_price = price + self.atr_stop_mult * atr
                    risk_per_unit = stop_price - price
                    if risk_per_unit > 0:
                        equity = self.equity
                        risk_amount = equity * self.risk_pct
                        size = int(round(risk_amount / risk_per_unit))
                        if size > 0:
                            print(f"🔻 SHORT ENTRY | OIV_z={z:.2f} (complacency) | Price={price:.2f} | Size={size} | Stop={stop_price:.2f} 🌙")
                            self.sell(size=size, sl=stop_price)
                            return


print("🌙 Setting up backtest engine... 🚀")
bt = Backtest(
    data,
    VolatilityContrarian,
    cash=1_000_000,
    commission=0.002
)

print("✨ Running backtest... 🌙")
stats = bt.run()
print(stats)
print(stats._strategy)
print("🌙 Moon Dev's VolatilityContrarian backtest complete! 🚀✨")