import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# 🌙 Moon Dev's KineticSqueeze Strategy 🌙
print("🚀 Initializing Moon Dev's KineticSqueeze Backtest...")
print("✨ Loading cosmic data from the moon...")

# Load data
data_path = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# 🌙 Clean the data columns
print("🧹 Cleaning data columns...")
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

print(f"🌙 Data loaded: {len(data)} rows of cosmic price action ✨")
print(f"🚀 Date range: {data.index[0]} to {data.index[-1]}")


class KineticSqueeze(Strategy):
    """
    🌙 KineticSqueeze Strategy 🌙
    Bollinger Bands + Stochastic Oscillator for momentum shifts after low volatility
    """
    
    # Strategy parameters
    bb_period = 20
    bb_std = 2.0
    stoch_period = 14
    stoch_k_smooth = 3
    stoch_d_smooth = 3
    atr_period = 14
    squeeze_lookback = 20  # 20-period average of BB width
    time_exit_bars = 10
    risk_pct = 0.01
    reward_risk = 2.0
    
    def init(self):
        print("🌙 Initializing KineticSqueeze indicators...")
        
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        
        # 🌙 Bollinger Bands
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS, close, timeperiod=self.bb_period,
            nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0,
            name=['BB_Upper', 'BB_Middle', 'BB_Lower']
        )
        
        # 🌙 Bollinger Band Width
        self.bb_width = self.I(
            lambda u, l: u - l, self.bb_upper, self.bb_lower,
            name='BB_Width'
        )
        
        # 🌙 BB Width average for squeeze detection
        self.bb_width_avg = self.I(
            talib.SMA, self.bb_width, timeperiod=self.squeeze_lookback,
            name='BB_Width_Avg'
        )
        
        # 🌙 Stochastic Oscillator (14,3,3)
        self.stoch_k, self.stoch_d = self.I(
            talib.STOCH, high, low, close,
            fastk_period=self.stoch_period,
            slowk_period=self.stoch_k_smooth,
            slowk_matype=0,
            slowd_period=self.stoch_d_smooth,
            slowd_matype=0,
            name=['Stoch_K', 'Stoch_D']
        )
        
        # 🌙 ATR for stop loss
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period, name='ATR')
        
        # Track entry bar and stop levels
        self.entry_bar = None
        self.stop_price = None
        self.take_profit_price = None
        self.trade_direction = None
        
        # Consecutive losses tracker
        self.consecutive_losses = 0
        self.last_trade_day = None
        self.trading_halted_today = False
        
        print("✨ All indicators initialized! Ready to squeeze the moon 🌙")
    
    def is_squeeze(self):
        """Check if Bollinger Bands are in a squeeze"""
        if np.isnan(self.bb_width[-1]) or np.isnan(self.bb_width_avg[-1]):
            return False
        return self.bb_width[-1] < self.bb_width_avg[-1]
    
    def next(self):
        # 🌙 Check for day change to reset daily halt
        current_day = self.data.index[-1].date()
        if self.last_trade_day != current_day:
            self.last_trade_day = current_day
            self.trading_halted_today = False
        
        # 🌙 Skip if trading halted for the day
        if self.trading_halted_today:
            return
        
        # 🌙 Time-based exit
        if self.position and self.entry_bar is not None:
            bars_in_trade = len(self.data) - self.entry_bar
            if bars_in_trade >= self.time_exit_bars:
                print(f"⏰ Moon Dev Time Exit: Closing position after {bars_in_trade} bars 🌙")
                self.position.close()
                self._reset_trade_state()
                return
        
        # 🌙 Manage existing position exits
        if self.position:
            self._manage_exit()
            return
        
        # 🌙 Check for entry signals
        if len(self.data) < 50:
            return
        
        # Check squeeze condition
        if not self.is_squeeze():
            return
        
        # Get current values
        price = self.data.Close[-1]
        bb_upper = self.bb_upper[-1]
        bb_lower = self.bb_lower[-1]
        stoch_k = self.stoch_k[-1]
        stoch_d = self.stoch_d[-1]
        stoch_k_prev = self.stoch_k[-2]
        stoch_d_prev = self.stoch_d[-2]
        atr = self.atr[-1]
        
        if np.isnan(bb_upper) or np.isnan(stoch_k) or np.isnan(atr):
            return
        
        # 🌙 LONG ENTRY: Price touches lower BB + Stoch K crosses above D from below 20
        long_signal = (
            price <= bb_lower * 1.001 and  # Touch/pierce lower BB
            stoch_k_prev <= stoch_d_prev and  # Was below
            stoch_k > stoch_d and  # Crossed above
            stoch_k < 20 and stoch_d < 20  # Both below 20
        )
        
        # 🌙 SHORT ENTRY: Price touches upper BB + Stoch K crosses below D from above 80
        short_signal = (
            price >= bb_upper * 0.999 and  # Touch/pierce upper BB
            stoch_k_prev >= stoch_d_prev and  # Was above
            stoch_k < stoch_d and  # Crossed below
            stoch_k > 80 and stoch_d > 80  # Both above 80
        )
        
        if long_signal:
            self._enter_long(price, bb_lower, atr)
        elif short_signal:
            self._enter_short(price, bb_upper, atr)
    
    def _enter_long(self, price, bb_lower, atr):
        """Enter a long position"""
        stop_price = bb_lower - atr
        risk_per_unit = price - stop_price
        
        if risk_per_unit <= 0:
            return
        
        # Position sizing: risk 1% of account
        risk_amount = self.equity * self.risk_pct
        position_size = risk_amount / risk_per_unit
        position_size = int(round(position_size))
        
        if position_size <= 0:
            return
        
        take_profit = price + (risk_per_unit * self.reward_risk)
        
        print(f"🚀🌙 MOON DEV LONG SIGNAL! 🚀🌙")
        print(f"   Entry: ${price:.2f} | Stop: ${stop_price:.2f} | TP: ${take_profit:.2f}")
        print(f"   Size: {position_size} | Risk: ${risk_amount:.2f}")
        
        self.buy(size=position_size)
        self.entry_bar = len(self.data)
        self.stop_price = stop_price
        self.take_profit_price = take_profit
        self.trade_direction = 'long'
    
    def _enter_short(self, price, bb_upper, atr):
        """Enter a short position"""
        stop_price = bb_upper + atr
        risk_per_unit = stop_price - price
        
        if risk_per_unit <= 0:
            return
        
        risk_amount = self.equity * self.risk_pct
        position_size = risk_amount / risk_per_unit
        position_size = int(round(position_size))
        
        if position_size <= 0:
            return
        
        take_profit = price - (risk_per_unit * self.reward_risk)
        
        print(f"🔻🌙 MOON DEV SHORT SIGNAL! 🔻🌙")
        print(f"   Entry: ${price:.2f} | Stop: ${stop_price:.2f} | TP: ${take_profit:.2f}")
        print(f"   Size: {position_size} | Risk: ${risk_amount:.2f}")
        
        self.sell(size=position_size)
        self.entry_bar = len(self.data)
        self.stop_price = stop_price
        self.take_profit_price = take_profit
        self.trade_direction = 'short'
    
    def _manage_exit(self):
        """Manage exit conditions for existing position"""
        price = self.data.Close[-1]
        stoch_k = self.stoch_k[-1]
        stoch_d = self.stoch_d[-1]
        stoch_k_prev = self.stoch_k[-2]
        stoch_d_prev = self.stoch_d[-2]
        bb_upper = self.bb_upper[-1]
        bb_lower = self.bb_lower[-1]
        
        if np.isnan(stoch_k) or np.isnan(bb_upper):
            return
        
        if self.trade_direction == 'long':
            # Stop loss hit
            if price <= self.stop_price:
                print(f"🛑 Moon Dev STOP LOSS hit on LONG at ${price:.2f} 🌙")
                self.position.close()
                self._reset_trade_state()
                return
            # Take profit hit
            if price >= self.take_profit_price:
                print(f"🎯 Moon Dev TAKE PROFIT hit on LONG at ${price:.2f} 🌙")
                self.position.close()
                self._reset_trade_state()
                return
            # Exit at upper BB
            if price >= bb_upper:
                print(f"🌙 Moon Dev Exit: Price closed above upper BB at ${price:.2f} ✨")
                self.position.close()
                self._reset_trade_state()
                return
            # Stoch K crosses below D above 80
            if (stoch_k_prev >= stoch_d_prev and stoch_k < stoch_d and 
                stoch_k > 80 and stoch_d > 80):
                print(f"🌙 Moon Dev Exit: Stoch bearish cross above 80 ✨")
                self.position.close()
                self._reset_trade_state()
                return
        
        elif self.trade_direction == 'short':
            # Stop loss hit
            if price >= self.stop_price:
                print(f"🛑 Moon Dev STOP LOSS hit on SHORT at ${price:.2f} 🌙")
                self.position.close()
                self._reset_trade_state()
                return
            # Take profit hit
            if price <= self.take_profit_price:
                print(f"🎯 Moon Dev TAKE PROFIT hit on SHORT at ${price:.2f} 🌙")
                self.position.close()
                self._reset_trade_state()
                return
            # Exit at lower BB
            if price <= bb_lower:
                print(f"🌙 Moon Dev Exit: Price closed below lower BB at ${price:.2f} ✨")
                self.position.close()
                self._reset_trade_state()
                return
            # Stoch K crosses above D below 20
            if (stoch_k_prev <= stoch_d_prev and stoch_k > stoch_d and 
                stoch_k < 20 and stoch_d < 20):
                print(f"🌙 Moon Dev Exit: Stoch bullish cross below 20 ✨")
                self.position.close()
                self._reset_trade_state()
                return
    
    def _reset_trade_state(self):
        """Reset trade tracking variables"""
        self.entry_bar = None
        self.stop_price = None
        self.take_profit_price = None
        self.trade_direction = None


print("🌙✨🚀 Launching Moon Dev's KineticSqueeze Backtest! 🚀✨🌙")

# Run backtest
bt = Backtest(
    data,
    KineticSqueeze,
    cash=1_000_000,
    commission=0.001,
    trade_on_close=False
)

stats = bt.run()
print("\n" + "="*80)
print("🌙✨ MOON DEV KINETICSQUEEZE BACKTEST RESULTS ✨🌙")
print("="*80)
print(stats)
print("\n" + "="*80)
print("🌙✨ STRATEGY DETAILS ✨🌙")
print("="*80)
print(stats._strategy)
print("\n🚀 Moon Dev out! To the moon! 🌙✨")