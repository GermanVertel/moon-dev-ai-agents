import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# ============================================================
# 🌙 Moon Dev's TwilightReversion Backtest 🌙
# Mean-reversion on late-day Bollinger Band breaches
# ============================================================

DATA_PATH = "/Users/germandavidvertelnarvaez/Developer/MoonDev/moon-dev-ai-agents/src/data/rbi/BTC-USD-15m.csv"


def load_and_clean_data(path):
    print("🌙✨ Loading cosmic data from:", path)
    data = pd.read_csv(path)
    # Clean column names
    data.columns = data.columns.str.strip().str.lower()
    # Drop unnamed columns
    data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
    # Map to backtesting.py required columns
    data = data.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume',
    })
    # Parse datetime
    if 'datetime' in data.columns:
        data['datetime'] = pd.to_datetime(data['datetime'])
        data = data.set_index('datetime')
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    data = data.dropna()
    print("🌙✨ Data loaded with", len(data), "rows 🚀")
    return data


class TwilightReversion(Strategy):
    # Strategy parameters
    bb_period = 20
    bb_std = 2.0
    rsi_period = 14
    rsi_threshold = 70
    # Late-day window (UTC hours — tune to your market)
    session_start_hour = 20   # 20:00 UTC ~ 2PM ET (adjust for your data)
    session_end_hour = 23     # 23:00 UTC ~ 5PM ET
    risk_pct = 0.01           # 1% risk per trade
    atr_period = 14
    atr_stop_mult = 1.5

    def init(self):
        print("🌙 TwilightReversion initializing indicators... ✨")
        close = self.data.Close
        high = self.data.High
        low = self.data.Low

        # Bollinger Bands via TA-Lib
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS, close, timeperiod=self.bb_period,
            nbdevup=self.bb_std, nbdevdn=self.bb_std, matype=0,
            name="BBANDS"
        )

        # RSI for overbought confirmation
        self.rsi = self.I(talib.RSI, close, timeperiod=self.rsi_period, name="RSI")

        # ATR for risk sizing / stops
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=self.atr_period, name="ATR")

        # Track session signal limit
        self.last_signal_date = None
        print("🌙 Indicators ready — Twilight is falling 🌆")

    def _in_session(self, ts):
        return self.session_start_hour <= ts.hour < self.session_end_hour

    def next(self):
        # Need enough bars
        if len(self.data) < self.bb_period + 2:
            return

        price = self.data.Close[-1]
        upper = self.bb_upper[-1]
        lower = self.bb_lower[-1]
        rsi = self.rsi[-1]
        atr = self.atr[-1]
        ts = self.data.index[-1]

        if np.isnan(upper) or np.isnan(lower) or np.isnan(rsi) or np.isnan(atr):
            return

        in_session = self._in_session(ts)
        current_date = ts.date()

        # ============================================================
        # EXIT LOGIC
        # ============================================================
        if self.position:
            # Primary: close below upper band → reversion complete
            if price < upper:
                print(f"🌙✨ [EXIT] Reversion to band at {ts} | Price {price:.2f} < Upper {upper:.2f} 🚀")
                self.position.close()
                return

            # Hard stop: close below lower band → trend continuation
            if price < lower:
                print(f"🌙⚠️ [STOP] Close below lower band at {ts} | Price {price:.2f} < Lower {lower:.2f} 💥")
                self.position.close()
                return

            # EOD flatten: outside session → close
            if not in_session:
                print(f"🌙🌆 [EOD FLATTEN] Session end at {ts} | Closing short 💤")
                self.position.close()
                return

        # ============================================================
        # ENTRY LOGIC (short only)
        # ============================================================
        if not self.position and in_session:
            # One signal per session
            if self.last_signal_date == current_date:
                return

            # Close above upper band → overextension
            if price > upper:
                # Optional RSI confirmation
                if rsi > self.rsi_threshold:
                    # Risk-based sizing
                    risk_amount = self.equity * self.risk_pct
                    stop_distance = atr * self.atr_stop_mult
                    if stop_distance <= 0:
                        return
                    # Short size — cap to equity-based sizing
                    position_size = int(round(risk_amount / stop_distance))
                    # Cap to something reasonable (equity / price)
                    max_size = int(self.equity // price) if price > 0 else 0
                    position_size = min(position_size, max_size)
                    if position_size <= 0:
                        position_size = 1

                    print(f"🌙🚀 [SHORT ENTRY] {ts} | Price {price:.2f} > Upper {upper:.2f} | RSI {rsi:.1f} | Size {position_size} 🌆")
                    self.sell(size=position_size)
                    self.last_signal_date = current_date


# ============================================================
# 🌙 RUN THE BACKTEST 🌙
# ============================================================

if __name__ == "__main__":
    data = load_and_clean_data(DATA_PATH)
    bt = Backtest(
        data,
        TwilightReversion,
        cash=1_000_000,
        commission=0.0002,
        exclusive_orders=True,
    )
    stats = bt.run()
    print(stats)
    print(stats._strategy)