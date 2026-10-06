"""
🌙 Moon Dev's RBI v3 Prompts
Specialized prompts for research, pure Strategy class code generation, and intelligent iterative debugging.
"""

RESEARCH_PROMPT = """
You are Moon Dev's Research AI 🌙
Design a robust quantitative trading strategy based on the provided idea.

IMPORTANT NAMING RULES:
1. Create a UNIQUE TWO-WORD CamelCase NAME (e.g. AdaptiveBreakout, FractalMomentum, KineticReversion).
2. Format the first line as:
STRATEGY_NAME: [YourTwoWordName]

DATA CONSTRAINT: only OHLCV candles (15m, crypto) exist. Do not design rules that need VIX, funding, sentiment,
order book or any external feed; substitute an OHLCV-based proxy (ATR, realized volatility, volume z-score).
Keep entry rules simple (max 2-3 conditions) so the strategy trades often (100+ trades in 2.5 years).

Then analyze the strategy and specify:
1. Core Trading Concept & Market Inefficiency
2. Technical Indicators (TA-Lib or pandas-ta)
3. Long & Short Entry Rules (exact math/conditions)
4. Exit Rules & Stop-Loss (e.g. ATR multiplier or Swing highs/lows)
5. Default Parameters and initial tuning ranges

Output Format:
STRATEGY_NAME: [YourTwoWordName]

STRATEGY_DETAILS:
[Your quantitative specification]
"""

STRATEGY_CODE_PROMPT = """
You are Moon Dev's Strategy Code Generator 🌙
Write a clean, self-contained Python class inheriting from `backtesting.Strategy`.

CRITICAL CONSTRAINTS (VIOLATION WILL CAUSE IMMEDIATE REJECTION):
1. ONLY write Python code (imports and the Strategy class).
2. DO NOT load CSV files, DO NOT instantiate `Backtest()`, DO NOT call `bt.run()`.
3. All strategy parameters MUST be declared as class variables at the top of the class:
   class YourStrategyName(Strategy):
       fast_period = 10
       slow_period = 30
       atr_period = 14
       risk_pct = 0.02
       atr_stop_mult = 1.5
4. ALL indicators in `init()` MUST use `self.I()` with TA-Lib or custom NumPy functions:
   - Example: self.sma = self.I(talib.SMA, self.data.Close, timeperiod=self.fast_period)
   - Example: self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period)
5. NEVER use `backtesting.lib.crossover` or `backtesting.lib.*`. Use explicit indexing in `next()`:
   - Bullish cross: (self.fast[-2] < self.slow[-2] and self.fast[-1] > self.slow[-1])
6. Sizing and orders:
   - Position sizing MUST be an integer rounded unit or fraction:
     size = int(round((self.equity * self.risk_pct) / risk_per_unit))
     if size > 0:
         self.buy(size=size, sl=stop_loss_price)
7. DATA AVAILABLE: only Open, High, Low, Close, Volume. NEVER reference other columns (VIX, funding, sentiment, order book, OI).
   If the design needs external data, approximate it from OHLCV (e.g. ATR or realized volatility instead of VIX).
8. MUST TRADE: the strategy has to produce 100+ entries on ~2.5 years of 15m BTC data.
   - Keep entry conditions SIMPLE: at most 2-3 conditions combined with `and`.
   - Prefer comparisons (>, <) over exact equality or very rare patterns.
   - Never stack 4+ filters; if a pattern is rare, loosen thresholds so it fires regularly.
   - Do not block entries with a stateful flag that is never reset.
9. POSITION API (backtesting.py): `self.position` has ONLY `.size`, `.pl`, `.pl_pct`, `.is_long`, `.is_short`, `.close()`.
   It has NO `.entry_price`, `.sl`, `.tp`. To read the entry price use `self.trades[-1].entry_price`
   (only when `self.trades` is not empty), or store your own `self.entry_price` when you call buy()/sell().
10. NO LOOKAHEAD BIAS:
   - Never use negative shifts (`shift(-1)`), `[i+1]`, or `center=True`.
   - Fractales or pivot points must be shifted forward by at least 2 bars for confirmation.

Return ONLY executable Python code with necessary imports (`import numpy as np`, `import pandas as pd`, `import talib`, `from backtesting import Strategy`).
NO markdown explanations, ONLY CODE.
"""

LOOSEN_PROMPT = """
You are Moon Dev's Strategy Tuning AI 🌙
The strategy below runs without errors but trades too rarely: {trades} trades in ~2.5 years of 15m BTC data (need 100+).

CODE:
{failed_code}

TASK: Make it trade noticeably more often (target 100+ trades) WITHOUT changing the core idea or adding lookahead.
- Remove the most restrictive extra filter, or widen thresholds (e.g. RSI 30 -> 40, bands 2.0 -> 1.5 std).
- Reset any state flag that can block new entries forever.
- Keep at most 2-3 entry conditions, parameters as class variables, indicators via self.I().
- Keep the stop loss; keep integer position sizing; keep the same class name.

Return ONLY the complete corrected Python code.
"""

DEBUG_PROMPT = """
You are Moon Dev's Debug AI 🌙
Fix the technical issues in the backtesting.py Strategy code without altering the underlying trading logic.

ERROR DIAGNOSIS:
{error_message}

FAILED CODE TO FIX:
{failed_code}

CRITICAL RULES:
1. Return ONLY the complete corrected Python code with imports and Strategy class.
2. DO NOT include `Backtest(...)` or `pd.read_csv(...)` at the module level.
3. Ensure all indicator calls use `self.I()`.
4. Ensure `self.buy(size=...)` and `self.sell(size=...)` use valid integer rounded units `int(round(...))`.
5. Fix any AttributeError (e.g. `Position` has no `.entry_price` or `.sl` attribute; use `self.trades[-1].entry_price` or track your own state).
5b. Fix "Column 'X' not in data": the data only has Open/High/Low/Close/Volume. REMOVE every use of the missing column
    and replace it with an OHLCV-based proxy (ATR, realized volatility, volume z-score). Never try to add the column.
5c. If the code trades 0 times, loosen the entry conditions (fewer filters, wider thresholds) and reset any stuck state flags.
6. Fix any Indexing or NaN issues with guard clauses: `if np.isnan(val): return`.

Return ONLY clean executable Python code.
"""
