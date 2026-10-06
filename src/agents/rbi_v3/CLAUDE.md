# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## RBI v3: Quantitative Trading Strategy Validation Engine

**RBI v3** (Research → Backtest → Validate → MultiAsset → OutOfSample) is a 9-phase LLM-driven strategy generation and validation pipeline. It generates trading strategies from ideas, backtests them with rigorous statistical gates, and archives certified winners.

## High-Level Architecture

RBI v3 executes a standardized validation pipeline:

```
Ideas → Phase 0: Jev Triage → Phase 1: Research AI (DeepSeek)
      → Phase 2: Code Generation → Phase 3-5: Static Lint & Debug Loop
      → Phase 6: In-Sample Gate (BTC) → Phase 7: Multi-Asset Test (ETH, SOL)
      → Phase 8: Out-of-Sample Blind Test → Phase 9: Winner Archiving & Leaderboard
```

**Core Module Responsibilities:**
- `harness.py`: Sandboxed subprocess execution of strategies using `backtesting.py` with standardized parameters ($100k cash, 0.045% commission)
- `data_manager.py`: Downloads continuous market data from Binance, partitions into in-sample (pre-2024-07-01) and out-of-sample (post-2024-07-01) splits
- `evaluator.py`: Implements quantitative gates and composite scoring logic
- `jev_gate.py`: Ultra-fast idea classification using TypeSafe Jev (via OpenRouter); provides local heuristic fallback
- `code_checks.py`: AST-based static validation, forbidden import detection, lookahead-bias regex checks
- `leaderboard.py`: Tracks all evaluated strategies in CSV/Markdown, archives certified winners with metadata
- `prompts.py`: Prompt templates for research, code generation, and debugging phases

**Orchestrator:**
- `rbi_agent_v3.py`: Main entry point; processes ideas from `src/data/rbi_v3/ideas.txt` or `src/data/rbi_v2/ideas.txt` through all 9 phases

## Running RBI v3

### Full Pipeline (processes all ideas in ideas.txt)
```bash
# From repo root (moon-dev-ai-agents/)
python -m src.agents.rbi_agent_v3
```

### With Environment Controls
```bash
# Limit to first N ideas
RBI_MAX_IDEAS=5 python -m src.agents.rbi_agent_v3

# Use verbose output (none currently; relies on termcolor cprint)
python -m src.agents.rbi_agent_v3
```

### Idea Generation (RBI v3 Dedicated)
```bash
# Generate batch of 5 new actionable OHLCV ideas
python -m src.agents.rbi_v3.idea_generator --count 5

# Continuous loop generating ideas periodically
python -m src.agents.rbi_v3.idea_generator --continuous --interval 20
```

### Data Management Only
```bash
# Download and partition market data (BTC/ETH/SOL, 15m/1h)
python src/agents/rbi_v3/data_manager.py
```

### Individual Module Testing
```bash
# Validate a strategy code string (returns ValidationResult namedtuple)
from src.agents.rbi_v3 import validate_strategy_code
result = validate_strategy_code(code_str)
print(result.is_valid, result.error_message, result.warnings, result.class_name)

# Load and run backtest on a dataset
from src.agents.rbi_v3 import run_backtest_harness
result = run_backtest_harness(strategy_code, class_name, data_path, cash=100_000, commission=0.00045, timeout_sec=180)

# Evaluate results against gates
from src.agents.rbi_v3 import StrategyEvaluator
evaluator = StrategyEvaluator()
is_passed, reason, score = evaluator.evaluate_in_sample(backtest_stats_dict)
```

## Data Directory Structure

```
src/data/rbi_v3/
├── market_data/              # Binance-downloaded OHLCV CSVs
│   ├── BTC-USD-15m_IS.csv   # In-sample (2022-01-01 to 2024-07-01)
│   ├── BTC-USD-15m_OOS.csv  # Out-of-sample (2024-07-01 to present)
│   ├── BTC-USD-15m_FULL.csv # Complete history
│   └── [ETH, SOL variants with same pattern]
├── strategies/               # Working strategy code (not winners)
│   └── StrategyName_<hash>.py
├── winners/                  # Certified winner archives
│   ├── StrategyName_<hash>_WINNER.py
│   └── StrategyName_<hash>_metadata.json
├── leaderboard.csv          # All evaluated strategies (sorted by score)
├── leaderboard.md           # GitHub-flavored markdown leaderboard
├── state.jsonl              # State log; one JSON record per idea processed
└── ideas.txt                # Input ideas (one per line; comments start with #)
```

## Key Configuration & Thresholds

### Gating Thresholds (src/agents/rbi_v3/evaluator.py)
```python
DEFAULT_GATES = {
    "min_trades": 100,              # In-sample (statistical significance)
    "max_trades": 800,              # Prevent overtrading/fee bleed
    "min_sharpe": 1.0,              # In-sample risk-adjusted return
    "min_profit_factor": 1.3,       # (gross_profit / gross_loss)
    "max_drawdown_pct": -25.0,      # Maximum capital loss allowed
    "min_return_pct": 0.0,          # Must be positive
    "min_multi_asset_pass": 2,      # Pass on >= 2 out of 3 assets (BTC, ETH, SOL)
    "min_oos_sharpe": 0.5,          # Out-of-sample (lower bar; prevents overfit)
    "min_oos_return": 0.0           # Must stay positive OOS
}
```

### Backtest Parameters (src/agents/rbi_v3/harness.py)
```python
cash = 100_000.0              # Standard account size
commission = 0.00045          # 0.045% Hyperliquid taker fee
timeout_sec = 180             # Max execution time per strategy per dataset
```

### Data Partitioning (src/agents/rbi_v3/data_manager.py)
```python
SPLIT_DATE = "2024-07-01 00:00:00"   # In-sample / Out-of-sample boundary
START_DATE = "2022-01-01 00:00:00"   # Historical download start
DEFAULT_SYMBOLS = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT"}
DEFAULT_TIMEFRAMES = ["15m", "1h"]
```

### Model Escalation (src/agents/rbi_agent_v3.py)
```python
BASE_MODEL_CONFIG = {"type": "deepseek", "name": "deepseek-chat"}
ESCALATION_MODEL_CONFIG = {"type": "openai", "name": "gpt-4o"}
MAX_DEBUG_ITERATIONS = 5         # Max retries per strategy
ESCALATE_ON_ATTEMPT = 4          # Switch to GPT-4o after 4 failed attempts
```

## Strategy Code Conventions

All generated strategies must follow these patterns (checked by static linter and harness):

### Class Structure
```python
from backtesting import Strategy
import talib
import numpy as np
import pandas_ta as ta

class YourStrategyName(Strategy):
    # Parameters at class level (immutable during backtest)
    fast_period = 10
    slow_period = 30
    atr_period = 14
    risk_pct = 0.02
    atr_stop_mult = 1.5

    def init(self):
        # Compute all indicators in init() using self.I()
        self.sma_fast = self.I(talib.SMA, self.data.Close, timeperiod=self.fast_period)
        self.sma_slow = self.I(talib.SMA, self.data.Close, timeperiod=self.slow_period)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period)

    def next(self):
        # Use explicit indexing (no backtesting.lib.crossover)
        if len(self) < 2:
            return
        
        # Example: bullish cross
        if self.sma_fast[-2] < self.sma_slow[-2] and self.sma_fast[-1] > self.sma_slow[-1]:
            risk_per_unit = self.data.Close[-1] * 0.02
            size = int(round((self.equity * self.risk_pct) / risk_per_unit))
            if size > 0:
                stop_loss = self.data.Close[-1] - (self.atr[-1] * self.atr_stop_mult)
                self.buy(size=size, sl=stop_loss)
```

### Forbidden Patterns (will fail AST validation)
- Imports: `os`, `sys`, `subprocess`, `socket`, `requests`, `shutil`, `urllib`, `backtesting.lib.*`
- Functions: `eval()`, `exec()`, `open()`, `__import__()`
- Lookahead bias: `shift(-n)`, `center=True` in rolling windows
- Future-leaking indexing: `df[i+1]` or `self.data[len(self)]`

### Allowed Indicators
- TA-Lib: `talib.SMA()`, `talib.EMA()`, `talib.RSI()`, `talib.MACD()`, `talib.ATR()`, `talib.ADX()`, `talib.CCI()`, `talib.STOCH()`, etc.
- pandas-ta: `ta.rsi()`, `ta.macd()`, `ta.bbands()`, `ta.atr()`, `ta.pivot()`, etc.
- Custom NumPy: `np.roll()`, `np.where()`, `np.argmax()`, etc.

## Execution Flow (rbi_agent_v3.py)

### Phase 0: Jev Idea Triage (<150ms)
Filters ideas using TypeSafe Jev (deterministic System-1 model via OpenRouter) or local heuristic.
- **Rejects:** Too short (<15 chars), headers, generic prompts, non-actionable text
- **Accepts:** Ideas with trading indicators (RSI, EMA, breakout, volume, etc.) or Jev approval

### Phase 1: Research AI (DeepSeek)
LLM designs a quantitative specification: core concept, indicators, entry/exit rules, parameter ranges.
- Output format: `STRATEGY_NAME: [CamelCaseName]` followed by structured description
- Extracts strategy name and identifies style tag (Trend-Following, Mean-Reversion, Volatility-Breakout, Liquidity-Scalp, Momentum)

### Phase 2: Code Generation (DeepSeek)
LLM generates pure Python `Strategy` class code (no CSV loading, no Backtest instantiation).
- Constraints enforced in prompt; static linter validates output
- Output is cleaned (extracts code from markdown if present)

### Phases 3-5: Static Lint & Debug Loop
1. **Phase 3:** AST parsing checks syntax, forbidden imports, forbidden calls, lookahead bias
   - If valid, advance to harness execution
   - If invalid, capture error and proceed to debug
2. **Phase 4:** Harness execution on BTC in-sample (15m)
   - Subprocess runs code in isolated temp file with standardized parameters
   - Extracts JSON metrics from delimited output (`###JSON_OUTPUT_START###...###JSON_OUTPUT_END###`)
   - On success, break loop; on error, capture stderr for debugging
3. **Phase 5:** Debug loop (max 5 iterations)
   - Hash-based loop detection: if same code generated twice, force model escalation
   - Attempt 1-3: Use DeepSeek
   - Attempt 4+: Escalate to GPT-4o (gpt-4o)
   - If max iterations exceeded, record as `FAILED_DEBUG`

### Phase 6: In-Sample Gate (BTC, 15m)
Evaluates backtest metrics against `DEFAULT_GATES`.
- **Rejection Reason Examples:** "Insufficient trades", "Low Sharpe ratio", "Excessive drawdown"
- **Score:** Composite metric combining Sharpe, Profit Factor, drawdown, and return
- **Status:** `FAILED_IS` if rejected, else proceed

### Phase 7: Multi-Asset Validation (ETH, SOL 15m)
Runs same strategy code on ETH and SOL in-sample datasets.
- **Pass Criteria:** >= 2 out of 3 assets (BTC, ETH, SOL) show positive return and >= 15 trades
- **Rejection Reason:** "Overfitted to BTC" if multi-asset check fails
- **Status:** `FAILED_MULTI` if rejected, else proceed

### Phase 8: Out-of-Sample Blind Test (BTC OOS, 15m)
Runs strategy on 2024-07-01 to present data with frozen in-sample parameters.
- **Pass Criteria:** >= 10 trades, Sharpe >= 0.5, return >= 0%
- **Rejection Reason:** "Strategy degraded in OOS validation" typical failure
- **Status:** `FAILED_OOS` if rejected, else proceed to archiving

### Phase 9: Winner Archiving
Calculates composite total score and records `CERTIFIED_WINNER` entry.
- **Composite Score Formula:** `(is_sharpe * 2.0) + (oos_sharpe * 3.0) + (multi_pass_count * 1.5) + (is_ret * 0.02) + (oos_ret * 0.04) - (max_dd * 0.08)`
- Archives strategy code and metadata JSON to `winners/` directory
- Updates leaderboard CSV (auto-sorts by total_score)
- Exports markdown leaderboard

## State Tracking & Idempotency

- **state.jsonl:** Append-only log; one JSON record per processed idea
- **Processed Hashes:** SHA256 hashes of idea text prevent re-processing same idea
- **Leaderboard CSV:** Keyed by `unique_id` (idea hash); duplicate entries update existing rows
- **Recovery:** Restart script safely; only unprocessed ideas are reprocessed

## Debugging Tips

### Common Failures

**"No class inheriting from 'Strategy' found"**
- Ensure generated code has exactly one class inheriting from `Strategy`
- Check that prompt is requesting class definition

**"Forbidden imports detected"**
- Code generator mistakenly added `import os` or `import requests`
- Debug AI should catch and remove; if not, escalate to GPT-4o

**"Execution timeout (180s exceeded)"**
- Strategy has inefficient logic or backtesting.py crashed
- Check for infinite loops in indicator calculation

**"Insufficient trades / Low Sharpe ratio"**
- Strategy is not generating entries, or entries are too infrequent
- Common on OOS data if strategy overfits to in-sample patterns

**"OOS negative return"**
- Classic overfitting signature; parameters optimized for 2022-2024 don't generalize
- May need different entry/exit thresholds or looser parameter ranges

### Manual Testing

```bash
# Test data download
python -c "from src.agents.rbi_v3 import DataManager; dm = DataManager(); dm.download_symbol_history('BTC', '15m')"

# Test code validation
python -c "
from src.agents.rbi_v3 import validate_strategy_code
code = '''
from backtesting import Strategy
import talib
class TestStrat(Strategy):
    def init(self):
        self.ma = self.I(talib.SMA, self.data.Close, 20)
    def next(self):
        if len(self) < 2: return
        if self.ma[-2] < self.ma[-1]:
            self.buy()
'''
result = validate_strategy_code(code)
print(result)
"

# Test harness directly
python -c "
from src.agents.rbi_v3 import run_backtest_harness
result = run_backtest_harness(
    strategy_code='...',
    class_name='TestStrat',
    data_path='src/data/rbi_v3/market_data/BTC-USD-15m_IS.csv'
)
print(result)
"
```

## Dependencies

- **backtesting.py:** Vectorized backtesting engine (data alignment, order execution simulation)
- **pandas / numpy:** Data manipulation
- **talib / pandas_ta:** Technical indicator computation
- **requests:** HTTP for Binance API and OpenRouter
- **termcolor:** Colored console output (Moon Dev style)
- **anthropic / openai / deepseek SDKs:** LLM calls via ModelFactory (unified interface in `src/models/`)

All deps are in repo `requirements.txt`. Ensure `conda activate tflow` before running.

## Extension Points

**Custom Gating Logic:**
Edit `DEFAULT_GATES` in `evaluator.py` or subclass `StrategyEvaluator` with custom `evaluate_*` methods.

**New Assets/Timeframes:**
Add to `DEFAULT_SYMBOLS` and `DEFAULT_TIMEFRAMES` in `data_manager.py`; DataManager auto-downloads.

**Alternative Idea Source:**
Modify `rbi_agent_v3.py:run()` to read ideas from database, API, etc. instead of text file.

**Custom LLM Models:**
Change `BASE_MODEL_CONFIG` and `ESCALATION_MODEL_CONFIG` in `rbi_agent_v3.py`; ModelFactory supports any registered provider.

## Moon Dev Style Guide

- Use `termcolor.cprint()` with color args for logging; emit emoji-heavy messages
- Docstrings: one-liner above function, optional emoji summary below
- State results directly; avoid narrative prose in console output
- Always include context (phase number, strategy name, metric values) in user-facing output
