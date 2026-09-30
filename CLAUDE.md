# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Moon Dev's AI agents for trading: a collection of mostly standalone Python agents (trading, risk, strategy, whale/funding/liquidation monitors, RBI backtest generator, stream chat bot, clips/tweet/video agents, etc.). Experimental research code — agents can place real trades on Solana (via `nice_funcs.py`) and Hyperliquid (via `nice_funcs_hl.py`), so treat anything touching execution with care.

## Environment & Running

- Python 3.10.9. Use the existing conda env: `conda activate tflow`. Do not create new virtual environments.
- Install deps: `pip install -r requirements.txt`. **Add any new package to `requirements.txt`** when you introduce it.
- Secrets live in `.env` at the repo root (template: `.env_example`). Model keys are `ANTHROPIC_KEY`, `OPENAI_KEY`, `DEEPSEEK_KEY`, `GROQ_API_KEY`, `GEMINI_KEY`, `GROK_API_KEY`, `OPENROUTER_API_KEY`; trading uses `BIRDEYE_API_KEY`, `RPC_ENDPOINT`, `SOLANA_PRIVATE_KEY`, `HYPER_LIQUID_ETH_PRIVATE_KEY`, `MOONDEV_API_KEY`.
- Orchestrator: `python src/main.py` — runs the agents enabled in its `ACTIVE_AGENTS` dict (risk → trading → strategy → copybot → sentiment) in a loop, sleeping `SLEEP_BETWEEN_RUNS_MINUTES`. All are off by default.
- Most agents are run individually. They import via `from src.config import *` / `from src import nice_funcs`, so run from the repo root as a module: `python -m src.agents.<agent_name>` (only some agents add the project root to `sys.path` themselves).
- There is no test suite, linter, or build step. Files named `test_*.py` in `src/agents/` are ad-hoc manual scripts, run the same way as agents.

## Architecture

- `src/config.py` — global settings imported with `*` by agents: monitored/excluded token addresses (USDC and SOL must never be traded/closed), position sizing, risk limits, slippage, data timeframe, default `AI_MODEL`. Many agents also define their own config constants at the top of their file.
- `src/nice_funcs.py` / `src/nice_funcs_hl.py` — shared trading & data toolkit (Birdeye token data/OHLCV, Jupiter market buy/sell, positions, `chunk_kill`, `close_all_positions`, entry helpers). Agents import these as `n`.
- `src/models/` — **model factory**: unified LLM interface. `ModelFactory` loads `.env`, initializes every provider whose key exists (`claude`, `openai`, `deepseek`, `groq`, `ollama`, `openrouter`; Gemini currently disabled due to a protobuf conflict), and exposes `get_model(type, name)`; each model implements `generate_response(system_prompt, user_content, temperature, max_tokens)`. Use this when adding AI calls to an agent rather than calling SDKs directly. See `src/models/README.md` for model names.
- `src/agents/` — one file per agent, each with its own prompts and config block at the top. `base_agent.py` is a minimal parent class with a `run()` method; not all agents use it. `src/agents/README.md` has the build list.
- `src/strategies/` — `BaseStrategy` with `generate_signals()` returning `{token, signal (0-1), direction, metadata}`. Custom strategies go in `src/strategies/custom/`; `strategy_agent.py` loads them and has an LLM approve/reject signals before execution.
- `src/data/` — agent outputs and state (CSVs, charts, generated videos). RBI output is organized as `src/data/rbi/MM_DD_YYYY/{research,backtests,backtests_package,backtests_final,charts}` with inputs from `src/data/rbi/ideas.txt` (one YouTube URL / PDF / text idea per line); `research_agent.py` fills `ideas.txt`. Large media files (e.g. `src/data/chat_agent/*.mov`) are git-ignored.
- `src/scripts/` — one-off utilities (local DeepSeek API server/client, code-folder-to-txt, CoinGecko examples, etc.).

## Conventions (from .cursorrules)

- Keep each file under 800 lines; if longer, split into a new file and update the README to explain it.
- Don't move or rename existing files without asking (creating new files is fine).
- Backtests use `backtesting.py`, but compute indicators with `pandas_ta` or `talib`, not backtesting.py's built-ins. Sample OHLCV data: `src/data/rbi/BTC-USD-15m.csv`.
- Match the existing style: `termcolor.cprint` for colored console output and the "🌙 Moon Dev" emoji-heavy docstrings/log messages.
