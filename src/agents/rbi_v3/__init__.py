"""
🌙 Moon Dev's RBI v3 Package
State of the art quantitative research, backtesting, and strategy validation engine.
"""

from .harness import run_backtest_harness
from .evaluator import StrategyEvaluator
from .data_manager import DataManager
from .code_checks import validate_strategy_code
from .leaderboard import LeaderboardManager
from .idea_generator import IdeaGeneratorV3
