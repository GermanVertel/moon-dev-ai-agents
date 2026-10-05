"""
🌙 Moon Dev's RBI AI v3.0 (Research - Backtest - Validate - MultiAsset - OOS)
Built with love by Moon Dev 🚀

Features in v3.0:
1. Phase 0: TypeSafe Jev ultra-fast Idea Triage (<150ms filter).
2. Phase 1: DeepSeek Research AI (Structured rules & parameter ranges).
3. Phase 2: Pure Strategy class generation (No boilerplate I/O or CSV loading).
4. Phase 3: Static AST linter & lookahead-bias detection.
5. Phase 4: Standardized Backtest Harness (Hyperliquid 0.045% commission, $100k cash).
6. Phase 5: Debug loop with SHA anti-loop hashing & escalation to OpenAI (gpt-4o / o3-mini).
7. Phase 6: Statistical Gates (In-Sample BTC 100+ trades, Sharpe >= 1.0, Profit Factor >= 1.3).
8. Phase 7: Multi-Asset Validation (ETH & SOL cross-validation).
9. Phase 8: Out-of-Sample Blind Test (2024-2026 data with frozen parameters).
10. Phase 9: Real-time Leaderboard & Certified Winner archiving.
"""

import os
import sys
import time
import re
import json
import hashlib
from datetime import datetime
from pathlib import Path
from termcolor import cprint

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

from src.models import model_factory
from src.agents.rbi_v3.data_manager import DataManager
from src.agents.rbi_v3.code_checks import validate_strategy_code
from src.agents.rbi_v3.harness import run_backtest_harness
from src.agents.rbi_v3.evaluator import StrategyEvaluator
from src.agents.rbi_v3.jev_gate import JevGatekeeper
from src.agents.rbi_v3.leaderboard import LeaderboardManager
from src.agents.rbi_v3.prompts import RESEARCH_PROMPT, STRATEGY_CODE_PROMPT, DEBUG_PROMPT

# Output Directories
RBI_V3_DATA_DIR = PROJECT_ROOT / "src" / "data" / "rbi_v3"
MARKET_DATA_DIR = RBI_V3_DATA_DIR / "market_data"
STRATEGIES_DIR = RBI_V3_DATA_DIR / "strategies"
WINNERS_DIR = RBI_V3_DATA_DIR / "winners"
STATE_LOG_PATH = RBI_V3_DATA_DIR / "state.jsonl"
IDEAS_FILE_V3 = RBI_V3_DATA_DIR / "ideas.txt"
IDEAS_FILE_V2 = PROJECT_ROOT / "src" / "data" / "rbi_v2" / "ideas.txt"

# Model Configurations (OpenRouter unified routing or direct API keys)
USE_OPENROUTER_FOR_ALL = os.getenv("USE_OPENROUTER_FOR_ALL", "false").lower() in ("true", "1")
HAS_OPENROUTER_KEY = bool(os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_KEY"))

# Base Model (DeepSeek chat for research and initial code generation)
if USE_OPENROUTER_FOR_ALL and HAS_OPENROUTER_KEY:
    BASE_MODEL_CONFIG = {"type": "openrouter", "name": "deepseek/deepseek-chat"}
else:
    BASE_MODEL_CONFIG = {"type": "deepseek", "name": "deepseek-chat"}

# Escalation Model (GPT-4o or Claude 3.5 Sonnet for difficult debugging)
DEFAULT_ESCALATION = "openai/gpt-4o" if HAS_OPENROUTER_KEY else "gpt-4o"
ESCALATION_MODEL_NAME = os.getenv("RBI_ESCALATION_MODEL", DEFAULT_ESCALATION)
ESCALATION_TYPE = "openrouter" if ("/" in ESCALATION_MODEL_NAME or HAS_OPENROUTER_KEY) else "openai"

ESCALATION_MODEL_CONFIG = {
    "type": ESCALATION_MODEL_TYPE,
    "name": ESCALATION_MODEL_NAME
}

MAX_DEBUG_ITERATIONS = 4        # Reduced from 5 for efficiency
ESCALATE_ON_ATTEMPT = 3         # Escalate to OpenRouter/GPT-4o/Claude after 2 failed base attempts
GLOBAL_MODEL_TIMEOUT = 90       # Timeout for LLM calls (seconds)


class RBIEngineV3:
    """Master orchestrator for the RBI v3 Quantitative Research Pipeline."""

    def __init__(self):
        cprint("\n🌙 Moon Dev's RBI v3 Engine Initializing...", "white", "on_blue")
        self.data_mgr = DataManager(MARKET_DATA_DIR)
        self.evaluator = StrategyEvaluator()
        self.jev = JevGatekeeper()
        self.leaderboard = LeaderboardManager(RBI_V3_DATA_DIR)
        
        # Ensure directories
        for d in [RBI_V3_DATA_DIR, MARKET_DATA_DIR, STRATEGIES_DIR, WINNERS_DIR]:
            d.mkdir(parents=True, exist_ok=True)

        self._ensure_market_data()
        self.processed_hashes = self._load_processed_hashes()

    def _ensure_market_data(self):
        """Ensure BTC, ETH, SOL datasets exist."""
        cprint("📊 Checking multi-asset market datasets (BTC, ETH, SOL)...", "cyan")
        self.data_mgr.ensure_datasets(symbols=["BTC", "ETH", "SOL"], timeframes=["15m", "1h"])

    def _load_processed_hashes(self) -> set:
        """Load set of already processed idea hashes from state log."""
        hashes = set()
        if STATE_LOG_PATH.exists():
            with open(STATE_LOG_PATH, "r") as f:
                for line in f:
                    if line.strip():
                        try:
                            record = json.loads(line)
                            hashes.add(record.get("idea_hash"))
                        except Exception:
                            pass
        return hashes

    def _log_state(self, idea_hash: str, strategy_name: str, status: str, details: dict):
        """Append state record to state.jsonl."""
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "idea_hash": idea_hash,
            "strategy_name": strategy_name,
            "status": status,
            "details": details
        }
        with open(STATE_LOG_PATH, "a") as f:
            f.write(json.dumps(record) + "\n")
        self.processed_hashes.add(idea_hash)

    def _call_model(self, system_prompt: str, user_content: str, model_config: dict) -> str:
        """Call LLM via unified ModelFactory with fallback resilience and timeout."""
        model = model_factory.get_model(model_config["type"], model_config["name"])
        if not model:
            cprint(f"⚠️ Model {model_config['name']} unavailable, falling back to deepseek-chat", "yellow")
            model = model_factory.get_model("deepseek", "deepseek-chat")

        try:
            res = model.generate_response(
                system_prompt=system_prompt,
                user_content=user_content,
                temperature=0.7,
                max_tokens=4000,
                timeout=GLOBAL_MODEL_TIMEOUT
            )
        except Exception as e:
            cprint(f"⚠️ Model timeout or error: {str(e)[:100]}", "yellow")
            raise

        if isinstance(res, str):
            return res
        return getattr(res, "content", str(res))

    def _clean_code_output(self, output: str) -> str:
        """Extract pure python code from LLM response."""
        code = output.strip()
        if "<think>" in code and "</think>" in code:
            code = code.split("</think>")[-1].strip()
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0].strip()
        elif "```" in code:
            code = code.split("```")[1].split("```")[0].strip()
        return code

    def process_idea(self, idea_text: str, idea_idx: int, total_ideas: int):
        """Process a single trading idea through all RBI v3 quantitative phases."""
        idea_hash = hashlib.md5(idea_text.encode("utf-8")).hexdigest()[:8]
        
        cprint(f"\n{'='*70}", "yellow")
        cprint(f"🌙 RBI v3 [{idea_idx}/{total_ideas}] | Hash: {idea_hash}", "cyan")
        cprint(f"📝 Idea: {idea_text[:120]}...", "yellow")
        cprint(f"{'='*70}", "yellow")

        # Phase 0: Jev Fast Idea Triage
        is_actionable, triage_reason = self.jev.is_actionable_idea(idea_text)
        if not is_actionable:
            cprint(f"⏭️ Phase 0 Skipped by Jev: {triage_reason}", "yellow")
            self._log_state(idea_hash, "Skipped", "JEV_REJECTED", {"reason": triage_reason})
            return

        cprint(f"✅ Phase 0 Jev Triage: {triage_reason}", "green")

        # Phase 1: Research AI
        cprint("\n🧪 Phase 1: Quantitative Research AI (DeepSeek)...", "cyan")
        research_raw = self._call_model(RESEARCH_PROMPT, f"Design a trading strategy for: {idea_text}", BASE_MODEL_CONFIG)
        
        strategy_name = "AdaptiveQuant"
        if "STRATEGY_NAME:" in research_raw:
            try:
                name_match = research_raw.split("STRATEGY_NAME:")[1].strip().split("\n")[0]
                strategy_name = re.sub(r"[^\w]", "", name_match)
            except Exception:
                pass

        style_tag = self.jev.classify_strategy_type(research_raw)
        cprint(f"🏷️ Strategy Name: {strategy_name} | Style: {style_tag}", "green")

        # Phase 2: Strategy Code Generation
        cprint("\n⚡ Phase 2: Generating pure Strategy class code...", "cyan")
        code_prompt_user = f"Write the backtesting.py Strategy class for this design:\n\n{research_raw}"
        code_raw = self._call_model(STRATEGY_CODE_PROMPT, code_prompt_user, BASE_MODEL_CONFIG)
        strategy_code = self._clean_code_output(code_raw)

        # Phase 3 & 5: Static Linting + Debug Loop with Model Escalation
        current_code = strategy_code
        debug_attempt = 0
        harness_result = {}
        validated_class_name = strategy_name

        btc_is_data_path = str(MARKET_DATA_DIR / "BTC-USD-15m_IS.csv")
        code_hashes_seen = set()

        while debug_attempt < MAX_DEBUG_ITERATIONS:
            # 3. Static AST validation
            validation = validate_strategy_code(current_code)
            if not validation.is_valid:
                cprint(f"🔍 Static Lint Error: {validation.error_message}", "yellow")
                err_msg = validation.error_message
            else:
                validated_class_name = validation.class_name or strategy_name
                # 4. Harness execution on BTC In-Sample
                cprint(f"🚀 Running Backtest Harness (Attempt {debug_attempt+1}/{MAX_DEBUG_ITERATIONS})...", "cyan")
                harness_result = run_backtest_harness(
                    strategy_code=current_code,
                    class_name=validated_class_name,
                    data_path=btc_is_data_path,
                    conda_env="tflow"
                )

                if harness_result.get("success"):
                    cprint("✨ Harness execution succeeded without runtime exceptions!", "green")
                    break
                else:
                    err_msg = harness_result.get("error", "Unknown harness error")
                    cprint(f"🐛 Execution Error: {err_msg[:200]}", "red")

            # Check for infinite loop on identical code
            code_hash = hashlib.md5(current_code.encode("utf-8")).hexdigest()
            if code_hash in code_hashes_seen:
                cprint("🛑 Repeated code generated by Debug AI; forcing model escalation...", "yellow")
                active_model = ESCALATION_MODEL_CONFIG
            else:
                code_hashes_seen.add(code_hash)
                active_model = ESCALATION_MODEL_CONFIG if (debug_attempt + 1) >= ESCALATE_ON_ATTEMPT else BASE_MODEL_CONFIG

            debug_attempt += 1
            if debug_attempt >= MAX_DEBUG_ITERATIONS:
                cprint(f"❌ Max debug iterations reached for {strategy_name}", "red")
                self.leaderboard.record_entry({
                    "strategy_name": strategy_name,
                    "unique_id": idea_hash,
                    "style_tag": style_tag,
                    "status": "FAILED_DEBUG",
                    "rejection_reason": f"Debug failed after {MAX_DEBUG_ITERATIONS} iterations",
                    "timestamp": datetime.utcnow().isoformat()
                })
                self._log_state(idea_hash, strategy_name, "FAILED_DEBUG", {"attempts": debug_attempt})
                return

            cprint(f"🔧 Debug AI repairing code using {active_model['name']} (attempt {debug_attempt})...", "yellow")
            debug_input = DEBUG_PROMPT.format(error_message=err_msg, failed_code=current_code)
            fixed_code_raw = self._call_model("You are an expert Python algorithmic debug assistant.", debug_input, active_model)
            current_code = self._clean_code_output(fixed_code_raw)

        # Save working strategy code file
        strategy_file = STRATEGIES_DIR / f"{strategy_name}_{idea_hash}.py"
        with open(strategy_file, "w") as f:
            f.write(current_code)

        # Phase 6: Quantitative Statistical Gate (In-Sample BTC)
        cprint("\n📊 Phase 6: Evaluating Quantitative Statistical Gates (In-Sample)...", "cyan")
        is_passed, is_reason, is_score = self.evaluator.evaluate_in_sample(harness_result)
        
        cprint(f"   • Return: {harness_result.get('return_pct', 0):.2f}% | Trades: {harness_result.get('trades', 0)}", "cyan")
        cprint(f"   • Sharpe: {harness_result.get('sharpe', 0):.2f} | Profit Factor: {harness_result.get('profit_factor', 0):.2f}", "cyan")
        cprint(f"   • Max Drawdown: {harness_result.get('max_dd_pct', 0):.2f}%", "cyan")

        if not is_passed:
            cprint(f"❌ Rejected In-Sample: {is_reason}", "yellow")
            self.leaderboard.record_entry({
                "strategy_name": strategy_name,
                "unique_id": idea_hash,
                "style_tag": style_tag,
                "status": "FAILED_IS",
                "is_return_pct": harness_result.get("return_pct"),
                "is_sharpe": harness_result.get("sharpe"),
                "is_max_dd_pct": harness_result.get("max_dd_pct"),
                "is_trades": harness_result.get("trades"),
                "is_win_rate": harness_result.get("win_rate"),
                "is_pf": harness_result.get("profit_factor"),
                "total_score": is_score,
                "rejection_reason": is_reason,
                "timestamp": datetime.utcnow().isoformat()
            }, current_code)
            self._log_state(idea_hash, strategy_name, "FAILED_IS", {"reason": is_reason})
            return

        cprint(f"🎉 PASSED IN-SAMPLE GATE! Score: {is_score}", "green", attrs=["bold"])

        # Phase 7: Multi-Asset Validation (ETH and SOL)
        cprint("\n🌐 Phase 7: Multi-Asset Robustness Test (ETH & SOL)...", "cyan")
        eth_path = str(MARKET_DATA_DIR / "ETH-USD-15m_IS.csv")
        sol_path = str(MARKET_DATA_DIR / "SOL-USD-15m_IS.csv")

        eth_res = run_backtest_harness(current_code, validated_class_name, eth_path, conda_env="tflow")
        sol_res = run_backtest_harness(current_code, validated_class_name, sol_path, conda_env="tflow")

        multi_results = {"BTC": harness_result, "ETH": eth_res, "SOL": sol_res}
        multi_passed, multi_count, multi_summary = self.evaluator.evaluate_multi_asset(multi_results)
        cprint(f"   • Multi-Asset Result: {multi_summary}", "green" if multi_passed else "yellow")

        if not multi_passed:
            cprint("❌ Rejected: Overfitted to BTC (failed cross-asset test)", "yellow")
            self.leaderboard.record_entry({
                "strategy_name": strategy_name,
                "unique_id": idea_hash,
                "style_tag": style_tag,
                "status": "FAILED_MULTI",
                "is_return_pct": harness_result.get("return_pct"),
                "is_sharpe": harness_result.get("sharpe"),
                "is_max_dd_pct": harness_result.get("max_dd_pct"),
                "is_trades": harness_result.get("trades"),
                "multi_pass_count": f"{multi_count}/3",
                "total_score": is_score,
                "rejection_reason": "Failed multi-asset validation",
                "timestamp": datetime.utcnow().isoformat()
            }, current_code)
            self._log_state(idea_hash, strategy_name, "FAILED_MULTI", {"pass_count": multi_count})
            return

        # Phase 8: Out-of-Sample Blind Test
        cprint("\n🔒 Phase 8: Out-of-Sample Blind Validation (2024-2026 Data)...", "cyan")
        btc_oos_path = str(MARKET_DATA_DIR / "BTC-USD-15m_OOS.csv")
        oos_res = run_backtest_harness(current_code, validated_class_name, btc_oos_path, conda_env="tflow")
        
        oos_passed, oos_reason = self.evaluator.evaluate_out_of_sample(oos_res)
        cprint(f"   • Out-Of-Sample Result: {oos_reason}", "green" if oos_passed else "red")

        if not oos_passed:
            cprint("❌ Rejected: Strategy degraded in Out-Of-Sample validation", "yellow")
            self.leaderboard.record_entry({
                "strategy_name": strategy_name,
                "unique_id": idea_hash,
                "style_tag": style_tag,
                "status": "FAILED_OOS",
                "is_return_pct": harness_result.get("return_pct"),
                "is_sharpe": harness_result.get("sharpe"),
                "multi_pass_count": f"{multi_count}/3",
                "oos_return_pct": oos_res.get("return_pct"),
                "oos_sharpe": oos_res.get("sharpe"),
                "oos_trades": oos_res.get("trades"),
                "total_score": is_score,
                "rejection_reason": oos_reason,
                "timestamp": datetime.utcnow().isoformat()
            }, current_code)
            self._log_state(idea_hash, strategy_name, "FAILED_OOS", {"reason": oos_reason})
            return

        # Phase 9: Certified Winner Archiving
        total_score = self.evaluator.calculate_total_score(harness_result, multi_count, oos_res)
        cprint(f"\n🏆 STRATEGY CERTIFIED AS PRODUCTION READY! Composite Score: {total_score} 🚀", "green", "on_black", attrs=["bold"])

        self.leaderboard.record_entry({
            "strategy_name": strategy_name,
            "unique_id": idea_hash,
            "style_tag": style_tag,
            "status": "CERTIFIED_WINNER",
            "total_score": total_score,
            "is_return_pct": harness_result.get("return_pct"),
            "is_sharpe": harness_result.get("sharpe"),
            "is_max_dd_pct": harness_result.get("max_dd_pct"),
            "is_trades": harness_result.get("trades"),
            "is_win_rate": harness_result.get("win_rate"),
            "is_pf": harness_result.get("profit_factor"),
            "multi_pass_count": f"{multi_count}/3",
            "oos_return_pct": oos_res.get("return_pct"),
            "oos_sharpe": oos_res.get("sharpe"),
            "oos_trades": oos_res.get("trades"),
            "rejection_reason": "None - Certified Winner",
            "timestamp": datetime.utcnow().isoformat()
        }, current_code)

        self._log_state(idea_hash, strategy_name, "CERTIFIED_WINNER", {"score": total_score})

    def run(self):
        """Execute RBI v3 loop over ideas file with exponential backoff on errors."""
        ideas_file = IDEAS_FILE_V3 if IDEAS_FILE_V3.exists() else IDEAS_FILE_V2
        if not ideas_file.exists():
            cprint(f"❌ ideas.txt not found at {ideas_file}", "red")
            return

        with open(ideas_file, "r") as f:
            ideas = [line.strip() for line in f if line.strip() and not line.startswith("#")]

        total_ideas = len(ideas)
        cprint(f"🎯 Loaded {total_ideas} raw ideas from {ideas_file.name}", "cyan")

        max_ideas_env = os.getenv("RBI_MAX_IDEAS")
        max_ideas = int(max_ideas_env) if max_ideas_env and max_ideas_env.isdigit() else None
        processed_count = 0
        consecutive_errors = 0
        max_consecutive_errors = 3

        for idx, idea in enumerate(ideas, 1):
            idea_hash = hashlib.md5(idea.encode("utf-8")).hexdigest()[:8]
            if idea_hash in self.processed_hashes:
                continue

            try:
                self.process_idea(idea, idx, total_ideas)
                processed_count += 1
                consecutive_errors = 0  # Reset error counter on success
                if max_ideas and processed_count >= max_ideas:
                    cprint(f"\n🛑 Reached RBI_MAX_IDEAS limit ({max_ideas}). Stopping.", "yellow")
                    break
            except Exception as e:
                consecutive_errors += 1
                backoff_sec = min(300, 2 ** consecutive_errors)  # Max 5 min wait
                cprint(f"\n❌ Error in idea {idx}: {str(e)[:100]}", "red")
                cprint(f"⏳ Backing off {backoff_sec}s (error {consecutive_errors}/{max_consecutive_errors})...", "yellow")
                time.sleep(backoff_sec)
                if consecutive_errors >= max_consecutive_errors:
                    cprint(f"🛑 Too many consecutive errors ({max_consecutive_errors}). Stopping.", "red")
                    break
                continue


if __name__ == "__main__":
    engine = RBIEngineV3()
    engine.run()
