"""
🌙 Moon Dev's RBI v3 Strategy Idea Generator
Built with love by Moon Dev 🚀

Generates quantitative trading ideas specifically tailored for RBI v3:
- Strictly relies on continuous OHLCV data (15m & 1h crypto: BTC, ETH, SOL).
- Zero external dependencies (no VIX, sentiment, funding rate, or order book).
- Targets actionable, robust logic with 100+ trades expected over 2.5 years.
- Automatically triaged through Jev Gatekeeper before persisting.
- Deduplicates against existing RBI v3 and v2 ideas.
"""

import os
import sys
import time
import re
import csv
import random
import hashlib
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Set
from termcolor import cprint

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

from src.models import model_factory
from src.agents.rbi_v3.jev_gate import JevGatekeeper

RBI_V3_DATA_DIR = PROJECT_ROOT / "src" / "data" / "rbi_v3"
IDEAS_V3_FILE = RBI_V3_DATA_DIR / "ideas.txt"
IDEAS_V2_FILE = PROJECT_ROOT / "src" / "data" / "rbi_v2" / "ideas.txt"
IDEAS_CSV_LOG = RBI_V3_DATA_DIR / "generated_ideas.csv"

# Diverse quantitative archetypes to ensure variety
IDEA_ARCHETYPES = [
    {
        "category": "Mean Reversion & Volatility Bands",
        "focus": "Bollinger Bands, Keltner Channel squeezes, RSI oversold/overbought with volume confirmation, ATR mean reversion."
    },
    {
        "category": "Trend Following & Momentum",
        "focus": "Multi-timeframe EMA alignment (e.g. 21/55/200), Supertrend combined with ADX regime filter, MACD momentum continuation."
    },
    {
        "category": "Volatility Breakout & Volume Z-Score",
        "focus": "Donchian channel breakout confirmed by Volume Z-score > 1.5, Bollinger Bandwidth squeeze followed by directional expansion."
    },
    {
        "category": "Oscillator Pullbacks in Trend",
        "focus": "Stochastic RSI or Williams %R pullbacks within an established EMA trend, taking entries on momentum resumption."
    },
    {
        "category": "Price Action & Statistical Regimes",
        "focus": "Rolling ATR realized volatility spike with candle wick rejection, VWAP proxy distance reversion."
    }
]

GENERATOR_PROMPT_TEMPLATE = """You are Moon Dev's Quantitative Trading Idea Architect 🌙

TASK: Formulate ONE specific, actionable quantitative trading idea for intraday crypto (15m timeframe on BTC/ETH/SOL).

THEME: {category}
INSPIRATION: {focus}

STRICT CONSTRAINTS:
1. DATA AVAILABLE: Only OHLCV candles (Open, High, Low, Close, Volume).
   NEVER mention or require VIX, funding rates, open interest, order book depth, liquidations, on-chain data, or news sentiment.
   If you need volatility, use ATR or historical standard deviation.
   If you need volume flow, use volume moving averages, Volume Z-score, or OBV.
2. MUST TRADE REGULARLY: Design rules with at most 2-3 clean entry conditions combined with 'and', so the strategy fires 100+ times across 2.5 years of 15m data.
3. CONCISE FORMAT: Output ONLY 1 to 2 sentences describing the entry setup, indicators used, and the exit rule (stop loss / profit target / indicator exit).
4. No intros, no bullet points, no markdown headers, no code. Just the raw idea text in 1-2 clear sentences.

Example good response:
"Go long when Close crosses above the 50 EMA while 14-period RSI is between 45 and 65 and Volume exceeds the 20-period SMA, with a 2x ATR trailing stop and exit when Close crosses below the 20 EMA."
"""


class IdeaGeneratorV3:
    """Quantitative Strategy Idea Generator for RBI v3."""

    def __init__(self, model_type: Optional[str] = None, model_name: Optional[str] = None):
        RBI_V3_DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.jev = JevGatekeeper()
        
        # Configure model (OpenRouter or DeepSeek)
        has_openrouter = bool(os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_KEY"))
        use_openrouter_all = os.getenv("USE_OPENROUTER_FOR_ALL", "false").lower() in ("true", "1")
        
        if model_type and model_name:
            self.model_type = model_type
            self.model_name = model_name
        elif use_openrouter_all and has_openrouter:
            self.model_type = "openrouter"
            self.model_name = "deepseek/deepseek-chat"
        elif os.getenv("DEEPSEEK_KEY"):
            self.model_type = "deepseek"
            self.model_name = "deepseek-chat"
        elif has_openrouter:
            self.model_type = "openrouter"
            self.model_name = "openai/gpt-4o-mini"
        else:
            self.model_type = "openai"
            self.model_name = "gpt-4o"

        self.existing_hashes = self._load_existing_hashes()
        cprint(f"✨ RBI v3 Idea Generator Initialized with {len(self.existing_hashes)} existing idea hashes.", "cyan")
        cprint(f"🤖 LLM Engine: {self.model_type} ({self.model_name})", "cyan")

    def _load_existing_hashes(self) -> Set[str]:
        """Collect hashes from both v3 and v2 to prevent duplicates."""
        hashes = set()
        for filepath in [IDEAS_V3_FILE, IDEAS_V2_FILE]:
            if filepath.exists():
                try:
                    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                        for line in f:
                            cleaned = line.strip().lower()
                            if cleaned and not cleaned.startswith("#"):
                                hashes.add(hashlib.md5(cleaned.encode("utf-8")).hexdigest())
                except Exception:
                    pass
        return hashes

    def _call_llm(self, prompt: str) -> str:
        """Query LLM via ModelFactory."""
        model = model_factory.get_model(self.model_type, self.model_name)
        if not model:
            cprint(f"⚠️ Failed to get model {self.model_type}/{self.model_name}, trying fallback", "yellow")
            model = model_factory.get_model("deepseek", "deepseek-chat") or model_factory.get_model("openai", "gpt-4o")

        response = model.generate_response(
            system_prompt="You are Moon Dev's quantitative algorithmic idea specialist. Output ONLY the 1-2 sentence idea.",
            user_content=prompt,
            temperature=0.8,
            max_tokens=250
        )
        content = getattr(response, "content", str(response)).strip()
        # Clean reasoning tags or quotes if any
        if "<think>" in content and "</think>" in content:
            content = content.split("</think>")[-1].strip()
        content = re.sub(r'^["\']|["\']$', '', content).strip()
        return content

    def generate_single_idea(self, archetype_idx: Optional[int] = None) -> Optional[str]:
        """Generate, validate, and return a single unique idea."""
        archetype = IDEA_ARCHETYPES[archetype_idx % len(IDEA_ARCHETYPES)] if archetype_idx is not None else random.choice(IDEA_ARCHETYPES)
        prompt = GENERATOR_PROMPT_TEMPLATE.format(category=archetype["category"], focus=archetype["focus"])

        try:
            raw_idea = self._call_llm(prompt)
            if not raw_idea or len(raw_idea) < 25:
                return None

            # 1. Deduplication check
            idea_hash = hashlib.md5(raw_idea.strip().lower().encode("utf-8")).hexdigest()
            if idea_hash in self.existing_hashes:
                cprint("⏭️ Duplicate idea generated; skipping...", "yellow")
                return None

            # 2. Phase 0 Triage via Jev Gatekeeper
            is_valid, triage_reason = self.jev.is_actionable_idea(raw_idea)
            if not is_valid:
                cprint(f"❌ Rejected by Jev Gate: {triage_reason}", "yellow")
                return None

            # 3. Success: register and persist
            self.existing_hashes.add(idea_hash)
            self._save_idea(raw_idea, archetype["category"])
            return raw_idea

        except Exception as e:
            cprint(f"⚠️ Error generating idea: {str(e)[:100]}", "red")
            return None

    def _save_idea(self, idea_text: str, category: str):
        """Append to ideas.txt and log to generated_ideas.csv."""
        # Append to ideas.txt
        with open(IDEAS_V3_FILE, "a", encoding="utf-8") as f:
            f.write(idea_text.strip() + "\n")

        # Append to CSV log
        write_header = not IDEAS_CSV_LOG.exists()
        with open(IDEAS_CSV_LOG, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["timestamp", "category", "model", "idea"])
            writer.writerow([datetime.utcnow().isoformat(), category, f"{self.model_type}:{self.model_name}", idea_text.strip()])

    def generate_batch(self, count: int = 5) -> List[str]:
        """Generate a specified batch of ideas."""
        cprint(f"\n🚀 Generating {count} quantitative ideas for RBI v3...", "cyan")
        successful = []
        attempts = 0
        max_attempts = count * 4

        while len(successful) < count and attempts < max_attempts:
            attempts += 1
            idx = len(successful)
            idea = self.generate_single_idea(archetype_idx=idx)
            if idea:
                successful.append(idea)
                cprint(f"\n[{len(successful)}/{count}] ✨ Added Idea:", "green")
                cprint(f"   \"{idea}\"", "white")
            time.sleep(0.5)

        cprint(f"\n✅ Completed batch: {len(successful)} ideas saved to {IDEAS_V3_FILE.name}", "green", "on_blue")
        return successful

    def run_continuous(self, interval_sec: int = 15):
        """Run continuous idea generation loop."""
        cprint("\n🌙 Starting RBI v3 Continuous Idea Generator (Press Ctrl+C to stop)...", "white", "on_magenta")
        count = 0
        try:
            while True:
                idea = self.generate_single_idea()
                if idea:
                    count += 1
                    cprint(f"\n[Total: {count}] ✨ New Idea saved ({datetime.now().strftime('%H:%M:%S')}):", "green")
                    cprint(f"   \"{idea}\"", "white")
                time.sleep(interval_sec)
        except KeyboardInterrupt:
            cprint(f"\n🛑 Stopped continuous generation. {count} ideas generated this session.", "yellow")


def main():
    parser = argparse.ArgumentParser(description="Moon Dev's RBI v3 Quantitative Idea Generator")
    parser.add_argument("--count", type=int, default=5, help="Number of ideas to generate (default: 5)")
    parser.add_argument("--continuous", action="store_true", help="Run in continuous generation loop")
    parser.add_argument("--interval", type=int, default=15, help="Interval in seconds between continuous runs")
    parser.add_argument("--model-type", type=str, default=None, help="LLM provider: deepseek, openrouter, openai")
    parser.add_argument("--model-name", type=str, default=None, help="Model name slug (e.g. deepseek-chat, openai/gpt-4o)")
    args = parser.parse_args()

    generator = IdeaGeneratorV3(model_type=args.model_type, model_name=args.model_name)

    if args.continuous:
        generator.run_continuous(interval_sec=args.interval)
    else:
        generator.generate_batch(count=args.count)


if __name__ == "__main__":
    main()
