"""
🌙 Moon Dev's RBI v3 Jev Gatekeeper
Integrates TypeSafe Jev (typesafe/jev-1.13 via OpenRouter Decisions API) for ultra-fast, cheap deterministic decisions.
Filters ideas in Phase 0 (<150ms) and classifies strategy types for Leaderboard tagging.
"""

import os
import re
import requests
from typing import Tuple
from termcolor import cprint

OPENROUTER_DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
JEV_MODEL = "typesafe/jev-1.13"

UNAVAILABLE_DATA_PATTERN = (
    r"\bvix\b|\bvxn\b|\bvvix\b|sentiment|fear\s*(and|&)\s*greed|funding\s*rate|open\s*interest|order\s*book|"
    r"\bdepth\b|on-?chain|whale|liquidation|put/?call|options?\s+(flow|chain|iv)|implied\s+vol|news|earnings|"
    r"\bcot\b|dxy|treasury|yield\s*curve|google\s*trends|social\s*media|twitter"
)


class JevGatekeeper:
    """Uses TypeSafe Jev System-1 decision engine via OpenRouter Decisions API."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_KEY")
        if not self.api_key:
            cprint("ℹ️ OPENROUTER_API_KEY not found; Jev will use local heuristic fallback.", "yellow")

    def _call_jev_decisions(self, state: str, questions: dict) -> dict:
        """Call OpenRouter alpha decisions endpoint."""
        if not self.api_key:
            return {}

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://moondev.com",
            "X-Title": "MoonDev RBI v3",
            "Content-Type": "application/json"
        }

        payload = {
            "model": JEV_MODEL,
            "state": state,
            "questions": questions
        }

        try:
            res = requests.post(OPENROUTER_DECISIONS_URL, headers=headers, json=payload, timeout=8)
            if res.status_code == 200:
                data = res.json()
                return data.get("answers", {})
            else:
                cprint(f"⚠️ Jev API returned {res.status_code}: {res.text[:120]}", "yellow")
        except Exception as e:
            cprint(f"⚠️ Jev connection error: {str(e)[:80]}", "yellow")
        return {}

    def is_actionable_idea(self, idea_text: str) -> Tuple[bool, str]:
        """Classify whether a raw idea line contains an actionable trading strategy thesis."""
        idea_clean = idea_text.strip()

        # Fast local heuristic check first
        if len(idea_clean) < 15:
            return False, "Idea too short (<15 chars)"
        if idea_clean.endswith(":") or idea_clean.startswith("#"):
            return False, "Section header or comment"
        if re.search(r"^(generate|create|write)\s+one\s+unique", idea_clean, re.IGNORECASE):
            return False, "Generic prompt artifact"

        if re.search(UNAVAILABLE_DATA_PATTERN, idea_clean, re.IGNORECASE):
            return False, "Needs data not in OHLCV feed (VIX/sentiment/funding/order book/on-chain)"

        # If Jev API is available, ask Jev for fast deterministic classification via Decisions API
        if self.api_key:
            questions = {
                "is_actionable": {
                    "type": "noul",
                    "instructions": "Does this text describe an actionable quantitative trading strategy idea with indicators, entry/exit rules or technical setup?"
                }
            }
            answers = self._call_jev_decisions(idea_clean, questions)
            if "is_actionable" in answers:
                score = answers["is_actionable"].get("noul", 0.0)
                if score >= 0.55:
                    return True, f"Jev Approved (Score: {score:.2f})"
                else:
                    return False, f"Jev Rejected (Score: {score:.2f})"

        # Fallback keyword validation
        keywords = ["rsi", "ma", "ema", "sma", "macd", "stoch", "breakout", "reversion", 
                    "trend", "volume", "volatility", "fractal", "fibonacci", "band", "liquidity",
                    "momentum", "cross", "divergence", "atr", "channel", "squeeze", "price"]
        found = any(k in idea_clean.lower() for k in keywords)
        if found:
            return True, "Heuristic Approved"
        return False, "No trading indicators or setup keywords found"

    def classify_strategy_type(self, strategy_text: str) -> str:
        """Classify strategy into a quantitative style tag."""
        if self.api_key:
            questions = {
                "strategy_type": {
                    "type": "choice",
                    "instructions": "Which quantitative category best fits this strategy?",
                    "criteria": {
                        "Trend-Following": "Follows prevailing trend or moving average direction",
                        "Mean-Reversion": "Trades counter-trend or bounces from oversold/overbought",
                        "Volatility-Breakout": "Trades channel or band expansion",
                        "Momentum": "Trades directional acceleration"
                    }
                }
            }
            answers = self._call_jev_decisions(strategy_text[:800], questions)
            if "strategy_type" in answers:
                choice = answers["strategy_type"].get("choice")
                if choice:
                    return choice

        # Local fallback heuristic
        text_lower = strategy_text.lower()
        if "reversion" in text_lower or "oversold" in text_lower or "overbought" in text_lower:
            return "Mean-Reversion"
        elif "breakout" in text_lower or "squeeze" in text_lower:
            return "Volatility-Breakout"
        elif "trend" in text_lower or "cross" in text_lower or "ema" in text_lower:
            return "Trend-Following"
        elif "momentum" in text_lower or "rsi" in text_lower:
            return "Momentum"
        return "Quantitative-Hybrid"
