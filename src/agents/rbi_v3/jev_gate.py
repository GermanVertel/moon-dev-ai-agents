"""
🌙 Moon Dev's RBI v3 Jev Gatekeeper
Integrates TypeSafe Jev (typesafe/jev-1.13 via OpenRouter) for ultra-fast, cheap deterministic decisions.
Filters ideas in Phase 0 (<150ms) and classifies strategy types for Leaderboard tagging.
"""

import os
import re
import requests
from typing import Tuple
from termcolor import cprint

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
JEV_MODEL = "typesafe/jev-1.13"


class JevGatekeeper:
    """Uses TypeSafe Jev System-1 decision engine via OpenRouter."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            cprint("ℹ️ OPENROUTER_API_KEY not found; Jev will use local heuristic fallback.", "yellow")

    def _call_jev(self, system_instruction: str, user_prompt: str) -> str:
        """Call Jev via OpenRouter chat completions endpoint."""
        if not self.api_key:
            return ""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://moondev.com",
            "X-Title": "MoonDev RBI v3",
            "Content-Type": "application/json"
        }

        payload = {
            "model": JEV_MODEL,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.0
        }

        try:
            res = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=8)
            if res.status_code == 200:
                data = res.json()
                return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            pass
        return ""

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

        # If Jev API is available, ask Jev for fast deterministic classification
        if self.api_key:
            system = "Classify if the input text describes an actionable trading strategy idea with indicators, entry/exit logic, or technical setup. Respond with only YES or NO."
            result = self._call_jev(system, f"Idea: {idea_clean}")
            if "YES" in result.upper():
                return True, "Jev Approved (Actionable Strategy)"
            elif "NO" in result.upper():
                return False, "Jev Rejected (Non-actionable / Junk)"

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
            system = "Classify the strategy into exactly one of: [Trend-Following, Mean-Reversion, Volatility-Breakout, Liquidity-Scalp, Momentum]. Output only the category name."
            result = self._call_jev(system, strategy_text[:800])
            for cat in ["Trend-Following", "Mean-Reversion", "Volatility-Breakout", "Liquidity-Scalp", "Momentum"]:
                if cat.lower() in result.lower():
                    return cat

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
