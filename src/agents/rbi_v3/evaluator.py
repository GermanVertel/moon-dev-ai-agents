"""
🌙 Moon Dev's RBI v3 Quantitative Evaluator
Enforces rigorous statistical gates (In-Sample, Multi-Asset, and Out-of-Sample validation).
"""

from typing import Dict, Any, Tuple, List

DEFAULT_GATES = {
    "min_trades": 100,          # Statistical significance / day-to-day operability
    "max_trades": 800,          # Protection against overtrading fees
    "min_sharpe": 1.0,          # Risk-adjusted return quality
    "min_profit_factor": 1.3,   # Gross profit / Gross loss ratio
    "max_drawdown_pct": -25.0,  # Maximum capital drawdown allowed
    "min_return_pct": 0.0,      # Must be strictly positive
    "min_multi_asset_pass": 2,  # Must be profitable in >= 2 assets out of 3
    "min_oos_sharpe": 0.5,      # Must retain positive Sharpe in Out-Of-Sample test
    "min_oos_return": 0.0       # Must maintain positive return in Out-Of-Sample test
}


class StrategyEvaluator:
    """Evaluates strategy metrics against multi-phase quantitative filters."""

    def __init__(self, gates: Dict[str, Any] = None):
        self.gates = DEFAULT_GATES.copy()
        if gates:
            self.gates.update(gates)

    def evaluate_in_sample(self, stats: Dict[str, Any]) -> Tuple[bool, str, float]:
        """Evaluate Phase 1: In-Sample performance gate."""
        if not stats or not stats.get("success", False):
            return False, "Execution failed or produced no metrics", 0.0

        trades = stats.get("trades", 0)
        ret_pct = stats.get("return_pct", 0.0)
        sharpe = stats.get("sharpe", 0.0)
        pf = stats.get("profit_factor", 0.0)
        dd = stats.get("max_dd_pct", 0.0)

        # 1. Trade count bounds
        if trades < self.gates["min_trades"]:
            return False, f"Insufficient trades ({trades} < {self.gates['min_trades']})", 0.0
        if trades > self.gates["max_trades"]:
            return False, f"Overtrading danger ({trades} > {self.gates['max_trades']} trades)", 0.0

        # 2. Return & Sharpe
        if ret_pct <= self.gates["min_return_pct"]:
            return False, f"Negative return ({ret_pct:.2f}% <= {self.gates['min_return_pct']}%)", 0.0
        if sharpe < self.gates["min_sharpe"]:
            return False, f"Low Sharpe ratio ({sharpe:.2f} < {self.gates['min_sharpe']})", 0.0

        # 3. Profit Factor & Drawdown
        if pf < self.gates["min_profit_factor"]:
            return False, f"Low Profit Factor ({pf:.2f} < {self.gates['min_profit_factor']})", 0.0
        if dd < self.gates["max_drawdown_pct"]:
            return False, f"Excessive drawdown ({dd:.2f}% < {self.gates['max_drawdown_pct']}%)", 0.0

        # Composite In-Sample Score
        score = (sharpe * 1.5) + (pf * 1.0) - (abs(dd) * 0.05) + (min(ret_pct, 100.0) * 0.02)
        return True, "Passed In-Sample Gate", round(score, 3)

    def evaluate_multi_asset(self, asset_results: Dict[str, Dict[str, Any]]) -> Tuple[bool, int, str]:
        """Evaluate Phase 2: Multi-Asset robustness gate (e.g. BTC, ETH, SOL)."""
        pass_count = 0
        details = []

        for symbol, stats in asset_results.items():
            if stats.get("success", False) and stats.get("return_pct", -1) > 0 and stats.get("trades", 0) >= 15:
                pass_count += 1
                details.append(f"{symbol}: +{stats.get('return_pct'):.1f}% (Sharpe {stats.get('sharpe'):.2f})")
            else:
                details.append(f"{symbol}: FAIL ({stats.get('return_pct', 0):.1f}%)")

        required = self.gates["min_multi_asset_pass"]
        passed = pass_count >= required
        summary = f"Passed {pass_count}/{len(asset_results)} assets [{', '.join(details)}]"
        return passed, pass_count, summary

    def evaluate_out_of_sample(self, oos_stats: Dict[str, Any]) -> Tuple[bool, str]:
        """Evaluate Phase 3: Out-of-Sample blind test validation."""
        if not oos_stats or not oos_stats.get("success", False):
            return False, "OOS execution failed"

        trades = oos_stats.get("trades", 0)
        ret_pct = oos_stats.get("return_pct", 0.0)
        sharpe = oos_stats.get("sharpe", 0.0)

        if trades < 10:
            return False, f"OOS took too few trades ({trades})"
        if ret_pct <= self.gates["min_oos_return"]:
            return False, f"OOS negative return ({ret_pct:.2f}%)"
        if sharpe < self.gates["min_oos_sharpe"]:
            return False, f"OOS Sharpe too low ({sharpe:.2f} < {self.gates['min_oos_sharpe']})"

        return True, f"OOS Validated (+{ret_pct:.2f}%, Sharpe {sharpe:.2f}, {trades} trades)"

    def calculate_total_score(self, is_stats: dict, multi_pass_count: int, oos_stats: dict) -> float:
        """Calculate master composite ranking score for Leaderboard."""
        is_sharpe = is_stats.get("sharpe", 0.0)
        oos_sharpe = oos_stats.get("sharpe", 0.0)
        is_ret = min(is_stats.get("return_pct", 0.0), 200.0)
        oos_ret = min(oos_stats.get("return_pct", 0.0), 200.0)
        max_dd = abs(is_stats.get("max_dd_pct", 0.0))

        score = (
            (is_sharpe * 2.0) +
            (oos_sharpe * 3.0) +
            (multi_pass_count * 1.5) +
            (is_ret * 0.02) +
            (oos_ret * 0.04) -
            (max_dd * 0.08)
        )
        return round(max(score, 0.0), 2)
