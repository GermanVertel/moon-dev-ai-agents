"""
🌙 Moon Dev's RBI v3 Code Checks & Static Linter
Performs AST-based syntax, safety, and lookahead-bias checks on generated strategy code.
"""

import ast
import re
from typing import NamedTuple, List, Optional


class ValidationResult(NamedTuple):
    is_valid: bool
    error_message: Optional[str]
    warnings: List[str]
    class_name: Optional[str]


FORBIDDEN_MODULES = {"os", "sys", "subprocess", "socket", "requests", "shutil", "urllib"}
FORBIDDEN_CALLS = {"eval", "exec", "open", "__import__"}


class StrategyVisitor(ast.NodeVisitor):
    def __init__(self):
        self.strategy_classes = []
        self.forbidden_imports = []
        self.forbidden_calls = []
        self.warnings = []

    def visit_ClassDef(self, node):
        # Check if class inherits from Strategy or BaseStrategy
        for base in node.bases:
            base_id = getattr(base, "id", getattr(base, "attr", None))
            if base_id in ["Strategy", "BaseStrategy"]:
                self.strategy_classes.append(node.name)
        self.generic_visit(node)

    def visit_Import(self, node):
        for alias in node.names:
            if alias.name in FORBIDDEN_MODULES:
                self.forbidden_imports.append(alias.name)
            if alias.name.startswith("backtesting.lib"):
                self.warnings.append(f"Forbidden backtesting.lib import: {alias.name}. Use TA-Lib or NumPy instead.")
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module in FORBIDDEN_MODULES:
            self.forbidden_imports.append(node.module)
        if node.module and node.module.startswith("backtesting.lib"):
            self.warnings.append(f"Forbidden import from backtesting.lib: {node.module}")
        self.generic_visit(node)

    def visit_Call(self, node):
        func_name = getattr(node.func, "id", None)
        if func_name in FORBIDDEN_CALLS:
            self.forbidden_calls.append(func_name)
        self.generic_visit(node)


def validate_strategy_code(code_str: str) -> ValidationResult:
    """Validate python code using AST parsing and static rule enforcement."""
    warnings = []
    
    # 1. Check syntax with AST
    try:
        tree = ast.parse(code_str)
    except SyntaxError as e:
        return ValidationResult(
            is_valid=False,
            error_message=f"SyntaxError on line {e.lineno}: {e.msg}",
            warnings=[],
            class_name=None
        )

    # 2. Check visitor
    visitor = StrategyVisitor()
    visitor.visit(tree)

    if visitor.forbidden_imports:
        return ValidationResult(
            is_valid=False,
            error_message=f"Forbidden imports detected for security: {', '.join(visitor.forbidden_imports)}",
            warnings=visitor.warnings,
            class_name=None
        )

    if visitor.forbidden_calls:
        return ValidationResult(
            is_valid=False,
            error_message=f"Forbidden built-in function calls: {', '.join(visitor.forbidden_calls)}",
            warnings=visitor.warnings,
            class_name=None
        )

    if not visitor.strategy_classes:
        return ValidationResult(
            is_valid=False,
            error_message="No class inheriting from 'Strategy' found in the generated code.",
            warnings=visitor.warnings,
            class_name=None
        )

    strategy_class = visitor.strategy_classes[0]
    warnings.extend(visitor.warnings)

    # 3. Lookahead Bias & Anti-pattern Regex Checks
    if re.search(r"\.shift\s*\(\s*-\d+\s*\)", code_str):
        warnings.append("Potential Lookahead Bias: Found negative shift (.shift(-n)) which leaks future data.")

    if re.search(r"center\s*=\s*True", code_str):
        warnings.append("Potential Lookahead Bias: Found rolling window with center=True.")

    if "backtesting.lib.crossover" in code_str:
        warnings.append("Anti-pattern: Found backtesting.lib.crossover. Use array comparisons (e.g. k[-2]<d[-2] and k[-1]>d[-1]).")

    return ValidationResult(
        is_valid=True,
        error_message=None,
        warnings=warnings,
        class_name=strategy_class
    )
