"""
🌙 Moon Dev's RBI v3 Leaderboard & Winner Storage
Tracks all evaluated strategies, records quantitative scorecards, and archives verified winners.
"""

import os
import json
import shutil
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
from termcolor import cprint

ROOT_DIR = Path(__file__).parent.parent.parent.parent
DEFAULT_RBI_DIR = ROOT_DIR / "src" / "data" / "rbi_v3"


class LeaderboardManager:
    """Manages strategy scorecards and archives certified winning strategies."""

    def __init__(self, rbi_dir: Path = DEFAULT_RBI_DIR):
        self.rbi_dir = Path(rbi_dir)
        self.winners_dir = self.rbi_dir / "winners"
        self.rbi_dir.mkdir(parents=True, exist_ok=True)
        self.winners_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.rbi_dir / "leaderboard.csv"
        self.md_path = self.rbi_dir / "leaderboard.md"
        self._init_csv()

    def _init_csv(self):
        if not self.csv_path.exists():
            df = pd.DataFrame(columns=[
                "rank", "strategy_name", "unique_id", "style_tag", "status",
                "total_score", "is_return_pct", "is_sharpe", "is_max_dd_pct",
                "is_trades", "is_win_rate", "is_pf", "multi_pass_count",
                "oos_return_pct", "oos_sharpe", "oos_trades", "rejection_reason", "timestamp"
            ])
            df.to_csv(self.csv_path, index=False)

    def record_entry(self, entry: Dict[str, Any], code_str: str = None) -> None:
        """Record an evaluation entry and update leaderboard files."""
        df = pd.read_csv(self.csv_path) if self.csv_path.exists() else pd.DataFrame()
        
        # Check if unique_id already exists; update or append
        uid = entry.get("unique_id")
        if uid and not df.empty and uid in df["unique_id"].values:
            df.loc[df["unique_id"] == uid, list(entry.keys())] = list(entry.values())
        else:
            df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)

        # Sort by total_score descending
        if "total_score" in df.columns:
            df["total_score"] = pd.to_numeric(df["total_score"], errors="coerce").fillna(0.0)
            df.sort_values(by="total_score", ascending=False, inplace=True)
            df.reset_index(drop=True, inplace=True)
            df["rank"] = df.index + 1

        df.to_csv(self.csv_path, index=False)

        # If certified winner, archive code and metadata
        if entry.get("status") == "CERTIFIED_WINNER" and code_str:
            s_name = entry.get("strategy_name", "Strategy")
            winner_file = self.winners_dir / f"{s_name}_{uid}_WINNER.py"
            with open(winner_file, "w") as f:
                f.write(code_str)

            meta_file = self.winners_dir / f"{s_name}_{uid}_metadata.json"
            with open(meta_file, "w") as f:
                json.dump(entry, f, indent=2)

            cprint(f"🏆 CERTIFIED WINNER ARCHIVED: {winner_file.name} (Score: {entry.get('total_score')})", "green", attrs=["bold"])

        self._export_markdown(df)

    def _export_markdown(self, df: pd.DataFrame):
        """Export clean GitHub-flavored markdown leaderboard."""
        if df.empty:
            return

        cols_display = [
            "rank", "strategy_name", "style_tag", "status", "total_score",
            "is_return_pct", "is_sharpe", "is_max_dd_pct", "is_trades",
            "multi_pass_count", "oos_return_pct", "oos_sharpe"
        ]
        available_cols = [c for c in cols_display if c in df.columns]
        top_df = df[available_cols].head(30)

        lines = [
            "# 🌙 Moon Dev RBI v3 Leaderboard\n",
            f"**Last Updated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} UTC\n",
            f"**Total Strategies Evaluated:** {len(df)} | **Certified Winners:** {len(df[df.get('status') == 'CERTIFIED_WINNER'])}\n\n",
            "| " + " | ".join(available_cols) + " |",
            "| " + " | ".join(["---"] * len(available_cols)) + " |"
        ]

        for _, row in top_df.iterrows():
            row_str = " | ".join(str(row[c]) if not pd.isna(row[c]) else "-" for c in available_cols)
            lines.append(f"| {row_str} |")

        with open(self.md_path, "w") as f:
            f.write("\n".join(lines) + "\n")
