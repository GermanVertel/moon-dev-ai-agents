"""
🌙 Moon Dev's RBI v3 Backtest Harness
Executes generated Strategy classes inside a controlled execution sandbox.
Standardizes data loading, initial cash ($100k), and Hyperliquid commission (0.045%).
Outputs structured JSON metrics without relying on regex stdout scraping.
"""

import os
import sys
import json
import time
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional

RUNNER_TEMPLATE = """
import sys
import json
import pandas as pd
import numpy as np
import talib
import pandas_ta as ta
from backtesting import Backtest, Strategy

# Load dataset
df = pd.read_csv("{data_path}")
df.columns = df.columns.str.strip().str.capitalize()
df["Datetime"] = pd.to_datetime(df["Datetime"])
df.set_index("Datetime", inplace=True)
data = df[["Open", "High", "Low", "Close", "Volume"]]

# Strategy code injected below
{strategy_code}

# Instantiate and run backtest
try:
    bt = Backtest(
        data, 
        {class_name}, 
        cash={cash}, 
        commission={commission}, 
        exclusive_orders=True, 
        finalize_trades=True
    )
    stats = bt.run()
    
    def safe_float(val, default=0.0):
        try:
            if pd.isna(val) or val is None or str(val).strip() in ["--", "nan", "NaN"]:
                return default
            return float(val)
        except Exception:
            return default

    def safe_int(val, default=0):
        try:
            if pd.isna(val) or val is None or str(val).strip() in ["--", "nan", "NaN"]:
                return default
            return int(val)
        except Exception:
            return default

    results = {{
        "success": True,
        "trades": safe_int(stats.get("# Trades", 0)),
        "return_pct": safe_float(stats.get("Return [%]", 0.0)),
        "buy_hold_pct": safe_float(stats.get("Buy & Hold Return [%]", 0.0)),
        "sharpe": safe_float(stats.get("Sharpe Ratio", 0.0)),
        "sortino": safe_float(stats.get("Sortino Ratio", 0.0)),
        "max_dd_pct": safe_float(stats.get("Max. Drawdown [%]", 0.0)),
        "win_rate": safe_float(stats.get("Win Rate [%]", 0.0)),
        "profit_factor": safe_float(stats.get("Profit Factor", 0.0)),
        "exposure_pct": safe_float(stats.get("Exposure Time [%]", 0.0)),
        "sqn": safe_float(stats.get("SQN", 0.0)),
        "start": str(stats.get("Start", "")),
        "end": str(stats.get("End", ""))
    }}
    
    # Prefix with delimiter for clean extraction
    print("###JSON_OUTPUT_START###")
    print(json.dumps(results))
    print("###JSON_OUTPUT_END###")
    sys.exit(0)

except Exception as e:
    import traceback
    err_info = {{
        "success": False,
        "error": str(e),
        "traceback": traceback.format_exc()
    }}
    print("###JSON_OUTPUT_START###")
    print(json.dumps(err_info))
    print("###JSON_OUTPUT_END###")
    sys.exit(1)
"""


def run_backtest_harness(
    strategy_code: str,
    class_name: str,
    data_path: str,
    cash: float = 100_000.0,
    commission: float = 0.00045, # 0.045% Hyperliquid taker fee
    timeout_sec: int = 180,
    conda_env: Optional[str] = "tflow"
) -> Dict[str, Any]:
    """Execute strategy inside harness and return structured metrics dictionary."""
    data_path_abs = str(Path(data_path).resolve())
    if not os.path.exists(data_path_abs):
        return {
            "success": False,
            "error": f"Dataset file not found: {data_path_abs}",
            "stats": {}
        }

    # Generate complete self-contained script
    full_script = RUNNER_TEMPLATE.format(
        data_path=data_path_abs,
        strategy_code=strategy_code,
        class_name=class_name,
        cash=cash,
        commission=commission
    )

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tmp:
        tmp.write(full_script)
        tmp_path = tmp.name

    start_time = time.time()
    try:
        if conda_env:
            cmd = ["conda", "run", "-n", conda_env, "python", tmp_path]
        else:
            cmd = [sys.executable, tmp_path]

        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_sec
        )
        
        exec_time = time.time() - start_time
        stdout = process.stdout
        stderr = process.stderr

        # Extract delimited JSON output
        if "###JSON_OUTPUT_START###" in stdout and "###JSON_OUTPUT_END###" in stdout:
            json_str = stdout.split("###JSON_OUTPUT_START###")[1].split("###JSON_OUTPUT_END###")[0].strip()
            try:
                parsed = json.loads(json_str)
                parsed["execution_time"] = exec_time
                parsed["stderr"] = stderr
                return parsed
            except Exception as json_err:
                return {
                    "success": False,
                    "error": f"JSON parse error: {str(json_err)}",
                    "stdout": stdout,
                    "stderr": stderr,
                    "execution_time": exec_time
                }

        # Fallback error detection
        return {
            "success": False,
            "error": stderr.strip() if stderr else "No JSON output produced by backtest harness",
            "stdout": stdout,
            "stderr": stderr,
            "execution_time": exec_time
        }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": f"Execution timeout ({timeout_sec}s expired)",
            "execution_time": timeout_sec
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Harness execution error: {str(e)}",
            "execution_time": time.time() - start_time
        }
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
