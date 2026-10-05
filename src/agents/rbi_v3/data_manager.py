"""
🌙 Moon Dev's RBI v3 Data Manager
Downloads, validates, and partitions multi-year historical market data (2022-present).
Uses Binance Public API (no key required) with in-sample / out-of-sample partitioning.
"""

import os
import sys
import time
import requests
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from termcolor import cprint
from tenacity import retry, stop_after_attempt, wait_exponential

# Default directory structure
ROOT_DIR = Path(__file__).parent.parent.parent.parent
DEFAULT_DATA_DIR = ROOT_DIR / "src" / "data" / "rbi_v3" / "market_data"

# Standard symbols and timeframes
DEFAULT_SYMBOLS = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT"
}

DEFAULT_TIMEFRAMES = ["15m", "1h"]

# In-sample vs Out-of-sample date split
SPLIT_DATE = "2024-07-01 00:00:00"
START_DATE = "2022-01-01 00:00:00"

BINANCE_API_URL = "https://api.binance.com/api/v3/klines"
BINANCE_VISION_URL = "https://data-api.binance.vision/api/v3/klines"


class DataManager:
    """Manages downloading, caching, and serving partitioned market data."""

    def __init__(self, data_dir: Path = DEFAULT_DATA_DIR):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _fetch_binance_klines(self, symbol: str, interval: str, start_time_ms: int, end_time_ms: int, limit: int = 1000):
        """Fetch a single batch of klines from Binance public API with exponential backoff retry."""
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_time_ms,
            "endTime": end_time_ms,
            "limit": limit
        }
        headers = {"User-Agent": "Mozilla/5.0 (MoonDev-RBI-v3)"}

        for url in [BINANCE_API_URL, BINANCE_VISION_URL]:
            try:
                response = requests.get(url, params=params, headers=headers, timeout=15)
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:
                    cprint("⚠️ Rate limited by Binance, retrying with exponential backoff...", "yellow")
                    raise Exception("Rate limit 429")
            except requests.exceptions.RequestException:
                pass
        raise Exception(f"Failed to fetch {symbol} from Binance after retries")

    def download_symbol_history(self, symbol: str, interval: str, start_dt: str = START_DATE) -> Path:
        """Download complete history from start_dt to present and save to disk."""
        binance_pair = DEFAULT_SYMBOLS.get(symbol.upper(), f"{symbol.upper()}USDT")
        cprint(f"📥 Moon Dev Data Manager: Downloading {symbol} ({binance_pair}) {interval} from {start_dt}...", "cyan")

        start_dt_obj = datetime.strptime(start_dt, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        current_time_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        start_time_ms = int(start_dt_obj.timestamp() * 1000)

        all_candles = []
        last_time = start_time_ms

        while last_time < current_time_ms:
            batch = self._fetch_binance_klines(binance_pair, interval, last_time, current_time_ms, limit=1000)
            if not batch:
                break

            for row in batch:
                # Binance kline format: [open_time, open, high, low, close, volume, close_time, ...]
                open_ts = int(row[0])
                all_candles.append({
                    "Datetime": datetime.fromtimestamp(open_ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                    "Open": float(row[1]),
                    "High": float(row[2]),
                    "Low": float(row[3]),
                    "Close": float(row[4]),
                    "Volume": float(row[5])
                })

            # Next startTime is 1ms after last candle close
            last_candle_close = int(batch[-1][6])
            if last_candle_close <= last_time:
                break
            last_time = last_candle_close + 1
            time.sleep(0.08)  # Polite rate limiting

        if not all_candles:
            raise ValueError(f"No candles retrieved for {symbol} ({binance_pair}) {interval}")

        df = pd.DataFrame(all_candles)
        df.drop_duplicates(subset=["Datetime"], inplace=True)
        df.sort_values(by="Datetime", inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Save full file
        full_path = self.data_dir / f"{symbol.upper()}-USD-{interval}_FULL.csv"
        df.to_csv(full_path, index=False)

        # Partition In-Sample (IS) and Out-Of-Sample (OOS)
        df_is = df[df["Datetime"] < SPLIT_DATE].copy()
        df_oos = df[df["Datetime"] >= SPLIT_DATE].copy()

        is_path = self.data_dir / f"{symbol.upper()}-USD-{interval}_IS.csv"
        oos_path = self.data_dir / f"{symbol.upper()}-USD-{interval}_OOS.csv"

        df_is.to_csv(is_path, index=False)
        df_oos.to_csv(oos_path, index=False)

        cprint(f"✅ {symbol} {interval} saved:", "green")
        cprint(f"   • Full: {len(df):,} bars -> {full_path.name}", "cyan")
        cprint(f"   • In-Sample (2022 to {SPLIT_DATE[:10]}): {len(df_is):,} bars", "cyan")
        cprint(f"   • Out-Of-Sample ({SPLIT_DATE[:10]} to present): {len(df_oos):,} bars", "cyan")

        return is_path

    def ensure_datasets(self, symbols=None, timeframes=None):
        """Ensure all required datasets exist; download if missing."""
        if symbols is None:
            symbols = list(DEFAULT_SYMBOLS.keys())
        if timeframes is None:
            timeframes = DEFAULT_TIMEFRAMES

        for sym in symbols:
            for tf in timeframes:
                is_file = self.data_dir / f"{sym.upper()}-USD-{tf}_IS.csv"
                if not is_file.exists() or is_file.stat().st_size < 1000:
                    self.download_symbol_history(sym, tf)
                else:
                    cprint(f"✨ Cached dataset found: {is_file.name}", "green")

    def load_dataset(self, symbol: str, timeframe: str = "15m", split: str = "IS") -> pd.DataFrame:
        """Load and format dataframe ready for backtesting.py."""
        symbol = symbol.upper()
        split = split.upper()
        suffix = "IS" if split in ["IS", "IN_SAMPLE"] else ("OOS" if split in ["OOS", "OUT_OF_SAMPLE"] else "FULL")
        
        file_path = self.data_dir / f"{symbol}-USD-{timeframe}_{suffix}.csv"
        if not file_path.exists():
            self.download_symbol_history(symbol, timeframe)

        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip().str.capitalize()
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        df.set_index("Datetime", inplace=True)
        return df[["Open", "High", "Low", "Close", "Volume"]]


if __name__ == "__main__":
    cprint("🌙 Moon Dev Data Manager CLI Initializing...", "white", "on_blue")
    dm = DataManager()
    dm.ensure_datasets()
    cprint("🚀 All datasets ready for RBI v3!", "green")
