#!/usr/bin/env python3
"""
🌙 Update BTC-USD historical data for backtesting (2024-2026)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from termcolor import cprint

def update_btc_data():
    """Fetch BTC-USD data 2024-2026 and save to CSV."""

    cprint("🌙 Moon Dev fetching BTC-USD data (2024-2026)...", "cyan")

    # Download 15m candles for BTC-USD
    end_date = datetime.now()
    start_date = datetime(2024, 1, 1)

    try:
        # yfinance for 1d interval (15m limited to 60 days)
        df = yf.download(
            "BTC-USD",
            start=start_date,
            end=end_date,
            interval="1d",
            progress=False
        )

        if df.empty:
            cprint("❌ No data fetched", "red")
            return False

        # Format to match existing CSV
        df.reset_index(inplace=True)
        df_export = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
        df_export.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        df_export['datetime'] = df_export['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')

        # Save to CSV (same location as original)
        output_path = "src/data/rbi/BTC-USD-15m.csv"

        # Write header + data
        with open(output_path, 'w') as f:
            f.write("datetime, open, high, low, close, volume,\n")
            for _, row in df_export.iterrows():
                f.write(f"{row['datetime']}, {row['open']}, {row['high']}, {row['low']}, {row['close']}, {row['volume']},\n")

        cprint(f"✅ Updated {output_path}", "green")
        cprint(f"📊 Total candles: {len(df_export)}", "yellow")
        cprint(f"📅 Date range: {df_export['datetime'].iloc[0]} to {df_export['datetime'].iloc[-1]}", "yellow")

        return True

    except Exception as e:
        cprint(f"❌ Error: {e}", "red")
        return False

if __name__ == "__main__":
    update_btc_data()
