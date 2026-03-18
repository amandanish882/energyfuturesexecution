"""One-time download of CME energy futures settlement strips from Databento.

Downloads ohlcv-1d data for CL, HO, RB, NG contract months and saves
them as strip parquet files in the data/ cache directory, matching the
format expected by CommodityDataLoader.

Usage: python download_settlement_data.py
"""

import sys
from pathlib import Path
from datetime import date, timedelta

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import databento as db

from config import get_databento_api_key, DATA_DIR, DEFAULT_VALUATION_DATE

MONTH_CODES = {
    1: "F", 2: "G", 3: "H", 4: "J", 5: "K", 6: "M",
    7: "N", 8: "Q", 9: "U", 10: "V", 11: "X", 12: "Z",
}

PRODUCTS = ["CL", "HO", "RB", "NG"]
N_MONTHS = 12  # number of forward contract months to fetch


def build_strip_symbols(product, ref_date):
    """Build a list of CME Globex contract symbols for the next N months.

    For example, if ref_date is 2026-03-10 and product is "CL":
    returns ["CLJ26", "CLK26", "CLM26", ...] (starting from next month).
    """
    symbols = []
    d = ref_date.replace(day=1)
    # Start from next month (current month contract may have expired)
    month = d.month + 1
    year = d.year
    if month > 12:
        month = 1
        year += 1

    for _ in range(N_MONTHS):
        code = MONTH_CODES[month]
        yr_suffix = year % 100
        symbols.append(f"{product}{code}{yr_suffix}")
        month += 1
        if month > 12:
            month = 1
            year += 1

    return symbols


def download_strips(api_key, valuation_date_str):
    """Download settlement strips for all products and save to cache."""
    client = db.Historical(api_key)
    val_date = date.fromisoformat(valuation_date_str)

    # We need ohlcv-1d for the valuation date — fetch a small window
    # to ensure we get data (markets may be closed on exact date)
    start = (val_date - timedelta(days=5)).isoformat()
    end = (val_date + timedelta(days=1)).isoformat()

    for product in PRODUCTS:
        symbols = build_strip_symbols(product, val_date)
        print(f"\n--- {product} ---")
        print(f"  Symbols: {symbols}")

        # Check cost first
        try:
            cost = client.metadata.get_cost(
                dataset="GLBX.MDP3",
                symbols=symbols,
                schema="ohlcv-1d",
                stype_in="raw_symbol",
                start=start,
                end=end,
            )
            print(f"  Estimated cost: ${cost:.4f}")
        except Exception as e:
            print(f"  Cost check failed: {e}")

        # Download
        try:
            data = client.timeseries.get_range(
                dataset="GLBX.MDP3",
                symbols=symbols,
                schema="ohlcv-1d",
                stype_in="raw_symbol",
                start=start,
                end=end,
            )
            df = data.to_df()

            if len(df) == 0:
                print(f"  No data returned for {product}")
                continue

            print(f"  Downloaded {len(df)} rows")

            # Get the most recent date's data (closest to valuation date)
            df = df.reset_index()
            if "ts_event" in df.columns:
                df["trade_date"] = pd.to_datetime(df["ts_event"]).dt.date
            elif "date" in df.columns:
                df["trade_date"] = pd.to_datetime(df["date"]).dt.date

            latest_date = df["trade_date"].max()
            latest = df[df["trade_date"] == latest_date]
            print(f"  Using data from: {latest_date}")

            # Build the strip: contract symbol -> close price
            strip = {}
            for _, row in latest.iterrows():
                sym = row.get("symbol", row.get("raw_symbol", ""))
                # Databento may return full instrument ID; extract the root symbol
                # The close price is the settlement price
                close = row["close"]
                if close > 0:
                    strip[sym] = close

            if not strip:
                print(f"  No valid prices for {product}")
                continue

            # Save as single-row DataFrame matching existing cache format
            strip_df = pd.DataFrame([strip])
            cache_path = DATA_DIR / f"strip_{product}_{valuation_date_str}.parquet"
            strip_df.to_parquet(cache_path)
            print(f"  Saved: {cache_path.name}")
            print(f"  Contracts: {len(strip)}")
            for sym, price in sorted(strip.items()):
                print(f"    {sym}: {price:.4f}")

        except Exception as e:
            print(f"  Download failed for {product}: {e}")


def main():
    api_key = get_databento_api_key()
    if not api_key:
        print("ERROR: No DATABENTO_API_KEY found.")
        print("Set it in .env file or DATABENTO_API_KEY environment variable.")
        sys.exit(1)

    valuation_date = DEFAULT_VALUATION_DATE
    print(f"Downloading CME settlement strips from Databento")
    print(f"Valuation date: {valuation_date}")
    print(f"Products: {PRODUCTS}")
    print(f"Cache dir: {DATA_DIR}")

    download_strips(api_key, valuation_date)

    print("\n" + "=" * 50)
    print("Download complete.")
    print(f"Strip files saved to: {DATA_DIR}")
    print("These will be used by run_full_demo.py")


if __name__ == "__main__":
    main()
