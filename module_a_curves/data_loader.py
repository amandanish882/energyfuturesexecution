"""Data loader for commodity forward curve construction.

Provides utilities for fetching CME futures settlement strips and EIA
petroleum inventory data. Results are cached locally as Parquet files to
reduce repeated API calls. The primary entry point for curve building is
CommodityDataLoader.get_strip_for_date(), which returns a list of
settlement dicts ready for ForwardCurveBootstrapper.
"""

import datetime
import os
from pathlib import Path

import numpy as np
import pandas as pd

# Contract month codes
MONTH_CODE = {
    "F": 1, "G": 2, "H": 3, "J": 4, "K": 5, "M": 6,
    "N": 7, "Q": 8, "U": 9, "V": 10, "X": 11, "Z": 12,
}

# EIA API series IDs
EIA_SERIES = {
    "crude_stocks": "PET.WCESTUS1.W",
    "gasoline_stocks": "PET.WGTSTUS1.W",
    "distillate_stocks": "PET.WDISTUS1.W",
    "refinery_util": "PET.WPULEUS3.W",
}

# 5-year seasonal averages (thousands of barrels)
SEASONAL_INVENTORY_AVG = {
    1: {"crude": 432000, "gasoline": 240000, "distillate": 128000},
    2: {"crude": 440000, "gasoline": 248000, "distillate": 122000},
    3: {"crude": 450000, "gasoline": 238000, "distillate": 116000},
    4: {"crude": 448000, "gasoline": 228000, "distillate": 113000},
    5: {"crude": 440000, "gasoline": 225000, "distillate": 112000},
    6: {"crude": 430000, "gasoline": 230000, "distillate": 118000},
    7: {"crude": 425000, "gasoline": 235000, "distillate": 125000},
    8: {"crude": 420000, "gasoline": 230000, "distillate": 130000},
    9: {"crude": 418000, "gasoline": 225000, "distillate": 135000},
    10: {"crude": 420000, "gasoline": 218000, "distillate": 140000},
    11: {"crude": 425000, "gasoline": 215000, "distillate": 138000},
    12: {"crude": 430000, "gasoline": 220000, "distillate": 132000},
}


def _contract_to_expiry(contract_code, valuation_date):
    """Convert a CME-style contract code to a time-to-expiry in years.

    Parses the month letter and two-digit year suffix from a contract code
    such as "CLZ26", derives the approximate expiry date (the 20th of the
    month preceding the delivery month), and computes the year-fraction
    distance from the valuation date. The minimum returned value is one
    calendar day to avoid zero or negative tenors.

    Args:
        contract_code: Exchange contract code string in the form
            "<product><month_letter><year_suffix>", e.g. "CLZ26" for
            WTI December 2026.
        valuation_date: ISO-format date string (e.g. "2024-12-31") used
            as the reference point for the time calculation.

    Returns:
        Time to expiry expressed as a year fraction (float), with a
        minimum value of 1/365.25.
    """
    month_code = contract_code[2]
    year_suffix = int(contract_code[3:])
    year = 2000 + year_suffix if year_suffix < 100 else year_suffix
    month = MONTH_CODE.get(month_code, 1)

    # Approximate expiry: 20th of month before delivery
    if month == 1:
        expiry_month, expiry_year = 12, year - 1
    else:
        expiry_month, expiry_year = month - 1, year

    expiry_date = datetime.date(expiry_year, expiry_month, 20)
    val_date = datetime.date.fromisoformat(valuation_date)
    days = (expiry_date - val_date).days
    return max(days / 365.25, 1 / 365.25)


class CommodityDataLoader:
    """Fetches and caches commodity futures and inventory data.

    Retrieves CME energy futures settlement strips and EIA petroleum
    inventory time series. All remote data is persisted to local Parquet
    files so that subsequent requests for the same date range are served
    from disk without hitting external APIs.

    Attributes:
        cache_dir: pathlib.Path to the directory used for Parquet cache
            files. Created automatically if it does not exist.
    """

    def __init__(self, eia_api_key=None, cache_dir="data/"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._eia_api_key = eia_api_key or os.environ.get("EIA_API_KEY")

    def fetch_futures_strip(self, product="CL", date=None):
        """Return a futures settlement strip from the local Parquet cache.

        Looks up a cached file keyed by product and date. If no cache entry
        exists, returns an empty dict (live API fetching for strips is not
        implemented here; populate the cache externally).

        Args:
            product: Commodity ticker symbol (e.g. "CL", "HO"). Defaults
                to "CL".
            date: ISO-format date string for which to retrieve the strip.
                Defaults to None, which resolves to today's date.

        Returns:
            A dict mapping contract code strings (e.g. "CLZ26") to their
            settlement prices as floats. Returns an empty dict when no
            cached data is found.
        """
        if date is None:
            date = datetime.date.today().isoformat()

        cache_key = f"strip_{product}_{date}"
        cached = self._read_cache(cache_key)
        if cached is not None:
            return cached.iloc[0].to_dict()

        return {}

    def fetch_inventory_history(self, start="2020-01-01", end=None):
        """Fetch weekly EIA petroleum inventory data for a date range.

        Checks the local Parquet cache first. On a cache miss, and when an
        EIA API key is configured, calls _fetch_eia_inventory() and writes
        the result to cache. Returns an empty DataFrame when neither cache
        nor API data is available.

        Args:
            start: ISO-format start date string for the history window.
                Defaults to "2020-01-01".
            end: ISO-format end date string. Defaults to None, which
                resolves to today's date.

        Returns:
            A pandas DataFrame indexed by week-end date with columns for
            each inventory series defined in EIA_SERIES (e.g.
            "crude_stocks", "gasoline_stocks", "distillate_stocks",
            "refinery_util"). Returns an empty DataFrame on failure.
        """
        if end is None:
            end = datetime.date.today().isoformat()

        cache_key = f"inventory_{start}_{end}"
        cached = self._read_cache(cache_key)
        if cached is not None:
            return cached

        if self._eia_api_key:
            try:
                df = self._fetch_eia_inventory(start, end)
                if df is not None and len(df) > 0:
                    self._write_cache(cache_key, df)
                    return df
            except Exception:
                pass

        return pd.DataFrame()

    def fetch_contract_expiries(self, product="CL", year=2026):
        """Compute the CME energy futures expiry dates for a given year.

        Approximates each monthly expiry as the third business day before
        the 25th of the month preceding the delivery month, consistent with
        standard CME WTI crude oil futures rules.

        Args:
            product: Commodity ticker symbol. Currently unused; the same
                expiry calculation is applied to all products. Defaults
                to "CL".
            year: Calendar year (int) for which to generate expiry dates.
                Defaults to 2026.

        Returns:
            A list of 12 datetime.date objects, one per calendar month,
            each representing the approximate futures expiry date for that
            delivery month in the specified year.
        """
        expiries = []
        for month in range(1, 13):
            if month == 1:
                exp_month, exp_year = 12, year - 1
            else:
                exp_month, exp_year = month - 1, year
            target = datetime.date(exp_year, exp_month, 25)
            bd_count = 0
            d = target
            while bd_count < 3:
                d -= datetime.timedelta(days=1)
                if d.weekday() < 5:
                    bd_count += 1
            expiries.append(d)
        return expiries

    def get_strip_for_date(self, date=None, product="CL"):
        """Return a settlement list ready for ForwardCurveBootstrapper.

        Fetches the cached futures strip, converts each contract code to a
        year-fraction time-to-expiry via _contract_to_expiry(), and packages
        the results as a list of dicts sorted by ascending time to expiry.
        Returns an empty list when no cached strip data is available.

        Args:
            date: ISO-format date string for which to build the settlement
                list. Defaults to None, which resolves to today's date.
            product: Commodity ticker symbol (e.g. "CL", "HO"). Defaults
                to "CL".

        Returns:
            A list of dicts, each containing the keys: "product",
            "contract_code", "settlement", "time_to_expiry", "volume", and
            "open_interest". The list is sorted by ascending "time_to_expiry".
            Returns an empty list if no strip data is found.
        """
        if date is None:
            date = datetime.date.today().isoformat()

        strip = self.fetch_futures_strip(product, date)
        if not strip:
            return []

        settlements = []
        for contract_code, price in strip.items():
            tte = _contract_to_expiry(contract_code, date)
            settlements.append({
                "product": product,
                "contract_code": contract_code,
                "settlement": price,
                "time_to_expiry": tte,
                "volume": 0,
                "open_interest": 0,
            })

        settlements.sort(key=lambda x: x["time_to_expiry"])
        return settlements

    def get_inventory_zscore(self, date=None):
        """Compute inventory z-scores relative to 5-year seasonal norms.

        Retrieves the most recent inventory observation from the fetched
        history and compares each stock level against the seasonal average
        for the relevant calendar month, stored in SEASONAL_INVENTORY_AVG.
        The standard deviation is assumed to be 5% of the seasonal mean.
        Returns zero for all series when no inventory data is available.

        Args:
            date: ISO-format date string used to determine the reference
                calendar month for seasonal comparison. Defaults to None,
                which uses today's month.

        Returns:
            A dict with keys "crude_stocks_z", "gasoline_stocks_z", and
            "distillate_stocks_z", each mapping to a float z-score rounded
            to three decimal places. Missing columns in the inventory data
            are reported as 0.0.
        """
        df = self.fetch_inventory_history()
        if df is None or len(df) == 0:
            return {"crude_stocks_z": 0.0, "gasoline_stocks_z": 0.0,
                    "distillate_stocks_z": 0.0}

        latest = df.iloc[-1]
        if date:
            month = datetime.date.fromisoformat(date).month
        else:
            month = datetime.date.today().month

        seasonal = SEASONAL_INVENTORY_AVG.get(month, SEASONAL_INVENTORY_AVG[1])

        results = {}
        for col, skey in [("crude_stocks", "crude"), ("gasoline_stocks", "gasoline"),
                          ("distillate_stocks", "distillate")]:
            if col in latest.index:
                std = seasonal[skey] * 0.05
                z = (latest[col] - seasonal[skey]) / std if std > 0 else 0.0
                results[f"{col}_z"] = round(z, 3)
            else:
                results[f"{col}_z"] = 0.0

        return results

    # -- Cache helpers --

    def _read_cache(self, key):
        """Read a cached DataFrame from a Parquet file.

        Constructs the file path from the cache directory and the provided
        key, then attempts to read and return the Parquet file. Returns None
        silently on any read failure or if the file does not exist.

        Args:
            key: Cache key string used to form the filename "<key>.parquet"
                inside the cache directory.

        Returns:
            A pandas DataFrame if the cache file exists and is readable,
            otherwise None.
        """
        path = self.cache_dir / f"{key}.parquet"
        if path.exists():
            try:
                return pd.read_parquet(path)
            except Exception:
                return None
        return None

    def _write_cache(self, key, data):
        """Persist a DataFrame or Series to a Parquet cache file.

        Converts a Series to a single-column DataFrame before writing.
        Write failures are silently suppressed so that a cache miss never
        disrupts the calling workflow.

        Args:
            key: Cache key string used to form the filename "<key>.parquet"
                inside the cache directory.
            data: A pandas DataFrame or Series to persist. Series objects
                are automatically converted to DataFrames before writing.
        """
        if isinstance(data, pd.Series):
            data = data.to_frame()
        path = self.cache_dir / f"{key}.parquet"
        try:
            data.to_parquet(path)
        except Exception:
            pass

    def _fetch_eia_inventory(self, start, end):
        """Fetch weekly petroleum inventory series from the EIA API v2.

        Iterates over all series defined in EIA_SERIES, requests each one
        from the EIA v2 REST endpoint, and assembles the results into a
        combined DataFrame. Individual series failures are silently skipped
        so a partial result is still returned when some calls succeed.

        Args:
            start: ISO-format start date string for the requested data range.
            end: ISO-format end date string for the requested data range.

        Returns:
            A pandas DataFrame indexed by week-end timestamps with one column
            per successfully retrieved series. Rows with all-NaN values are
            dropped. Returns None when no series could be fetched.
        """
        import requests

        frames = {}
        for label, series_id in EIA_SERIES.items():
            try:
                url = (f"https://api.eia.gov/v2/seriesid/{series_id}"
                       f"?api_key={self._eia_api_key}"
                       f"&start={start}&end={end}")
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                if "response" in data and "data" in data["response"]:
                    records = data["response"]["data"]
                    s = pd.Series(
                        {pd.Timestamp(r["period"]): float(r["value"]) for r in records},
                        name=label,
                    )
                    frames[label] = s
            except Exception:
                pass

        if not frames:
            return None
        return pd.DataFrame(frames).dropna(how="all").sort_index()
