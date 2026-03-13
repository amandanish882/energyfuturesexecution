"""Data loader for commodity forward curve construction.

Provides utilities for fetching CME futures settlement strips and EIA
petroleum inventory data. Results are cached locally as Parquet files to
reduce repeated API calls. The primary entry point for curve building is
CommodityDataLoader.get_strip_for_date(), which returns a list of
settlement dicts ready for ForwardCurveBootstrapper.
"""

import datetime
import os
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Contract month codes
MONTH_CODE = {
    "F": 1, "G": 2, "H": 3, "J": 4, "K": 5, "M": 6,
    "N": 7, "Q": 8, "U": 9, "V": 10, "X": 11, "Z": 12,
}

# Reverse mapping: month number -> letter code
MONTH_LETTER = {v: k for k, v in MONTH_CODE.items()}

# EIA API series IDs — petroleum
EIA_SERIES = {
    "crude_stocks": "PET.WCESTUS1.W",
    "gasoline_stocks": "PET.WGTSTUS1.W",
    "distillate_stocks": "PET.WDISTUS1.W",
    "refinery_util": "PET.WPULEUS3.W",
}

# EIA API series IDs — daily spot prices
EIA_SPOT_SERIES = {
    "CL": "PET.RWTC.D",                       # WTI Cushing $/bbl
    "HO": "PET.EER_EPD2F_PF4_Y35NY_DPG.D",   # NY Harbor No.2 Heating Oil $/gal
    "RB": "PET.EER_EPMRU_PF4_Y35NY_DPG.D",   # NY Harbor RBOB Gasoline $/gal
    "NG": "NG.RNGWHHD.D",                     # Henry Hub $/MMBtu
}

# EIA API series ID — natural gas storage (lower-48 working gas, Bcf)
EIA_NG_STORAGE_SERIES = "NG.NW2_EPG0_SWO_R48_BCF.W"

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

# 5-year seasonal averages for NG working gas storage (Bcf)
NG_SEASONAL_STORAGE_AVG = {
    1: 2600,   # mid-winter draw
    2: 2200,   # late-winter draw
    3: 1800,   # end-of-winter low
    4: 1700,   # early injection season
    5: 2000,   # injection ramp-up
    6: 2400,   # strong injection
    7: 2800,   # peak injection
    8: 3100,   # late injection
    9: 3350,   # nearing peak
    10: 3550,  # peak storage (pre-winter)
    11: 3400,  # early withdrawal
    12: 3000,  # winter draw begins
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

    def __init__(self, eia_api_key=None, databento_api_key=None, cache_dir="data/"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._eia_api_key = eia_api_key or os.environ.get("EIA_API_KEY")
        self._databento_api_key = databento_api_key or os.environ.get("DATABENTO_API_KEY")

    def fetch_futures_strip(self, product="CL", date=None):
        """Return a futures settlement strip, downloading from Databento on cache miss.

        Looks up a cached file keyed by product and date. If no cache entry
        exists and a Databento API key is configured, downloads ohlcv-1d
        data from CME Globex (GLBX.MDP3) and caches the result.

        Args:
            product: Commodity ticker symbol (e.g. "CL", "HO"). Defaults
                to "CL".
            date: ISO-format date string for which to retrieve the strip.
                Defaults to None, which resolves to today's date.

        Returns:
            A dict mapping contract code strings (e.g. "CLZ26") to their
            settlement prices as floats. Returns an empty dict when no
            cached data is found and no API key is available.
        """
        if date is None:
            date = datetime.date.today().isoformat()

        cache_key = f"strip_{product}_{date}"
        cached = self._read_cache(cache_key)
        if cached is not None:
            return cached.iloc[0].to_dict()

        # Cache miss — try downloading from Databento
        if self._databento_api_key:
            strip = self._fetch_databento_strip(product, date)
            if strip:
                strip_df = pd.DataFrame([strip])
                self._write_cache(cache_key, strip_df)
                return strip
        return {}

    def _build_strip_symbols(self, product, ref_date_str):
        """Build CME Globex contract symbols for the next 12 months.

        For example, if ref_date is "2026-03-10" and product is "CL",
        returns ["CLJ26", "CLK26", "CLM26", ...].

        Args:
            product: Root ticker string (e.g. "CL").
            ref_date_str: ISO-format date string.

        Returns:
            A list of 12 contract symbol strings.
        """
        ref = datetime.date.fromisoformat(ref_date_str)
        symbols = []
        month = ref.month + 1
        year = ref.year
        if month > 12:
            month = 1
            year += 1

        # Databento symbol year format varies by product:
        #   CL, HO, RB: single-digit year (CLJ6 = April 2026)
        #   NG: two-digit year (NGJ26 = April 2026)
        two_digit_products = {"NG"}

        for _ in range(12):
            code = MONTH_LETTER[month]
            if product in two_digit_products:
                yr = year % 100
            else:
                yr = year % 10 if year < 2030 else year % 100
            symbols.append(f"{product}{code}{yr}")
            month += 1
            if month > 12:
                month = 1
                year += 1
        return symbols

    def _fetch_databento_strip(self, product, date_str):
        """Download a settlement strip from Databento ohlcv-1d data.

        Fetches daily OHLCV bars for 12 forward contract months from
        CME Globex via the Databento Historical API. Uses the close
        price as the settlement price.

        Args:
            product: Commodity ticker (e.g. "CL", "HO").
            date_str: ISO-format valuation date string.

        Returns:
            A dict mapping contract symbols (e.g. "CLK26") to close
            prices, or an empty dict on failure.
        """
        try:
            import databento as db
        except ImportError:
            return {}

        symbols = self._build_strip_symbols(product, date_str)
        val_date = datetime.date.fromisoformat(date_str)
        start = (val_date - timedelta(days=5)).isoformat()
        end = (val_date + timedelta(days=1)).isoformat()

        try:
            client = db.Historical(self._databento_api_key)
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
                return {}

            df = df.reset_index()
            if "ts_event" in df.columns:
                df["trade_date"] = pd.to_datetime(df["ts_event"]).dt.date
            elif "date" in df.columns:
                df["trade_date"] = pd.to_datetime(df["date"]).dt.date

            latest_date = df["trade_date"].max()
            latest = df[df["trade_date"] == latest_date]

            strip = {}
            for _, row in latest.iterrows():
                sym = row.get("symbol", row.get("raw_symbol", ""))
                close = row["close"]
                if close > 0:
                    # Normalise to two-digit year (CLJ6 -> CLJ26) for cache
                    # consistency with existing parquet files
                    root = sym[:len(product)]
                    rest = sym[len(product):]  # e.g. "J6" or "J26"
                    month_char = rest[0]
                    yr_part = rest[1:]
                    if len(yr_part) == 1:
                        yr_part = f"2{yr_part}"
                    strip[f"{root}{month_char}{yr_part}"] = close

            return strip
        except Exception:
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

        Retrieves the most recent petroleum inventory and natural gas
        storage observations, compares each against its seasonal average
        for the relevant calendar month, and returns z-scores for all
        four series. Petroleum std is 5% of seasonal mean; NG storage
        std is 8% (higher volatility).

        Args:
            date: ISO-format date string used to determine the reference
                calendar month for seasonal comparison. Defaults to None,
                which uses today's month.

        Returns:
            A dict with keys "crude_stocks_z", "gasoline_stocks_z",
            "distillate_stocks_z", and "ng_storage_z", each mapping to
            a float z-score rounded to three decimal places.
        """
        results = {"crude_stocks_z": 0.0, "gasoline_stocks_z": 0.0,
                   "distillate_stocks_z": 0.0, "ng_storage_z": 0.0}

        if date:
            month = datetime.date.fromisoformat(date).month
        else:
            month = datetime.date.today().month

        # Petroleum inventories
        df = self.fetch_inventory_history(end=date)
        if df is not None and len(df) > 0:
            latest = df.iloc[-1]
            seasonal = SEASONAL_INVENTORY_AVG.get(month, SEASONAL_INVENTORY_AVG[1])
            for col, skey in [("crude_stocks", "crude"), ("gasoline_stocks", "gasoline"),
                              ("distillate_stocks", "distillate")]:
                if col in latest.index:
                    std = seasonal[skey] * 0.05
                    z = (latest[col] - seasonal[skey]) / std if std > 0 else 0.0
                    results[f"{col}_z"] = round(z, 3)

        # Natural gas storage
        ng_df = self.fetch_ng_storage_history(end=date)
        if ng_df is not None and len(ng_df) > 0:
            ng_latest = ng_df.iloc[-1]["ng_storage_bcf"]
            ng_seasonal = NG_SEASONAL_STORAGE_AVG.get(month, NG_SEASONAL_STORAGE_AVG[1])
            ng_std = ng_seasonal * 0.08
            ng_z = (ng_latest - ng_seasonal) / ng_std if ng_std > 0 else 0.0
            results["ng_storage_z"] = round(ng_z, 3)

        return results

    def fetch_ng_storage_history(self, start="2020-01-01", end=None):
        """Fetch weekly EIA natural gas storage data for a date range.

        Retrieves the lower-48 working gas in underground storage series
        from EIA API v2. Results are cached as Parquet.

        Args:
            start: ISO-format start date string. Defaults to "2020-01-01".
            end: ISO-format end date string. Defaults to today.

        Returns:
            A pandas DataFrame indexed by week-end date with a single
            "ng_storage_bcf" column (values in Bcf). Returns an empty
            DataFrame on failure.
        """
        if end is None:
            end = datetime.date.today().isoformat()

        cache_key = f"ng_storage_{start}_{end}"
        cached = self._read_cache(cache_key)
        if cached is not None:
            return cached

        if self._eia_api_key:
            try:
                import requests

                url = (f"https://api.eia.gov/v2/seriesid/{EIA_NG_STORAGE_SERIES}"
                       f"?api_key={self._eia_api_key}"
                       f"&start={start}&end={end}")
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                if "response" in data and "data" in data["response"]:
                    records = data["response"]["data"]
                    s = pd.Series(
                        {pd.Timestamp(r["period"]): float(r["value"]) for r in records},
                        name="ng_storage_bcf",
                    )
                    df = s.to_frame().dropna().sort_index()
                    if len(df) > 0:
                        self._write_cache(cache_key, df)
                        return df
            except Exception:
                pass

        return pd.DataFrame()

    def fetch_spot_price(self, product="CL", date=None):
        """Fetch the EIA daily spot price for a commodity.

        Retrieves the most recent daily spot price observation on or before
        the given date from the EIA API v2. Results are cached as Parquet.

        Args:
            product: Commodity ticker (e.g. "CL", "HO", "RB", "NG").
                Defaults to "CL".
            date: ISO-format date string. Defaults to today.

        Returns:
            Float spot price in USD per unit, or None if unavailable.
        """
        if date is None:
            date = datetime.date.today().isoformat()

        series_id = EIA_SPOT_SERIES.get(product)
        if series_id is None:
            return None

        cache_key = f"spot_{product}_{date}"
        cached = self._read_cache(cache_key)
        if cached is not None and len(cached) > 0:
            return float(cached.iloc[-1]["spot_price"])

        if self._eia_api_key:
            try:
                import requests

                # Fetch last 10 days to handle weekends/holidays
                start = (datetime.date.fromisoformat(date) - timedelta(days=10)).isoformat()

                if product == "NG":
                    # NG spot uses native v2 natural-gas endpoint (not /seriesid/)
                    url = (f"https://api.eia.gov/v2/natural-gas/pri/fut/data"
                           f"?api_key={self._eia_api_key}"
                           f"&frequency=daily&data[0]=value"
                           f"&facets[series][]=RNGWHHD"
                           f"&start={start}&end={date}")
                else:
                    url = (f"https://api.eia.gov/v2/seriesid/{series_id}"
                           f"?api_key={self._eia_api_key}"
                           f"&start={start}&end={date}")

                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                if "response" in data and "data" in data["response"]:
                    records = data["response"]["data"]
                    s = pd.Series(
                        {pd.Timestamp(r["period"]): float(r["value"]) for r in records},
                        name="spot_price",
                    )
                    df = s.to_frame().dropna().sort_index()
                    if len(df) > 0:
                        self._write_cache(cache_key, df)
                        return float(df.iloc[-1]["spot_price"])
            except Exception:
                pass

        return None

    def get_ng_storage_zscore(self, date=None):
        """Compute natural gas storage z-score relative to seasonal norms.

        Retrieves the most recent NG storage observation and compares it
        against the 5-year seasonal average for the relevant calendar
        month from NG_SEASONAL_STORAGE_AVG. The standard deviation is
        assumed to be 8% of the seasonal mean (NG storage is more
        volatile than petroleum inventories).

        Args:
            date: ISO-format date string for the reference month.
                Defaults to today.

        Returns:
            A dict with key "ng_storage_z" mapping to a float z-score
            rounded to three decimal places.
        """
        df = self.fetch_ng_storage_history(end=date)
        if df is None or len(df) == 0:
            return {"ng_storage_z": 0.0}

        latest = df.iloc[-1]["ng_storage_bcf"]
        if date:
            month = datetime.date.fromisoformat(date).month
        else:
            month = datetime.date.today().month

        seasonal = NG_SEASONAL_STORAGE_AVG.get(month, NG_SEASONAL_STORAGE_AVG[1])
        std = seasonal * 0.08
        z = (latest - seasonal) / std if std > 0 else 0.0
        return {"ng_storage_z": round(z, 3)}

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
