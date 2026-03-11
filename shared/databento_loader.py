"""Databento MBP-10 data loader for energy commodity futures.

Loads Level 2 order book data (Market-By-Price, 10 levels) from
Databento for NYMEX energy futures via the Historical API.

Databento uses CME Globex symbology where contract symbols follow the
pattern ``{root}{month_code}{year_digit}`` — for example ``CLH5`` is
the March 2025 WTI Crude Oil future. The ``MONTH_CODES`` mapping
converts calendar months to the standard single-letter codes.

The default dataset is ``GLBX.MDP3`` (CME Globex) and the default
schema is ``mbp-10`` (Market-By-Price with 10 depth levels).
"""

import numpy as np
import pandas as pd

import databento as db


TICKERS = ["CL", "HO", "RB", "NG"]

DATASET = "GLBX.MDP3"  # CME Globex

SCHEMA = "mbp-10"  # Market-By-Price, 10 levels

MONTH_CODES = {
    1: "F", 2: "G", 3: "H", 4: "J", 5: "K", 6: "M",
    7: "N", 8: "Q", 9: "U", 10: "V", 11: "X", 12: "Z",
}


def front_month_symbol(product, date_str):
    """Build the Databento front-month contract symbol for a product and date.

    Determines the nearest active contract month based on the given
    date. If the current month is past the 15th, rolls to the next
    month to avoid expiry. Constructs a CME Globex symbol using the
    single-letter month code and single-digit year.

    Args:
        product: Root ticker string (e.g. ``'CL'``, ``'HO'``).
        date_str: Date string in ``YYYY-MM-DD`` format used to
            determine the nearest contract month.

    Returns:
        Symbol string in CME Globex format (e.g. ``'CLH5'`` for
        March 2025 WTI Crude).
    """
    parts = date_str.split("-")
    year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
    # CL expires around the 20th of the month before delivery,
    # so by mid-month the front contract is already next month
    month += 1
    if month > 12:
        month = 1
        year += 1
    code = MONTH_CODES[month]
    return f"{product}{code}{year % 10}"


class DatabentoConfig:
    """Databento API connection configuration.

    Attributes:
        api_key: Databento API key string.
        dataset: CME Globex dataset identifier. Defaults to
            ``'GLBX.MDP3'``.
        schema: Market data schema. Defaults to ``'mbp-10'``
            (Market-By-Price, 10 depth levels).
    """

    def __init__(self, api_key="", dataset=DATASET, schema=SCHEMA):
        self.api_key = api_key
        self.dataset = dataset
        self.schema = schema


class DatabentoLoader:
    """Loads MBP-10 order book data and trades from Databento.

    Connects to the Databento Historical API and retrieves Level 2
    book snapshots and trade records for NYMEX energy futures. Symbols
    are resolved via ``front_month_symbol`` to the correct CME Globex
    contract format.

    Attributes:
        _config: DatabentoConfig with API key and dataset settings.
        _client: databento.Historical client instance.
    """

    def __init__(self, config):
        """Initialise the Databento loader.

        Args:
            config: DatabentoConfig with a valid API key.
        """
        self._config = config
        self._client = db.Historical(config.api_key)

    def load_book_snapshots(self, symbol, date, n_snapshots=100):
        """Load L2 book snapshots for a symbol on a given date.

        Retrieves MBP-10 data from the NYMEX open (09:00 ET) to
        close (14:30 ET) window, limited to ``n_snapshots`` rows.

        Args:
            symbol: CME Globex contract symbol (e.g. ``'CLH5'``).
                Use ``front_month_symbol`` to build this from a
                product code and date.
            date: Date string in ``YYYY-MM-DD`` format.
            n_snapshots: Maximum number of book update rows to
                return. Defaults to 100.

        Returns:
            pandas.DataFrame with MBP-10 columns including
            ``bid_px_00`` through ``bid_px_09``, corresponding
            ``ask_px_*`` and size columns, and ``ts_event``.
        """
        data = self._client.timeseries.get_range(
            dataset=self._config.dataset,
            schema=self._config.schema,
            symbols=[symbol],
            start=f"{date}T09:00",
            end=f"{date}T14:30",
            limit=n_snapshots,
        )
        return data.to_df()

    def load_trades(self, symbol, date, n_trades=500):
        """Load trade records for a symbol on a given date.

        Retrieves trade-level data from the NYMEX trading window,
        limited to ``n_trades`` rows.

        Args:
            symbol: CME Globex contract symbol (e.g. ``'CLH5'``).
            date: Date string in ``YYYY-MM-DD`` format.
            n_trades: Maximum number of trade rows to return.
                Defaults to 500.

        Returns:
            pandas.DataFrame with trade columns including ``price``,
            ``size``, ``side``, and ``ts_event``.
        """
        data = self._client.timeseries.get_range(
            dataset=self._config.dataset,
            schema="trades",
            symbols=[symbol],
            start=f"{date}T09:00",
            end=f"{date}T14:30",
            limit=n_trades,
        )
        return data.to_df()

    def __repr__(self):
        """Return a string representation of the loader.

        Returns:
            String of the form ``DatabentoLoader(tickers=[...])``.
        """
        return f"DatabentoLoader(tickers={TICKERS})"
