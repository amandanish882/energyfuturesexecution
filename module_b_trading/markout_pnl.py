"""Markout P&L analyser for commodity futures market-making.

Simulates and aggregates markout price paths across a set of
configurable time horizons, converts raw price moves into dollar
P&L, and summarises results by product and client segment to
support adverse-selection analysis for a market-making desk.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd


_PRODUCT_VOL = {
    "CL": 1.50,
    "HO": 0.04,
    "RB": 0.05,
    "NG": 0.15,
}

_CONTRACT_SIZES = {
    "CL": 1000,
    "HO": 42000,
    "RB": 42000,
    "NG": 10000,
}

MARKOUT_HORIZONS = [1, 5, 15, 30, 60, 300, 900, 3600]  # seconds


class MarkoutAnalyzer:
    """Simulates and analyses markout P&L for commodity futures market-making trades.

    For each trade, markout price moves are simulated at a set of
    configurable time horizons using a combination of random diffusion
    (scaled to the product's historical volatility) and an
    exponentially decaying alpha edge. The resulting price moves are
    converted to dollar P&L using per-product contract sizes, then
    aggregated by product and client segment to expose adverse
    selection and edge decay patterns.

    Attributes:
        _horizons: List of integer time horizons in seconds at which
            markout prices are evaluated (e.g. [1, 5, 15, 30, ...]).
    """

    def __init__(self, horizons=None):
        """Initialise the markout analyser.

        Args:
            horizons: Optional list of integer time horizons in
                seconds. If None, ``MARKOUT_HORIZONS`` is used,
                which covers [1, 5, 15, 30, 60, 300, 900, 3600].
        """
        self._horizons = horizons or MARKOUT_HORIZONS

    def simulate_markouts(self, trades, seed=42):
        """Simulate markout price paths for a set of trades.

        For each horizon in ``_horizons``, draws a normally
        distributed random price move scaled to the product's
        volatility and the horizon's time fraction of a trading day,
        then adds an exponentially decaying alpha component. Results
        are stored as new columns (``markout_Xs`` for horizon X) on
        a copy of the input DataFrame.

        Args:
            trades: pandas.DataFrame where each row represents one
                trade. Expected columns: ``product`` (str), ``edge``
                (float, expected edge per unit), ``direction``
                (``'buy'`` or ``'sell'``).
            seed: Integer random seed for reproducibility. Defaults
                to 42.

        Returns:
            pandas.DataFrame identical to ``trades`` with additional
            float columns ``markout_1s``, ``markout_5s``, etc. for
            every horizon, representing the simulated price move in
            USD per unit.
        """
        rng = np.random.RandomState(seed)
        result = trades.copy()

        for h in self._horizons:
            markouts = []
            for _, row in trades.iterrows():
                product = row.get("product", "CL")
                vol = _PRODUCT_VOL.get(product, 1.0)
                edge = row.get("edge", 0.0)
                direction = 1 if row.get("direction", "buy") == "buy" else -1

                dt = h / (252 * 6.5 * 3600)
                random_move = rng.normal(0, vol * np.sqrt(dt * 252))
                alpha_decay = edge * np.exp(-h / 300)
                markout = direction * (random_move + alpha_decay * 0.3)
                markouts.append(markout)

            result[f"markout_{h}s"] = markouts

        return result

    def compute_markout_pnl(self, markout_df):
        """Convert markout price moves to dollar P&L for each horizon.

        Multiplies each ``markout_Xs`` column by the number of
        contracts (``num_contracts``) and the per-product contract
        size to produce ``pnl_Xs`` columns in USD. Operates on a
        copy of the input DataFrame.

        Args:
            markout_df: pandas.DataFrame that must contain columns
                ``product``, optional ``num_contracts`` (defaults to
                1 if absent), and ``markout_Xs`` columns for each
                configured horizon.

        Returns:
            pandas.DataFrame with all original columns plus new
            ``pnl_Xs`` float columns representing dollar P&L at each
            horizon.
        """
        result = markout_df.copy()
        for h in self._horizons:
            col = f"markout_{h}s"
            pnl_col = f"pnl_{h}s"
            if col in result.columns:
                cs = result["product"].map(_CONTRACT_SIZES).fillna(1000)
                nc = result.get("num_contracts", 1)
                result[pnl_col] = result[col] * nc * cs
        return result

    def summary_by_product(self, markout_df):
        """Summarise markout P&L statistics grouped by product.

        If ``pnl_Xs`` columns are not yet present in the DataFrame,
        ``compute_markout_pnl`` is called first. Groups rows by the
        ``product`` column and computes mean, standard deviation, and
        count for each P&L horizon column.

        Args:
            markout_df: pandas.DataFrame containing at least a
                ``product`` column and either ``pnl_Xs`` or
                ``markout_Xs`` columns for each configured horizon.

        Returns:
            pandas.DataFrame with a MultiIndex of columns
            (horizon, statistic) grouped by ``product``. Returns an
            empty DataFrame if no P&L columns are present or if the
            ``product`` column is missing.
        """
        pnl_cols = [f"pnl_{h}s" for h in self._horizons if f"pnl_{h}s" in markout_df.columns]
        if not pnl_cols:
            markout_df = self.compute_markout_pnl(markout_df)
            pnl_cols = [f"pnl_{h}s" for h in self._horizons if f"pnl_{h}s" in markout_df.columns]

        if not pnl_cols or "product" not in markout_df.columns:
            return pd.DataFrame()

        return markout_df.groupby("product")[pnl_cols].agg(["mean", "std", "count"])

    def summary_by_segment(self, markout_df):
        """Summarise markout P&L statistics grouped by client segment.

        If ``pnl_Xs`` columns are not yet present in the DataFrame,
        ``compute_markout_pnl`` is called first. Groups rows by the
        ``client_segment`` column and computes mean, standard
        deviation, and count for each P&L horizon column.

        Args:
            markout_df: pandas.DataFrame containing at least a
                ``client_segment`` column and either ``pnl_Xs`` or
                ``markout_Xs`` columns for each configured horizon.

        Returns:
            pandas.DataFrame with a MultiIndex of columns
            (horizon, statistic) grouped by ``client_segment``.
            Returns an empty DataFrame if no P&L columns are present
            or if the ``client_segment`` column is missing.
        """
        pnl_cols = [f"pnl_{h}s" for h in self._horizons if f"pnl_{h}s" in markout_df.columns]
        if not pnl_cols:
            markout_df = self.compute_markout_pnl(markout_df)
            pnl_cols = [f"pnl_{h}s" for h in self._horizons if f"pnl_{h}s" in markout_df.columns]

        if not pnl_cols or "client_segment" not in markout_df.columns:
            return pd.DataFrame()

        return markout_df.groupby("client_segment")[pnl_cols].agg(["mean", "std", "count"])

    def adverse_selection_score(self, markout_df, horizon=300):
        """Compute an adverse selection score for each client segment.

        Groups trades by ``client_segment`` and, for the specified
        horizon, calculates the mean and standard deviation of dollar
        P&L. The adverse selection score is defined as:
            adverse_score = -mean_pnl / std_pnl

        A high positive score indicates that the market maker
        consistently loses money at that horizon after trading with
        a segment (a signal of informed counterparties).

        Args:
            markout_df: pandas.DataFrame containing at least
                ``client_segment`` and either ``pnl_Xs`` or
                ``markout_Xs`` columns. The column ``pnl_{horizon}s``
                must be derivable from the data.
            horizon: Integer time horizon in seconds for which to
                compute the score. Must be a value present in
                ``_horizons``. Defaults to 300 (5 minutes).

        Returns:
            pandas.DataFrame with columns ``client_segment``,
            ``mean_pnl``, ``std_pnl``, ``n_trades``, and
            ``adverse_score``, one row per segment. Returns an empty
            DataFrame if the required columns are unavailable.
        """
        pnl_col = f"pnl_{horizon}s"
        if pnl_col not in markout_df.columns:
            markout_df = self.compute_markout_pnl(markout_df)

        if pnl_col not in markout_df.columns or "client_segment" not in markout_df.columns:
            return pd.DataFrame()

        scores = markout_df.groupby("client_segment").agg(
            mean_pnl=(pnl_col, "mean"),
            std_pnl=(pnl_col, "std"),
            n_trades=(pnl_col, "count"),
        ).reset_index()

        scores["adverse_score"] = -scores["mean_pnl"] / scores["std_pnl"].replace(0, np.nan)
        scores["adverse_score"] = scores["adverse_score"].fillna(0)
        return scores

    def __repr__(self):
        """Return a concise string representation of the analyser.

        Returns:
            String of the form ``MarkoutAnalyzer(horizons=[...])``
            listing the configured time horizons in seconds.
        """
        return f"MarkoutAnalyzer(horizons={self._horizons})"
