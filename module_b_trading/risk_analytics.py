"""Risk analytics: bump-and-revalue sensitivities for energy commodity portfolios.

Implements a bump-and-revalue framework that rebuilds the forward
curve after applying price shocks and computes first- and
second-order sensitivities (delta, key-contract delta, gamma) for
individual positions and full portfolios. Also provides a risk
ladder for parallel shift scenarios and a scenario hedge-ratio
calculator.
"""

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from module_a_curves.curve_bootstrapper import ForwardCurveBootstrapper, ForwardCurve
from module_b_trading.futures_pricer import FuturesPricer, FuturesPosition


class RiskAnalytics:
    """Bump-and-revalue risk engine for commodity futures portfolios.

    Builds a base forward curve from the supplied settlement prices
    at construction time and exposes methods to compute first-order
    (parallel delta, key-contract delta), second-order (gamma), and
    scenario-based (risk ladder) sensitivities by rebuilding the
    curve after applying price bumps.

    Attributes:
        bootstrapper: ForwardCurveBootstrapper instance used to
            rebuild the curve after each bump.
        base_settlements: List of settlement objects (dicts or
            dataclass instances) that define the base curve.
        valuation_date: String valuation date passed to the
            bootstrapper (e.g. ``'2024-12-31'``).
        product: Product code string (e.g. ``'CL'``) used when
            bootstrapping the curve.
        base_curve: ForwardCurve constructed from the unshocked
            settlement prices; serves as the reference for all
            bump-and-revalue calculations.
        base_pricer: FuturesPricer wrapping ``base_curve``, used to
            compute the base mark-to-market value before any bump.
    """

    def __init__(self, bootstrapper, base_settlements, valuation_date="2026-03-09",
                 product="CL"):
        """Initialise the risk analytics engine.

        Bootstraps the base forward curve immediately so that all
        subsequent sensitivity calculations have a reference valuation
        available without re-running the bootstrapper.

        Args:
            bootstrapper: ForwardCurveBootstrapper instance.
            base_settlements: Iterable of settlement objects (dicts
                with ``time_to_expiry`` and ``settlement`` keys, or
                dataclass instances with the same attributes) defining
                the initial curve.
            valuation_date: Valuation date string forwarded to the
                bootstrapper. Defaults to ``'2024-12-31'``.
            product: Product code string forwarded to the
                bootstrapper. Defaults to ``'CL'``.
        """
        self.bootstrapper = bootstrapper
        self.base_settlements = list(base_settlements)
        self.valuation_date = valuation_date
        self.product = product

        self.base_curve = self.bootstrapper.bootstrap(
            self.base_settlements, self.valuation_date, self.product
        )
        self.base_pricer = FuturesPricer(self.base_curve)

    def parallel_delta(self, position, bump_usd=1.0):
        """Compute the parallel delta for a position via centered finite difference.

        Shifts every settlement price simultaneously up and down by
        ``bump_usd``, rebuilds the curve each time, marks the position
        to market, and returns the average of the two one-sided
        differences.

        Args:
            position: FuturesPosition instance to value.
            bump_usd: Magnitude of the parallel shift applied to all
                settlement prices, in USD per unit. Defaults to 1.0.

        Returns:
            Float representing the P&L change (in USD) for a
            ``bump_usd`` parallel shift in forward prices.
        """
        bump_map_up = {i: bump_usd for i in range(len(self.base_settlements))}
        curve_up = self._rebuild_with_bump(bump_map_up)
        v_up = FuturesPricer(curve_up).mark_to_market(position)

        bump_map_down = {i: -bump_usd for i in range(len(self.base_settlements))}
        curve_down = self._rebuild_with_bump(bump_map_down)
        v_down = FuturesPricer(curve_down).mark_to_market(position)

        return (v_up - v_down) / 2.0

    def key_contract_delta(self, position, bump_usd=1.0):
        """Compute the key-contract delta for each tenor point independently.

        For every settlement in ``base_settlements``, bumps that
        single price up by ``bump_usd`` while leaving all others
        unchanged, rebuilds the curve, marks the position to market,
        and records the resulting P&L change relative to the base.

        Args:
            position: FuturesPosition instance to value.
            bump_usd: USD amount by which each individual settlement
                price is bumped. Defaults to 1.0.

        Returns:
            pandas.Series indexed by tenor (in years) with values
            representing the P&L sensitivity (in USD) to a
            ``bump_usd`` move in each individual contract, sorted
            by ascending tenor.
        """
        v_base = self.base_pricer.mark_to_market(position)
        kcd = {}

        for i, s in enumerate(self.base_settlements):
            curve_bumped = self._rebuild_with_bump({i: bump_usd})
            v_bumped = FuturesPricer(curve_bumped).mark_to_market(position)
            tenor = s["time_to_expiry"] if isinstance(s, dict) else s.time_to_expiry
            kcd[tenor] = v_bumped - v_base

        result = pd.Series(kcd, name="KCD")
        result.index.name = "tenor"
        return result.sort_index()

    def gamma(self, position, bump_usd=5.0):
        """Compute the second-order price sensitivity (gamma) via centered finite difference.

        Applies a parallel bump of ``+bump_usd`` and ``-bump_usd``
        to all settlements, rebuilds the curve, and approximates
        gamma as:
            gamma = (V_up + V_down - 2 * V_base) / bump_usd^2

        Args:
            position: FuturesPosition instance to value.
            bump_usd: Size of the parallel shift in USD per unit
                applied in each direction. Defaults to 5.0.

        Returns:
            Float representing the second-order P&L sensitivity to
            a unit price move, in USD per (USD/unit)^2.
        """
        v_base = self.base_pricer.mark_to_market(position)

        bump_map_up = {i: bump_usd for i in range(len(self.base_settlements))}
        curve_up = self._rebuild_with_bump(bump_map_up)
        v_up = FuturesPricer(curve_up).mark_to_market(position)

        bump_map_down = {i: -bump_usd for i in range(len(self.base_settlements))}
        curve_down = self._rebuild_with_bump(bump_map_down)
        v_down = FuturesPricer(curve_down).mark_to_market(position)

        return (v_up + v_down - 2.0 * v_base) / (bump_usd ** 2)

    def portfolio_risk(self, positions):
        """Aggregate key-contract delta sensitivities across a portfolio.

        Computes the key-contract delta for each position and sums
        them by tenor to produce a portfolio-level risk ladder.
        Also derives a cumulative delta and a percentage-of-total
        column.

        Args:
            positions: Non-empty list of FuturesPosition instances.

        Returns:
            pandas.DataFrame indexed by tenor (in years) with columns
            ``KCD`` (aggregate key-contract delta in USD),
            ``cumulative`` (cumulative sum of KCD), and
            ``pct_of_total`` (each tenor's absolute KCD as a
            percentage of the sum of absolute KCDs).

        Raises:
            ValueError: If ``positions`` is empty.
        """
        if not positions:
            raise ValueError("Portfolio must contain at least one position")

        total_kcd = None
        for pos in positions:
            kcd_i = self.key_contract_delta(pos)
            if total_kcd is None:
                total_kcd = kcd_i.copy()
            else:
                total_kcd = total_kcd.add(kcd_i, fill_value=0.0)

        total_kcd = total_kcd.sort_index()
        result = pd.DataFrame({"KCD": total_kcd})
        result["cumulative"] = result["KCD"].cumsum()
        abs_total = result["KCD"].abs().sum()
        if abs_total > 0:
            result["pct_of_total"] = result["KCD"].abs() / abs_total * 100.0
        else:
            result["pct_of_total"] = 0.0
        result.index.name = "tenor"
        return result

    def risk_ladder(self, position, scenarios=None):
        """Compute P&L for a position under a series of parallel price shifts.

        For each shift level, bumps all settlement prices by that
        amount, rebuilds the curve, and records the resulting change
        in mark-to-market value relative to the base.

        Args:
            position: FuturesPosition instance to stress-test.
            scenarios: Optional list of integer or float shift values
                in USD per unit to apply as parallel bumps. If None,
                the default ladder ``[-20, -10, -5, -2, -1, 0, 1, 2,
                5, 10, 20]`` is used.

        Returns:
            pandas.DataFrame with columns ``scenario_usd`` (the
            applied shift) and ``pnl`` (the resulting USD P&L change
            relative to base), one row per scenario.
        """
        if scenarios is None:
            scenarios = [-20, -10, -5, -2, -1, 0, 1, 2, 5, 10, 20]

        v_base = self.base_pricer.mark_to_market(position)
        rows = []
        for shift in scenarios:
            if shift == 0:
                pnl = 0.0
            else:
                bump_map = {i: shift for i in range(len(self.base_settlements))}
                curve_shifted = self._rebuild_with_bump(bump_map)
                v_shifted = FuturesPricer(curve_shifted).mark_to_market(position)
                pnl = v_shifted - v_base
            rows.append({"scenario_usd": shift, "pnl": pnl})
        return pd.DataFrame(rows)

    def scenario_hedge_ratio(self, position, target_delta=0.0):
        """Calculate the number of additional contracts needed to hit a target delta.

        Computes the current parallel delta of the position and
        determines how many additional contracts of the same product
        would be required to close the gap to ``target_delta``.

        Args:
            position: FuturesPosition instance whose delta is to be
                hedged. The position's ``contract_size`` attribute is
                used as the denominator.
            target_delta: Desired parallel delta in USD after hedging.
                Defaults to 0.0 (flat/delta-neutral).

        Returns:
            Integer number of additional contracts required. Positive
            values indicate additional long contracts are needed;
            negative values indicate additional short contracts.
            Returns 0 if ``position.contract_size`` is not positive.
        """
        current_delta = self.parallel_delta(position)
        spec = position.contract_size
        if spec <= 0:
            return 0
        return int(round((target_delta - current_delta) / spec))

    def _rebuild_with_bump(self, bump_map):
        """Rebuild the forward curve after bumping selected settlement prices.

        Deep-copies the base settlement list, applies the specified
        additive bumps to the indicated indices, and bootstraps a new
        ForwardCurve from the modified settlements.

        Args:
            bump_map: Dictionary mapping integer settlement indices to
                the USD amount by which their price should be bumped
                (positive or negative).

        Returns:
            New ForwardCurve instance constructed from the bumped
            settlements.

        Raises:
            IndexError: If any index in ``bump_map`` is outside the
                valid range [0, len(base_settlements) - 1].
        """
        bumped = copy.deepcopy(self.base_settlements)
        for idx, bump_amount in bump_map.items():
            if idx < 0 or idx >= len(bumped):
                raise IndexError(f"Settlement index {idx} out of range")
            s = bumped[idx]
            if isinstance(s, dict):
                s["settlement"] += bump_amount
            else:
                s.settlement += bump_amount
        return self.bootstrapper.bootstrap(bumped, self.valuation_date, self.product)
