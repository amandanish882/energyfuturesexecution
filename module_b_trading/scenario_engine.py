"""Scenario engine for commodity portfolio stress testing.

Applies named or custom price shocks to a bootstrapped forward
curve and computes the resulting P&L impact on a commodity futures
portfolio. Supports parallel shifts, curve twists, and arbitrary
tenor-specific bumps. Ships with a library of standard energy-market
stress scenarios (OPEC cuts, inventory surprises, geopolitical
spikes, demand destruction, etc.).
"""

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from module_a_curves.curve_bootstrapper import ForwardCurveBootstrapper, ForwardCurve
from module_b_trading.futures_pricer import FuturesPricer, FuturesPosition


STANDARD_SCENARIOS = {
    "opec_cut_10": {
        "type": "parallel", "shift_usd": 10.0,
        "description": "OPEC supply cut (+$10/bbl)",
    },
    "opec_flood_minus10": {
        "type": "parallel", "shift_usd": -10.0,
        "description": "OPEC flood market (-$10/bbl)",
    },
    "inventory_build_5m": {
        "type": "parallel", "shift_usd": -3.0,
        "description": "EIA surprise build 5M bbl (-$3)",
    },
    "inventory_draw_5m": {
        "type": "parallel", "shift_usd": 3.0,
        "description": "EIA surprise draw 5M bbl (+$3)",
    },
    "contango_steepen": {
        "type": "twist", "front_usd": -2.0, "back_usd": 2.0,
        "description": "Contango deepens (front -$2, back +$2)",
    },
    "backwardation_deepen": {
        "type": "twist", "front_usd": 3.0, "back_usd": -1.0,
        "description": "Backwardation deepens (front +$3, back -$1)",
    },
    "demand_destruction": {
        "type": "parallel", "shift_usd": -15.0,
        "description": "Recession/demand destruction (-$15)",
    },
    "geopolitical_spike": {
        "type": "parallel", "shift_usd": 20.0,
        "description": "Geopolitical supply disruption (+$20)",
    },
    "crack_blowout": {
        "type": "parallel", "shift_usd": 5.0,
        "description": "Refinery outage: crude rallies (+$5)",
    },
}


class ScenarioEngine:
    """Runs a commodity futures portfolio through named and custom stress scenarios.

    Builds a base forward curve at construction time and provides
    methods to apply individual scenarios, iterate over all standard
    scenarios, or run arbitrary tenor-specific custom bumps. Scenario
    P&L is always measured as the change in mark-to-market value
    relative to the unshocked base curve.

    Attributes:
        bootstrapper: ForwardCurveBootstrapper used to rebuild the
            curve for each scenario.
        base_settlements: List of settlement objects defining the
            unshocked base curve.
        valuation_date: String valuation date forwarded to the
            bootstrapper (e.g. ``'2024-12-31'``).
        product: Product code string forwarded to the bootstrapper
            (e.g. ``'CL'``).
        base_curve: ForwardCurve constructed from the base
            settlements; used as the reference valuation.
        base_pricer: FuturesPricer wrapping ``base_curve``.
    """

    def __init__(self, bootstrapper, base_settlements, valuation_date="2026-03-09",
                 product="CL"):
        """Initialise the scenario engine and bootstrap the base curve.

        Args:
            bootstrapper: ForwardCurveBootstrapper instance used to
                rebuild the forward curve under each scenario.
            base_settlements: Iterable of settlement objects (dicts
                with ``time_to_expiry`` and ``settlement`` keys, or
                dataclass instances with equivalent attributes).
            valuation_date: Valuation date string forwarded to the
                bootstrapper. Defaults to ``'2024-12-31'``.
            product: Product code string forwarded to the
                bootstrapper. Defaults to ``'CL'``.
        """
        self.bootstrapper = bootstrapper
        self.base_settlements = base_settlements
        self.valuation_date = valuation_date
        self.product = product
        self.base_curve = bootstrapper.bootstrap(base_settlements, valuation_date, product)
        self.base_pricer = FuturesPricer(self.base_curve)

    def apply_scenario(self, scenario):
        """Apply a price shock scenario to the settlement prices and return a new curve.

        Computes per-settlement bumps via ``_compute_bumps``, applies
        them to shallow copies of the base settlement objects, and
        bootstraps a new ForwardCurve from the shocked prices.

        Args:
            scenario: Dictionary describing the shock. Must contain a
                ``type`` key (``'parallel'``, ``'twist'``, or
                ``'custom'``) plus type-specific parameters:
                    - parallel: ``shift_usd`` (float).
                    - twist: ``front_usd`` (float), ``back_usd``
                      (float).
                    - custom: ``bumps_by_tenor`` ({tenor: usd}).

        Returns:
            New ForwardCurve instance constructed from the shocked
            settlement prices.
        """
        bumps = self._compute_bumps(scenario)
        bumped_settlements = []
        for i, s in enumerate(self.base_settlements):
            if isinstance(s, dict):
                new_s = copy.copy(s)
                new_s["settlement"] = s["settlement"] + bumps[i]
            else:
                new_s = copy.copy(s)
                new_s.settlement = s.settlement + bumps[i]
            bumped_settlements.append(new_s)
        return self.bootstrapper.bootstrap(bumped_settlements, self.valuation_date, self.product)

    def run_scenario(self, scenario, portfolio):
        """Run a single scenario on a portfolio and return position-level P&L.

        Applies the scenario to build a shocked forward curve, marks
        every position to market under both the base and shocked
        curves, and records the P&L difference for each position.

        Args:
            scenario: Scenario dictionary accepted by ``apply_scenario``
                (type plus type-specific shock parameters).
            portfolio: List of FuturesPosition instances constituting
                the portfolio to stress-test.

        Returns:
            pandas.DataFrame with one row per position and columns:
            ``position`` (ticker), ``product``, ``contracts``,
            ``base_mtm`` (USD), ``scenario_mtm`` (USD),
            ``scenario_pnl`` (USD), and ``pnl_per_contract`` (USD).
        """
        scenario_curve = self.apply_scenario(scenario)
        scenario_pricer = FuturesPricer(scenario_curve)
        rows = []
        for pos in portfolio:
            base_mtm = self.base_pricer.mark_to_market(pos)
            scen_mtm = scenario_pricer.mark_to_market(pos)
            pnl = scen_mtm - base_mtm
            rows.append({
                "position": pos.ticker,
                "product": pos.product,
                "contracts": pos.num_contracts,
                "base_mtm": base_mtm,
                "scenario_mtm": scen_mtm,
                "scenario_pnl": pnl,
                "pnl_per_contract": pnl / pos.num_contracts if pos.num_contracts > 0 else 0,
            })
        return pd.DataFrame(rows)

    def run_all_standard(self, portfolio):
        """Run every scenario in STANDARD_SCENARIOS and return a pivot table.

        Iterates over the STANDARD_SCENARIOS dictionary, calls
        ``run_scenario`` for each, and assembles the per-position
        P&L results into a single pivot table with scenario
        descriptions as the row index and position tickers as
        columns.

        Args:
            portfolio: List of FuturesPosition instances to
                stress-test across all standard scenarios.

        Returns:
            pandas.DataFrame where rows are scenario description
            strings and columns are position ticker strings. Each
            cell contains the scenario P&L in USD for that
            (scenario, position) combination.
        """
        results = {}
        for name, scenario in STANDARD_SCENARIOS.items():
            df = self.run_scenario(scenario, portfolio)
            label = scenario.get("description", name)
            results[label] = {
                row["position"]: row["scenario_pnl"] for _, row in df.iterrows()
            }
        return pd.DataFrame(results).T

    def run_custom_scenario(self, bumps_by_tenor, portfolio):
        """Run a custom tenor-specific shock scenario on a portfolio.

        Constructs a ``'custom'`` scenario dictionary from the
        provided tenor-to-bump mapping and delegates to
        ``run_scenario``.

        Args:
            bumps_by_tenor: Dictionary mapping tenor values in years
                (floats) to the USD bump to apply at that tenor.
                Intermediate tenors are interpolated linearly.
            portfolio: List of FuturesPosition instances to
                stress-test.

        Returns:
            pandas.DataFrame identical in structure to the output of
            ``run_scenario``: one row per position with columns
            ``position``, ``product``, ``contracts``, ``base_mtm``,
            ``scenario_mtm``, ``scenario_pnl``, and
            ``pnl_per_contract``.
        """
        scenario = {"type": "custom", "bumps_by_tenor": bumps_by_tenor}
        return self.run_scenario(scenario, portfolio)

    def _compute_bumps(self, scenario):
        """Compute a per-settlement bump array for a given scenario definition.

        Interprets the scenario type and parameters to derive an
        array of additive USD bumps, one per settlement in
        ``base_settlements``:
            - ``'parallel'``: uniform ``shift_usd`` applied to all.
            - ``'twist'``: linear interpolation between ``front_usd``
              at the shortest tenor and ``back_usd`` at the longest.
            - ``'custom'``: per-tenor values from
              ``bumps_by_tenor``, interpolated linearly for
              intermediate tenors.

        Args:
            scenario: Dictionary with a ``type`` key and associated
                parameters (see ``apply_scenario`` for the full
                specification).

        Returns:
            numpy.ndarray of float bump values with length equal to
            ``len(base_settlements)``.
        """
        n = len(self.base_settlements)
        tenors = np.array([
            s["time_to_expiry"] if isinstance(s, dict) else s.time_to_expiry
            for s in self.base_settlements
        ])
        bumps = np.zeros(n)
        stype = scenario["type"]

        if stype == "parallel":
            bumps[:] = scenario["shift_usd"]
        elif stype == "twist":
            front_usd = scenario["front_usd"]
            back_usd = scenario["back_usd"]
            t_min, t_max = tenors.min(), tenors.max()
            if t_max > t_min:
                for i, t in enumerate(tenors):
                    w = (t - t_min) / (t_max - t_min)
                    bumps[i] = front_usd * (1 - w) + back_usd * w
            else:
                bumps[:] = (front_usd + back_usd) / 2
        elif stype == "custom":
            bumps_by_tenor = scenario.get("bumps_by_tenor", {})
            if bumps_by_tenor:
                custom_tenors = sorted(bumps_by_tenor.keys())
                custom_bumps = [bumps_by_tenor[t] for t in custom_tenors]
                for i, t in enumerate(tenors):
                    bumps[i] = np.interp(t, custom_tenors, custom_bumps)
        return bumps
