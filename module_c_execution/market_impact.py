"""Almgren-Chriss market impact model for energy commodity futures.

This module implements the Almgren-Chriss optimal liquidation framework,
providing market impact estimation, optimal execution trajectory computation,
and cross-horizon strategy comparison for NYMEX energy futures contracts.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


ENERGY_FUTURES = {
    "CL": {
        "name": "WTI Crude Oil",
        "exchange": "NYMEX",
        "contract_size": 1000,
        "tick_size": 0.01,
        "tick_value": 10.0,
        "avg_daily_volume": 350000,
        "avg_daily_oi": 1800000,
        "daily_vol_pct": 0.022,
        "bid_ask_ticks": 1,
    },
    "HO": {
        "name": "Heating Oil",
        "exchange": "NYMEX",
        "contract_size": 42000,
        "tick_size": 0.0001,
        "tick_value": 4.20,
        "avg_daily_volume": 120000,
        "avg_daily_oi": 350000,
        "daily_vol_pct": 0.020,
        "bid_ask_ticks": 2,
    },
    "RB": {
        "name": "RBOB Gasoline",
        "exchange": "NYMEX",
        "contract_size": 42000,
        "tick_size": 0.0001,
        "tick_value": 4.20,
        "avg_daily_volume": 100000,
        "avg_daily_oi": 300000,
        "daily_vol_pct": 0.024,
        "bid_ask_ticks": 2,
    },
    "NG": {
        "name": "Natural Gas",
        "exchange": "NYMEX",
        "contract_size": 10000,
        "tick_size": 0.001,
        "tick_value": 10.0,
        "avg_daily_volume": 250000,
        "avg_daily_oi": 1200000,
        "daily_vol_pct": 0.035,
        "bid_ask_ticks": 1,
    },
}


class ImpactEstimate:
    """Immutable result record produced by the Almgren-Chriss impact model.

    Captures all cost components for a single order estimation run,
    including temporary and permanent impact expressed both in USD and
    basis points, together with the participation rate and number of
    execution slices recommended by the model.

    Attributes:
        product: Commodity ticker symbol (e.g. ``"CL"``, ``"NG"``).
        num_contracts: Total number of contracts in the order.
        participation_rate: Fraction of estimated window volume the order
            represents (0.0 – 1.0).
        temporary_impact: Temporary price impact as a fraction of price,
            caused by short-term liquidity consumption.
        permanent_impact: Permanent price impact as a fraction of price,
            reflecting lasting information or supply/demand shift.
        total_cost_usd: Aggregate execution cost in US dollars.
        cost_bps: Total cost expressed in basis points of notional value.
        slices: Recommended number of child slices for execution.
    """

    def __init__(self, product, num_contracts, participation_rate,
                 temporary_impact, permanent_impact, total_cost_usd,
                 cost_bps, slices):
        self.product = product
        self.num_contracts = num_contracts
        self.participation_rate = participation_rate
        self.temporary_impact = temporary_impact
        self.permanent_impact = permanent_impact
        self.total_cost_usd = total_cost_usd
        self.cost_bps = cost_bps
        self.slices = slices


class AlmgrenChrissModel:
    """Almgren-Chriss optimal execution and market impact model.

    Implements the continuous-time Almgren-Chriss (2001) framework for
    optimal liquidation of a large position in a futures market. Separates
    market impact into a temporary component (proportional to the
    instantaneous trading rate) and a permanent component (proportional to
    total volume traded), and derives the risk-adjusted optimal trajectory
    that minimises expected cost plus a variance penalty.

    Attributes:
        _eta: Temporary impact coefficient controlling the sensitivity of
            instantaneous impact to the per-slice trading rate.
        _lambda: Permanent impact coefficient controlling the sensitivity
            of long-run price shift to overall order size relative to ADV.
        _gamma: Power-law exponent applied to the participation rate when
            computing temporary impact.
        _risk_aversion: Risk-aversion parameter used in the optimal
            trajectory calculation; higher values front-load execution.
    """

    def __init__(self, eta=0.1, lambda_=0.05, gamma=0.5, risk_aversion=5000.0):
        """Initialise the Almgren-Chriss model with calibration parameters.

        Args:
            eta: Temporary impact coefficient. Controls how strongly
                the per-slice trading rate affects the instantaneous
                execution price. Defaults to 0.1.
            lambda_: Permanent impact coefficient. Controls the
                proportion of daily volatility that translates into
                a lasting price shift per unit of ADV traded.
                Defaults to 0.05.
            gamma: Power-law exponent applied to the normalised
                per-slice trading rate when computing temporary impact.
                Values below 1.0 produce concave (sub-linear) impact.
                Defaults to 0.5.
            risk_aversion: Quadratic risk-aversion coefficient used in
                the hyperbolic optimal trajectory formula. Higher values
                accelerate early execution. Defaults to 5000.0.
        """
        self._eta = eta
        self._lambda = lambda_
        self._gamma = gamma
        self._risk_aversion = risk_aversion

    def estimate_impact(self, product, num_contracts,
                        execution_horizon_min=60.0, price=None):
        """Estimate market impact cost for a single futures order.

        Computes temporary and permanent impact components using the
        Almgren-Chriss model calibrated to the product's average daily
        volume and daily volatility. Temporary impact is derived from the
        per-slice participation rate; permanent impact scales with the
        order's fraction of average daily volume.

        Args:
            product: Commodity ticker symbol (``"CL"``, ``"HO"``,
                ``"RB"``, or ``"NG"``). Falls back to ``"CL"`` specs
                if unrecognised.
            num_contracts: Total number of contracts to execute.
            execution_horizon_min: Total time available for execution,
                in minutes. Determines the number of slices and the
                volume available during the window. Defaults to 60.0.
            price: Reference mid-market price used to convert fractional
                impact into USD cost and basis points. If ``None``,
                defaults to 70.0.

        Returns:
            ImpactEstimate containing participation_rate,
            temporary_impact, permanent_impact, total_cost_usd,
            cost_bps, and slices fields.
        """
        spec = ENERGY_FUTURES.get(product, ENERGY_FUTURES["CL"])
        adv = spec["avg_daily_volume"]
        sigma = spec["daily_vol_pct"]
        cs = spec["contract_size"]

        session_minutes = 330.0
        volume_in_window = adv * (execution_horizon_min / session_minutes)
        participation = num_contracts / volume_in_window if volume_in_window > 0 else 1.0

        slices = max(1, int(execution_horizon_min))
        contracts_per_slice = num_contracts / slices
        slice_volume = adv / session_minutes

        if slice_volume > 0:
            temp_impact = self._eta * sigma * (contracts_per_slice / slice_volume) ** self._gamma
        else:
            temp_impact = 0.0

        perm_impact = self._lambda * sigma * (num_contracts / adv)

        ref_price = price or 70.0
        notional = ref_price * num_contracts * cs

        temp_cost = temp_impact * ref_price * num_contracts * cs
        perm_cost = perm_impact * ref_price * num_contracts * cs
        total_cost = temp_cost + perm_cost
        cost_bps = (total_cost / notional * 10000) if notional > 0 else 0.0

        return ImpactEstimate(
            product=product,
            num_contracts=num_contracts,
            participation_rate=participation,
            temporary_impact=temp_impact,
            permanent_impact=perm_impact,
            total_cost_usd=total_cost,
            cost_bps=cost_bps,
            slices=slices,
        )

    def optimal_trajectory(self, product, num_contracts, n_slices=10):
        """Compute the Almgren-Chriss optimal execution trajectory.

        Derives the risk-adjusted optimal liquidation schedule as a
        cumulative fraction of total order size over normalised time
        [0, 1]. The trajectory uses a hyperbolic sine formula
        parameterised by kappa, which balances market impact cost against
        execution risk. When kappa * T is negligible the schedule
        degenerates to a linear (TWAP) trajectory.

        Args:
            product: Commodity ticker symbol (``"CL"``, ``"HO"``,
                ``"RB"``, or ``"NG"``). Daily volatility is read from
                the product spec; falls back to ``"CL"`` if unrecognised.
            num_contracts: Total number of contracts to execute. Used
                contextually; the returned trajectory is expressed as
                fractions independent of this value.
            n_slices: Number of discrete time steps in the trajectory.
                Defaults to 10.

        Returns:
            NumPy array of length ``n_slices + 1`` containing the
            cumulative fraction of the order executed at each time
            step, starting at 0.0 and ending at 1.0.
        """
        spec = ENERGY_FUTURES.get(product, ENERGY_FUTURES["CL"])
        sigma = spec["daily_vol_pct"]

        kappa = np.sqrt(self._risk_aversion * sigma**2 / self._eta)
        tau = np.linspace(0, 1, n_slices + 1)

        kT = kappa * 1.0
        if abs(np.sinh(kT)) < 1e-10:
            trajectory = 1.0 - tau
        else:
            trajectory = np.sinh(kappa * (1.0 - tau)) / np.sinh(kT)

        cum_executed = 1.0 - trajectory
        return cum_executed

    def compare_strategies(self, product, num_contracts, horizons_min=None):
        """Compare market impact cost across multiple execution horizons.

        Calls ``estimate_impact`` for each horizon in ``horizons_min``
        and assembles the results into a single DataFrame for easy
        comparison and plotting. Longer horizons reduce participation
        rate and temporary impact at the cost of greater market risk.

        Args:
            product: Commodity ticker symbol (``"CL"``, ``"HO"``,
                ``"RB"``, or ``"NG"``).
            num_contracts: Total number of contracts to execute.
            horizons_min: List of execution horizons in minutes to
                evaluate. Defaults to ``[5, 15, 30, 60, 120, 240]``
                if ``None``.

        Returns:
            DataFrame with one row per horizon and columns:
            ``horizon_min``, ``participation_rate``,
            ``temporary_impact``, ``permanent_impact``,
            ``total_cost_usd``, ``cost_bps``, ``slices``.
        """
        if horizons_min is None:
            horizons_min = [5, 15, 30, 60, 120, 240]

        rows = []
        for h in horizons_min:
            est = self.estimate_impact(product, num_contracts, h)
            rows.append({
                "horizon_min": h,
                "participation_rate": est.participation_rate,
                "temporary_impact": est.temporary_impact,
                "permanent_impact": est.permanent_impact,
                "total_cost_usd": est.total_cost_usd,
                "cost_bps": est.cost_bps,
                "slices": est.slices,
            })

        return pd.DataFrame(rows)

    def __repr__(self):
        """Return an unambiguous string representation of the model instance.

        Returns:
            String showing the model class name together with the values
            of the three primary calibration parameters: ``eta``,
            ``lambda``, and ``gamma``.
        """
        return (f"AlmgrenChrissModel(eta={self._eta}, lambda={self._lambda}, "
                f"gamma={self._gamma})")
