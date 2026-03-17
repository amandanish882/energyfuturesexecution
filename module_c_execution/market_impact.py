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

SESSION_MINUTES = 330.0  # NYMEX 09:00-14:30 ET


class ImpactEstimate:
    """Result record from the Almgren-Chriss impact model.

    Attributes:
        product: Commodity ticker symbol.
        num_contracts: Total contracts in the order.
        participation_rate: Fraction of window volume consumed.
        temporary_bps: Temporary impact in basis points.
        permanent_bps: Permanent impact in basis points.
        total_cost_bps: Expected cost (permanent + temporary) in bps.
        timing_risk_bps: 1-sigma execution risk in bps (NOT added to cost).
        total_cost_usd: Expected cost in USD.
        optimal_horizon_min: Almgren-Chriss optimal execution horizon.
    """

    def __init__(self, product, num_contracts, participation_rate,
                 temporary_bps, permanent_bps, total_cost_bps,
                 timing_risk_bps, total_cost_usd, optimal_horizon_min):
        self.product = product
        self.num_contracts = num_contracts
        self.participation_rate = participation_rate
        self.temporary_bps = temporary_bps
        self.permanent_bps = permanent_bps
        self.total_cost_bps = total_cost_bps
        self.timing_risk_bps = timing_risk_bps
        self.total_cost_usd = total_cost_usd
        self.optimal_horizon_min = optimal_horizon_min


class AlmgrenChrissModel:
    """Almgren-Chriss optimal execution and market impact model.

    Separates market impact into:
    - Temporary impact: proportional to instantaneous trading rate
    - Permanent impact: γ * Q² / (2 * V_day) — quadratic in order size
    - Temporary impact: η * (Q / (V_day * T)) * σ — depends on execution rate
    - Timing risk: σ * sqrt(Q * T / V_day) — grows with horizon

    Total expected IS = Permanent + Temporary + Timing Risk.
    The optimal horizon T* minimises this total.

    Attributes:
        _eta: Temporary impact coefficient.
        _gamma: Permanent impact coefficient.
        _risk_aversion: Lambda in the A-C objective for optimal horizon.
    """

    def __init__(self, eta=0.1, gamma=0.01, risk_aversion=0.01):
        self._eta = eta
        self._gamma = gamma
        self._risk_aversion = risk_aversion

    def estimate_impact(self, product, num_contracts,
                        execution_horizon_min=60.0, price=None):
        """Estimate market impact cost for a single futures order.

        Uses the proper Almgren-Chriss formulas:
            Permanent = γ * Q² / (2 * V_day)
            Temporary = η * (Q / (V_day * T)) * σ
            Timing    = σ * sqrt(Q * T / V_day)

        Returns:
            ImpactEstimate with all three components. Total cost
            includes all three (permanent + temporary + timing risk).
        """
        spec = ENERGY_FUTURES.get(product, ENERGY_FUTURES["CL"])
        adv = spec["avg_daily_volume"]
        sigma = spec["daily_vol_pct"]
        cs = spec["contract_size"]

        # T in fraction of trading day
        T = execution_horizon_min / SESSION_MINUTES
        Q = num_contracts

        volume_in_window = adv * T
        participation = Q / volume_in_window if volume_in_window > 0 else 1.0
        participation = min(participation, 1.0)

        # Almgren-Chriss three-part cost decomposition (all in fractional units)
        permanent = self._gamma * Q**2 / (2 * max(adv, 1))
        temporary = self._eta * (Q / (max(adv, 1) * max(T, 1e-6))) * sigma
        timing_risk = sigma * np.sqrt(Q * max(T, 1e-6) / max(adv, 1))

        # Convert to bps
        permanent_bps = permanent * 1e4
        temporary_bps = temporary * 1e4
        timing_risk_bps = timing_risk * 1e4
        total_cost_bps = permanent_bps + temporary_bps + timing_risk_bps

        # USD conversion
        ref_price = price or 70.0
        notional = ref_price * Q * cs
        total_cost_usd = total_cost_bps / 1e4 * notional

        # Optimal horizon
        optimal_horizon = self.optimal_execution_horizon(product, num_contracts)

        return ImpactEstimate(
            product=product,
            num_contracts=num_contracts,
            participation_rate=round(participation, 4),
            temporary_bps=round(temporary_bps, 4),
            permanent_bps=round(permanent_bps, 4),
            total_cost_bps=round(total_cost_bps, 4),
            timing_risk_bps=round(timing_risk_bps, 4),
            total_cost_usd=round(total_cost_usd, 2),
            optimal_horizon_min=round(optimal_horizon, 1),
        )

    def optimal_execution_horizon(self, product, num_contracts):
        """Compute Almgren-Chriss optimal execution horizon.

        T* = (1/κ) * cosh⁻¹(1 + κ * Q / V_day)
        H* = T* × 330 minutes

        Returns:
            Optimal horizon in minutes, clipped to [5, 330].
        """
        spec = ENERGY_FUTURES.get(product, ENERGY_FUTURES["CL"])
        sigma = spec["daily_vol_pct"]
        adv = spec["avg_daily_volume"]

        if self._risk_aversion < 1e-15 or sigma < 1e-15:
            return 60.0
        kappa = np.sqrt(self._risk_aversion * sigma**2 / self._eta) if self._eta > 0 else 1.0
        t_star = np.arccosh(1 + kappa * num_contracts / max(adv, 1)) / kappa
        return float(np.clip(t_star * SESSION_MINUTES, 5.0, SESSION_MINUTES))

    def optimal_trajectory(self, product, num_contracts, n_slices=10,
                           kappa=1.5):
        """Compute sinh-shaped optimal execution trajectory.

        Args:
            kappa: Urgency parameter. Higher = more front-loaded.
                Passed directly (not derived from risk_aversion).

        Returns:
            Cumulative fraction array of length n_slices + 1,
            starting at 0.0 and ending at 1.0.
        """
        tau = np.linspace(0, 1, n_slices + 1)
        if abs(kappa) < 1e-6:
            return tau  # degenerates to TWAP
        remaining = np.sinh(kappa * (1.0 - tau)) / np.sinh(kappa)
        return 1.0 - remaining

    def compare_strategies(self, product, num_contracts,
                           horizons_min=None, price=None):
        """Compare impact cost and timing risk across execution horizons.

        Returns DataFrame with impact_bps, risk_bps (separate, not summed),
        and total_cost_bps (expected cost only).
        """
        if horizons_min is None:
            horizons_min = [5, 15, 30, 60, 120, 240]

        rows = []
        for h in horizons_min:
            est = self.estimate_impact(product, num_contracts, h, price=price)
            rows.append({
                "horizon_min": h,
                "participation_rate": est.participation_rate,
                "impact_bps": est.total_cost_bps,
                "risk_bps": est.timing_risk_bps,
                "impact_cost_usd": est.total_cost_usd,
                "slices": max(1, int(h)),
            })

        return pd.DataFrame(rows)

    def __repr__(self):
        return (f"AlmgrenChrissModel(eta={self._eta}, gamma={self._gamma}, "
                f"risk_aversion={self._risk_aversion})")
