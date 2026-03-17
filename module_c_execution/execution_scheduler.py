"""Execution scheduling algorithms for energy commodity futures.

This module provides TWAP, VWAP, and Adaptive execution schedulers that
decompose parent orders into child slices aligned to the NYMEX intraday
volume profile. Each scheduler returns a structured ExecutionSchedule
that can be converted to a DataFrame for downstream analysis.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# NYMEX intraday volume profile (30-min buckets, 09:00-14:30 ET)
_NYMEX_VOLUME_PROFILE = np.array([
    0.12, 0.11, 0.10, 0.09, 0.08, 0.07, 0.07, 0.08, 0.09, 0.10, 0.09
])
_NYMEX_VOLUME_PROFILE = _NYMEX_VOLUME_PROFILE / _NYMEX_VOLUME_PROFILE.sum()


class ExecutionSlice:
    """A single child order slice within a parent execution schedule."""

    def __init__(self, slice_id, time_label, target_fraction,
                 target_contracts, cumulative_fraction):
        self.slice_id = slice_id
        self.time_label = time_label
        self.target_fraction = target_fraction
        self.target_contracts = target_contracts
        self.cumulative_fraction = cumulative_fraction


class ExecutionSchedule:
    """Complete execution schedule decomposing a parent order into child slices."""

    def __init__(self, product, total_contracts, strategy,
                 slices=None, total_duration_min=0.0):
        self.product = product
        self.total_contracts = total_contracts
        self.strategy = strategy
        self.slices = slices if slices is not None else []
        self.total_duration_min = total_duration_min

    def to_dataframe(self):
        return pd.DataFrame([
            {
                "slice_id": s.slice_id,
                "time_label": s.time_label,
                "target_fraction": s.target_fraction,
                "target_contracts": s.target_contracts,
                "cumulative_fraction": s.cumulative_fraction,
            }
            for s in self.slices
        ])


class ExecutionStrategy:
    """Abstract base class for execution scheduling strategies."""

    def schedule(self, product, num_contracts, n_slices=10, duration_min=330.0):
        raise NotImplementedError


class TWAPScheduler(ExecutionStrategy):
    """Time-Weighted Average Price: equal slices over horizon."""

    def schedule(self, product, num_contracts, n_slices=10, duration_min=330.0):
        base = num_contracts // n_slices
        remainder = num_contracts % n_slices
        slices = []
        cum = 0

        for i in range(n_slices):
            qty = base + (1 if i < remainder else 0)
            cum += qty
            time_min = i * (duration_min / n_slices)
            hour = 9 + int(time_min // 60)
            minute = int(time_min % 60)

            slices.append(ExecutionSlice(
                slice_id=i,
                time_label=f"{hour:02d}:{minute:02d}",
                target_fraction=1.0 / n_slices,
                target_contracts=qty,
                cumulative_fraction=cum / num_contracts if num_contracts > 0 else 0,
            ))

        return ExecutionSchedule(
            product=product,
            total_contracts=num_contracts,
            strategy="TWAP",
            slices=slices,
            total_duration_min=duration_min,
        )


class VWAPScheduler(ExecutionStrategy):
    """Volume-Weighted Average Price: proportional to NYMEX volume profile."""

    def __init__(self, volume_profile=None):
        self._profile = volume_profile if volume_profile is not None else _NYMEX_VOLUME_PROFILE

    def schedule(self, product, num_contracts, n_slices=10, duration_min=330.0):
        # Interpolate volume profile to n_slices points within horizon
        profile_times = np.linspace(0, 330, len(self._profile))
        slice_times = np.linspace(0, duration_min, n_slices, endpoint=False)
        weights = np.interp(slice_times, profile_times, self._profile)
        weights = weights / weights.sum()

        raw = weights * num_contracts
        quantities = np.floor(raw).astype(int)
        remainder = num_contracts - quantities.sum()
        fracs = raw - quantities
        top_idx = np.argsort(-fracs)[:int(remainder)]
        quantities[top_idx] += 1

        slices = []
        cum = 0
        for i in range(n_slices):
            qty = int(quantities[i])
            cum += qty
            time_min = slice_times[i]
            hour = 9 + int(time_min // 60)
            minute = int(time_min % 60)

            slices.append(ExecutionSlice(
                slice_id=i,
                time_label=f"{hour:02d}:{minute:02d}",
                target_fraction=float(weights[i]),
                target_contracts=qty,
                cumulative_fraction=cum / num_contracts if num_contracts > 0 else 0,
            ))

        return ExecutionSchedule(
            product=product,
            total_contracts=num_contracts,
            strategy="VWAP",
            slices=slices,
            total_duration_min=duration_min,
        )


class AdaptiveScheduler(ExecutionStrategy):
    """Almgren-Chriss optimal trajectory: sinh-based urgency schedule.

    kappa controls urgency: higher = more front-loaded execution.
    kappa=0 degenerates to TWAP, kappa=1.5 is moderate, kappa=3+ is aggressive.
    """

    def __init__(self, kappa=1.5, n_slices=10):
        self.kappa = kappa
        self.n_slices = n_slices

    def _trajectory(self, n):
        """Compute cumulative execution trajectory [0, ..., 1.0]."""
        tau = np.linspace(0, 1, n + 1)
        if abs(self.kappa) < 1e-6:
            return tau
        remaining = np.sinh(self.kappa * (1 - tau)) / np.sinh(self.kappa)
        return 1.0 - remaining

    def schedule(self, product, num_contracts, n_slices=None, duration_min=None):
        if n_slices is None:
            n_slices = self.n_slices

        # Compute optimal horizon if not provided
        if duration_min is None:
            from module_c_execution.market_impact import AlmgrenChrissModel
            model = AlmgrenChrissModel()
            duration_min = model.optimal_execution_horizon(product, num_contracts)

        traj = self._trajectory(n_slices)
        quantities = np.diff(np.round(traj * num_contracts)).astype(int)
        diff = num_contracts - quantities.sum()
        if diff != 0:
            quantities[-1] += diff

        slices = []
        cum = 0
        minutes_per_slice = duration_min / n_slices

        for i in range(n_slices):
            qty = int(quantities[i])
            cum += qty
            time_min = i * minutes_per_slice
            hour = 9 + int(time_min // 60)
            minute = int(time_min % 60)

            slices.append(ExecutionSlice(
                slice_id=i,
                time_label=f"{hour:02d}:{minute:02d}",
                target_fraction=float(quantities[i]) / max(num_contracts, 1),
                target_contracts=qty,
                cumulative_fraction=cum / num_contracts if num_contracts > 0 else 0,
            ))

        return ExecutionSchedule(
            product=product,
            total_contracts=num_contracts,
            strategy=f"Adaptive(kappa={self.kappa:.1f})",
            slices=slices,
            total_duration_min=float(duration_min),
        )


def compare_strategies(product, num_contracts, kappa=1.5, duration_min=330.0):
    """Compare all three strategies side by side."""
    return {
        "TWAP": TWAPScheduler().schedule(product, num_contracts,
                                          duration_min=duration_min),
        "VWAP": VWAPScheduler().schedule(product, num_contracts,
                                          duration_min=duration_min),
        "Adaptive": AdaptiveScheduler(kappa=kappa).schedule(
            product, num_contracts, duration_min=duration_min),
    }
