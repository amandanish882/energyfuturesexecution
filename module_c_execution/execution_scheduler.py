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


NYMEX_VOLUME_PROFILE = {
    "09:00": 0.12,
    "09:30": 0.11,
    "10:00": 0.10,
    "10:30": 0.09,
    "11:00": 0.08,
    "11:30": 0.07,
    "12:00": 0.07,
    "12:30": 0.08,
    "13:00": 0.09,
    "13:30": 0.10,
    "14:00": 0.09,
}


class ExecutionSlice:
    """A single child order slice within a parent execution schedule.

    Represents one discrete unit of a parent order broken down by time
    period, holding both the absolute contract quantity and the fractional
    contribution to the total order for that slice.

    Attributes:
        slice_id: Zero-based integer index of this slice within its
            parent schedule.
        time_label: Human-readable time string (``"HH:MM"``) indicating
            when this slice should be submitted.
        target_fraction: Fraction of the total parent order allocated to
            this slice (0.0 – 1.0).
        target_contracts: Absolute number of contracts to trade during
            this slice, rounded to the nearest integer.
        cumulative_fraction: Running cumulative fraction of the parent
            order executed up to and including this slice (0.0 – 1.0).
    """

    def __init__(self, slice_id, time_label, target_fraction,
                 target_contracts, cumulative_fraction):
        self.slice_id = slice_id
        self.time_label = time_label
        self.target_fraction = target_fraction
        self.target_contracts = target_contracts
        self.cumulative_fraction = cumulative_fraction


class ExecutionSchedule:
    """Complete execution schedule decomposing a parent order into child slices.

    Aggregates all ExecutionSlice objects for a single parent order and
    provides metadata about the strategy used and the overall execution
    window. Supports DataFrame export for analysis and visualisation.

    Attributes:
        product: Commodity ticker symbol the schedule applies to.
        total_contracts: Total number of contracts in the parent order.
        strategy: Human-readable name of the scheduling algorithm used
            (e.g. ``"TWAP"``, ``"VWAP"``).
        slices: Ordered list of ExecutionSlice objects comprising the
            schedule.
        total_duration_min: Total elapsed time of the schedule in
            minutes from the first to the last slice submission.
    """

    def __init__(self, product, total_contracts, strategy,
                 slices=None, total_duration_min=0.0):
        self.product = product
        self.total_contracts = total_contracts
        self.strategy = strategy
        self.slices = slices if slices is not None else []
        self.total_duration_min = total_duration_min

    def to_dataframe(self):
        """Export the execution schedule to a pandas DataFrame.

        Converts each ExecutionSlice in the schedule to a row,
        preserving all slice attributes as named columns.

        Returns:
            DataFrame with one row per slice and columns:
            ``slice_id``, ``time_label``, ``target_fraction``,
            ``target_contracts``, ``cumulative_fraction``.
        """
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
    """Abstract base class for all execution scheduling strategies.

    Concrete subclasses must implement the ``schedule`` method, which
    decomposes a parent order into an ExecutionSchedule. The interface
    is deliberately minimal so that TWAP, VWAP, and Adaptive schedulers
    can be used interchangeably.
    """

    def schedule(self, product, num_contracts, n_slices=10):
        """Generate an execution schedule for a parent order.

        Subclasses must override this method to implement their specific
        slicing logic.

        Args:
            product: Commodity ticker symbol (e.g. ``"CL"``).
            num_contracts: Total number of contracts to execute.
            n_slices: Desired number of child slices. Concrete
                implementations may ignore or override this value
                (e.g. VWAP fixes slices to the volume profile length).

        Raises:
            NotImplementedError: Always raised by the base implementation;
                subclasses must provide their own override.
        """
        raise NotImplementedError


class TWAPScheduler(ExecutionStrategy):
    """Time-Weighted Average Price execution scheduler.

    Divides the parent order into equal-sized child slices spread
    uniformly across the NYMEX session (09:00 – 14:30, 330 minutes).
    Each slice receives an identical fraction of the total order,
    regardless of intraday volume patterns. Any rounding residual is
    added to the final slice to ensure full order completion.
    """

    def schedule(self, product, num_contracts, n_slices=10):
        """Split a parent order into equal time slices.

        Computes a uniform per-slice fraction, maps each slice to a
        clock time across the 330-minute NYMEX session, and adjusts
        the last slice for integer-rounding residuals.

        Args:
            product: Commodity ticker symbol (e.g. ``"CL"``).
            num_contracts: Total number of contracts to execute.
            n_slices: Number of equal time intervals to divide the
                session into. Defaults to 10.

        Returns:
            ExecutionSchedule with ``strategy="TWAP"`` and
            ``total_duration_min=330.0``, containing ``n_slices``
            ExecutionSlice objects with equal target fractions.
        """
        fraction = 1.0 / n_slices
        slices = []
        cum = 0.0

        for i in range(n_slices):
            cum += fraction
            contracts = int(round(num_contracts * fraction))
            time_min = i * (330.0 / n_slices)
            hour = 9 + int(time_min // 60)
            minute = int(time_min % 60)

            slices.append(ExecutionSlice(
                slice_id=i,
                time_label=f"{hour:02d}:{minute:02d}",
                target_fraction=fraction,
                target_contracts=contracts,
                cumulative_fraction=min(cum, 1.0),
            ))

        executed = sum(s.target_contracts for s in slices)
        if executed != num_contracts and slices:
            slices[-1].target_contracts += (num_contracts - executed)

        return ExecutionSchedule(
            product=product,
            total_contracts=num_contracts,
            strategy="TWAP",
            slices=slices,
            total_duration_min=330.0,
        )


class VWAPScheduler(ExecutionStrategy):
    """Volume-Weighted Average Price execution scheduler.

    Allocates each child slice a fraction of the parent order
    proportional to the expected intraday volume at that time interval,
    based on the NYMEX volume profile. Executing in line with volume
    reduces expected market impact relative to a flat TWAP schedule.

    Attributes:
        _profile: Dictionary mapping ``"HH:MM"`` time labels to their
            relative volume weights across the session.
    """

    def __init__(self, volume_profile=None):
        """Initialise the VWAP scheduler with an intraday volume profile.

        Args:
            volume_profile: Optional dictionary mapping ``"HH:MM"``
                time-period labels to relative volume weights. Weights
                do not need to sum to 1.0; they are normalised
                internally. Defaults to ``NYMEX_VOLUME_PROFILE`` if
                ``None``.
        """
        self._profile = volume_profile or NYMEX_VOLUME_PROFILE

    def schedule(self, product, num_contracts, n_slices=11):
        """Distribute a parent order according to the NYMEX intraday volume profile.

        Normalises the volume profile weights and allocates contracts
        to each time period proportionally. Integer rounding residuals
        are absorbed into the final slice.

        Args:
            product: Commodity ticker symbol (e.g. ``"CL"``).
            num_contracts: Total number of contracts to execute.
            n_slices: Ignored; the number of slices is determined by
                the length of the volume profile. Included for
                interface compatibility with the base class.

        Returns:
            ExecutionSchedule with ``strategy="VWAP"`` and
            ``total_duration_min=330.0``, containing one
            ExecutionSlice per entry in the volume profile.
        """
        times = list(self._profile.keys())
        weights = list(self._profile.values())
        total_w = sum(weights)
        fractions = [w / total_w for w in weights]

        slices = []
        cum = 0.0
        for i, (t, frac) in enumerate(zip(times, fractions)):
            cum += frac
            contracts = int(round(num_contracts * frac))
            slices.append(ExecutionSlice(
                slice_id=i,
                time_label=t,
                target_fraction=frac,
                target_contracts=contracts,
                cumulative_fraction=min(cum, 1.0),
            ))

        executed = sum(s.target_contracts for s in slices)
        if executed != num_contracts and slices:
            slices[-1].target_contracts += (num_contracts - executed)

        return ExecutionSchedule(
            product=product,
            total_contracts=num_contracts,
            strategy="VWAP",
            slices=slices,
            total_duration_min=330.0,
        )


class AdaptiveScheduler(ExecutionStrategy):
    """Urgency-adjusted execution scheduler built on top of VWAP.

    Modifies a baseline VWAP schedule by scaling each slice's contract
    quantity up or down according to an urgency score. High-urgency
    orders front-load execution by inflating early slice sizes; low-urgency
    orders back-load execution to reduce market impact. Any unfilled
    residual contracts are appended to the final slice.

    Attributes:
        _max_part: Maximum participation rate cap (currently stored for
            reference but not enforced per-slice in this implementation).
        _urgency: Scalar urgency score in [0.0, 1.0]. Values above 0.7
            accelerate execution; values below 0.3 slow it down.
        _vwap: Underlying VWAPScheduler used to produce the base schedule
            that is then adjusted by urgency scaling.
    """

    def __init__(self, max_participation=0.10, urgency=0.5):
        """Initialise the adaptive scheduler with participation and urgency settings.

        Args:
            max_participation: Maximum fraction of estimated period volume
                the scheduler is allowed to consume in a single slice.
                Stored for reference; defaults to 0.10 (10 %).
            urgency: Execution urgency score on a [0.0, 1.0] scale.
                Values above 0.7 inflate slice sizes to accelerate
                execution; values below 0.3 deflate them to reduce
                market impact. Defaults to 0.5 (neutral).
        """
        self._max_part = max_participation
        self._urgency = urgency
        self._vwap = VWAPScheduler()

    def schedule(self, product, num_contracts, n_slices=11):
        """Generate an urgency-adjusted execution schedule.

        Builds a baseline VWAP schedule and then scales each slice's
        contract quantity by a factor derived from ``_urgency``. Remaining
        contracts not allocated due to integer truncation or early
        depletion of ``remaining`` are added to the last slice.

        Args:
            product: Commodity ticker symbol (e.g. ``"CL"``).
            num_contracts: Total number of contracts to execute.
            n_slices: Passed through to the underlying VWAPScheduler;
                effectively determines the number of slices via the
                volume profile length. Defaults to 11.

        Returns:
            ExecutionSchedule with ``strategy`` set to
            ``"Adaptive(urgency=<value>)"`` and
            ``total_duration_min=330.0``, containing one
            ExecutionSlice per VWAP period with urgency-adjusted
            contract counts.
        """
        base = self._vwap.schedule(product, num_contracts, n_slices)

        adjusted_slices = []
        remaining = num_contracts
        cum = 0.0

        for s in base.slices:
            if self._urgency > 0.7:
                target = int(round(s.target_contracts * (1 + self._urgency)))
            elif self._urgency < 0.3:
                target = int(round(s.target_contracts * (1 - 0.3 * (1 - self._urgency))))
            else:
                target = s.target_contracts

            target = min(target, remaining)
            target = max(target, 0)
            remaining -= target
            cum += target / num_contracts if num_contracts > 0 else 0

            adjusted_slices.append(ExecutionSlice(
                slice_id=s.slice_id,
                time_label=s.time_label,
                target_fraction=target / num_contracts if num_contracts > 0 else 0,
                target_contracts=target,
                cumulative_fraction=min(cum, 1.0),
            ))

        if remaining > 0 and adjusted_slices:
            adjusted_slices[-1].target_contracts += remaining

        return ExecutionSchedule(
            product=product,
            total_contracts=num_contracts,
            strategy=f"Adaptive(urgency={self._urgency:.1f})",
            slices=adjusted_slices,
            total_duration_min=330.0,
        )
