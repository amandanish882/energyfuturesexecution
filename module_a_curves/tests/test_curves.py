"""Tests for forward curve bootstrapping and interpolation."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
import numpy as np

from module_a_curves.curve_bootstrapper import (
    ForwardCurve,
    ForwardCurveBootstrapper,
    FuturesSettlement,
)
from module_a_curves.interpolation import (
    LogLinearInterpolator,
    MonotoneConvexInterpolator,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_settlements():
    """Twelve-month CL strip."""
    return [
        FuturesSettlement("CL", "CLF26", 72.50, 1/12),
        FuturesSettlement("CL", "CLG26", 72.80, 2/12),
        FuturesSettlement("CL", "CLH26", 73.10, 3/12),
        FuturesSettlement("CL", "CLJ26", 73.35, 4/12),
        FuturesSettlement("CL", "CLK26", 73.55, 5/12),
        FuturesSettlement("CL", "CLM26", 73.70, 6/12),
        FuturesSettlement("CL", "CLN26", 73.80, 7/12),
        FuturesSettlement("CL", "CLQ26", 73.85, 8/12),
        FuturesSettlement("CL", "CLU26", 73.85, 9/12),
        FuturesSettlement("CL", "CLV26", 73.80, 10/12),
        FuturesSettlement("CL", "CLX26", 73.70, 11/12),
        FuturesSettlement("CL", "CLZ26", 73.55, 12/12),
    ]


@pytest.fixture
def forward_curve(sample_settlements):
    bootstrapper = ForwardCurveBootstrapper()
    return bootstrapper.bootstrap(sample_settlements)


# ---------------------------------------------------------------------------
# ForwardCurve tests
# ---------------------------------------------------------------------------
class TestForwardCurve:
    def test_construction(self, forward_curve):
        assert forward_curve is not None
        assert len(forward_curve.times) == 12

    def test_forward_price_at_node(self, forward_curve):
        price = forward_curve.forward_price(1/12)
        assert abs(price - 72.50) < 0.01

    def test_forward_price_interpolation(self, forward_curve):
        t_mid = 1.5 / 12  # between month 1 and 2
        price = forward_curve.forward_price(t_mid)
        assert 72.50 <= price <= 72.80

    def test_forward_price_extrapolation(self, forward_curve):
        price = forward_curve.forward_price(1.5)
        assert price > 0

    def test_convenience_yield(self, forward_curve):
        cy = forward_curve.convenience_yield(0.5)
        assert isinstance(cy, float)

    def test_roll_yield(self, forward_curve):
        ry = forward_curve.roll_yield(1/12, 2/12)
        assert isinstance(ry, float)

    def test_calendar_spread(self, forward_curve):
        spread = forward_curve.calendar_spread(1/12, 6/12)
        assert isinstance(spread, float)

    def test_is_contango(self, forward_curve):
        result = forward_curve.is_contango()
        assert result is True or result is False or isinstance(result, (bool, np.bool_))

    def test_shift(self, forward_curve):
        shifted = forward_curve.shift(1.0)
        original_price = forward_curve.forward_price(0.5)
        shifted_price = shifted.forward_price(0.5)
        assert abs(shifted_price - original_price - 1.0) < 0.01

    def test_empty_curve_raises(self):
        with pytest.raises((ValueError, IndexError)):
            ForwardCurve([], [])


class TestForwardCurveBootstrapper:
    def test_bootstrap(self, sample_settlements):
        bootstrapper = ForwardCurveBootstrapper()
        curve = bootstrapper.bootstrap(sample_settlements)
        assert curve is not None
        assert len(curve.times) == len(sample_settlements)

    def test_prices_positive(self, forward_curve):
        for t in forward_curve.times:
            assert forward_curve.forward_price(t) > 0


# ---------------------------------------------------------------------------
# Interpolation tests
# ---------------------------------------------------------------------------
class TestLogLinearInterpolator:
    def test_interpolate(self):
        xs = np.array([0.25, 0.5, 1.0, 2.0])
        ys = np.array([72.0, 73.0, 74.0, 75.0])
        interp = LogLinearInterpolator(xs, ys)
        val = interp(0.75)
        assert 73.0 < val < 74.0

    def test_at_node(self):
        xs = np.array([0.25, 0.5])
        ys = np.array([72.0, 73.0])
        interp = LogLinearInterpolator(xs, ys)
        assert abs(interp(0.25) - 72.0) < 1e-10
