"""Tests for the C++ kernel (commodities_cpp)."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
import numpy as np

import commodities_cpp as cpp


class TestCppForwardCurve:
    def test_construction(self):
        times = [0.25, 0.5, 1.0]
        prices = [72.0, 73.0, 74.0]
        curve = cpp.ForwardCurve(times, prices)
        assert curve.size() == 3

    def test_forward_price(self):
        curve = cpp.ForwardCurve([0.25, 0.5, 1.0], [72.0, 73.0, 74.0])
        price = curve.forward_price(0.5)
        assert abs(price - 73.0) < 0.01

    def test_convenience_yield(self):
        curve = cpp.ForwardCurve([0.25, 0.5, 1.0], [72.0, 73.0, 74.0])
        cy = curve.convenience_yield(0.5, 72.0)
        assert isinstance(cy, float)

    def test_roll_yield(self):
        curve = cpp.ForwardCurve([0.25, 0.5, 1.0], [74.0, 73.0, 72.0])
        ry = curve.roll_yield(0.25, 0.5)
        assert ry > 0  # backwardation

    def test_shift(self):
        curve = cpp.ForwardCurve([0.25, 0.5], [72.0, 73.0])
        shifted = curve.shift(1.0)
        assert abs(shifted.forward_price(0.25) - 73.0) < 0.01

    def test_is_contango(self):
        contango = cpp.ForwardCurve([0.25, 0.5], [72.0, 74.0])
        assert contango.is_contango()
        backw = cpp.ForwardCurve([0.25, 0.5], [74.0, 72.0])
        assert not backw.is_contango()


class TestCppFuturesPricer:
    def test_mark_to_market(self):
        curve = cpp.ForwardCurve([0.25, 0.5, 1.0], [73.0, 73.5, 74.0])
        pricer = cpp.FuturesPricer(curve)
        pos = cpp.FuturesPosition()
        pos.ticker = "CLZ26"
        pos.product = "CL"
        pos.num_contracts = 10
        pos.direction = 1
        pos.entry_price = 72.50
        pos.contract_size = 1000.0
        mtm = pricer.mark_to_market(pos, 0.25)
        assert mtm > 0  # price went up, long position

    def test_crack_spread(self):
        curve = cpp.ForwardCurve([0.25], [72.0])
        pricer = cpp.FuturesPricer(curve)
        val = pricer.crack_spread_321(72.0, 2.35, 2.45)
        assert isinstance(val, float)


class TestCppExecutionEngine:
    def test_optimal_trajectory(self):
        engine = cpp.ExecutionEngine()
        traj = engine.optimal_trajectory("CL", 100, 10)
        assert len(traj) == 11
        assert abs(traj[-1] - 1.0) < 0.01

    def test_temporary_impact(self):
        engine = cpp.ExecutionEngine()
        impact = engine.temporary_impact("CL", 100.0, 350000.0)
        assert impact > 0

    def test_permanent_impact(self):
        engine = cpp.ExecutionEngine()
        impact = engine.permanent_impact("CL", 500.0, 350000.0)
        assert impact > 0
