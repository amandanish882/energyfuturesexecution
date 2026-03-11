"""Tests for execution module: impact, scheduling, simulation."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
import numpy as np

from module_c_execution.market_impact import AlmgrenChrissModel, ENERGY_FUTURES
from module_c_execution.execution_scheduler import (
    TWAPScheduler,
    VWAPScheduler,
    AdaptiveScheduler,
)
from module_c_execution.order_simulator import OrderSimulator


class TestAlmgrenChriss:
    def test_estimate_impact(self):
        model = AlmgrenChrissModel()
        est = model.estimate_impact("CL", 100)
        assert est.total_cost_usd > 0
        assert est.cost_bps > 0
        assert est.participation_rate > 0

    def test_larger_order_more_impact(self):
        model = AlmgrenChrissModel()
        small = model.estimate_impact("CL", 10)
        large = model.estimate_impact("CL", 1000)
        assert large.total_cost_usd > small.total_cost_usd

    def test_optimal_trajectory(self):
        model = AlmgrenChrissModel()
        traj = model.optimal_trajectory("CL", 100, n_slices=10)
        assert len(traj) == 11
        assert abs(traj[0]) < 1e-6
        assert abs(traj[-1] - 1.0) < 0.01

    def test_compare_strategies(self):
        model = AlmgrenChrissModel()
        df = model.compare_strategies("CL", 100)
        assert len(df) > 0
        assert "total_cost_usd" in df.columns

    @pytest.mark.parametrize("product", ["CL", "HO", "RB", "NG"])
    def test_all_products(self, product):
        model = AlmgrenChrissModel()
        est = model.estimate_impact(product, 50)
        assert est.product == product
        assert est.total_cost_usd >= 0


class TestSchedulers:
    def test_twap_schedule(self):
        sched = TWAPScheduler()
        result = sched.schedule("CL", 100, n_slices=10)
        assert len(result.slices) == 10
        total = sum(s.target_contracts for s in result.slices)
        assert total == 100

    def test_vwap_schedule(self):
        sched = VWAPScheduler()
        result = sched.schedule("CL", 100)
        assert len(result.slices) > 0
        total = sum(s.target_contracts for s in result.slices)
        assert total == 100

    def test_adaptive_schedule(self):
        sched = AdaptiveScheduler(urgency=0.8)
        result = sched.schedule("CL", 100)
        assert len(result.slices) > 0

    def test_schedule_to_dataframe(self):
        sched = TWAPScheduler()
        result = sched.schedule("NG", 50, n_slices=5)
        df = result.to_dataframe()
        assert len(df) == 5
        assert "target_contracts" in df.columns


class TestOrderSimulator:
    def test_generate_book(self):
        sim = OrderSimulator()
        book = sim.generate_book("CL", 72.50)
        assert len(book.bids) > 0
        assert len(book.asks) > 0
        assert book.mid_price > 0
        assert book.spread >= 0

    def test_walk_book(self):
        sim = OrderSimulator()
        book = sim.generate_book("CL", 72.50, depth_contracts=100)
        fills = sim.walk_book(book, "buy", 10)
        assert len(fills) > 0
        total_filled = sum(f.size for f in fills)
        assert total_filled == 10

    def test_simulate_execution(self):
        sim = OrderSimulator()
        df = sim.simulate_execution("CL", "buy", 50, 72.50, n_slices=5)
        assert len(df) > 0
        assert "price" in df.columns
        assert "slippage" in df.columns
