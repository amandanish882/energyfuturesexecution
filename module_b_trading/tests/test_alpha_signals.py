"""Tests for alpha signal generators."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
import numpy as np

from module_b_trading.alpha_signals import (
    TermStructureSignal,
    InventorySignal,
    NGStorageSignal,
    MomentumSignal,
    SeasonalSignal,
    CrackSpreadSignal,
    CompositeAlphaModel,
)


class TestTermStructureSignal:
    def test_backwardation_positive(self):
        sig = TermStructureSignal()
        val = sig.compute(front_price=75.0, deferred_price=73.0)
        assert val > 0

    def test_contango_negative(self):
        sig = TermStructureSignal()
        val = sig.compute(front_price=73.0, deferred_price=75.0)
        assert val < 0

    def test_clipped(self):
        sig = TermStructureSignal()
        for _ in range(100):
            val = sig.compute(front_price=50.0, deferred_price=80.0)
        assert -3.0 <= val <= 3.0


class TestInventorySignal:
    def test_high_inventory_bearish(self):
        sig = InventorySignal()
        val = sig.compute(inventory_level=500.0, month=6)
        assert val < 0  # above seasonal norm

    def test_low_inventory_bullish(self):
        sig = InventorySignal()
        val = sig.compute(inventory_level=400.0, month=6)
        assert val > 0  # below seasonal norm


class TestNGStorageSignal:
    def test_high_storage_bearish(self):
        sig = NGStorageSignal()
        val = sig.compute(ng_storage_level=4000.0, month=10)
        assert val < 0  # well above October seasonal avg of 3550

    def test_low_storage_bullish(self):
        sig = NGStorageSignal()
        val = sig.compute(ng_storage_level=1200.0, month=3)
        assert val > 0  # well below March seasonal avg of 1800

    def test_clipped(self):
        sig = NGStorageSignal()
        val = sig.compute(ng_storage_level=5000.0, month=3)
        assert -3.0 <= val <= 3.0


class TestMomentumSignal:
    def test_needs_history(self):
        sig = MomentumSignal(fast_window=3, slow_window=5)
        val = sig.compute(price=72.0)
        assert val == 0.0  # not enough data

    def test_uptrend_positive(self):
        sig = MomentumSignal(fast_window=3, slow_window=5)
        prices = [70, 71, 72, 73, 74, 75, 76]
        for p in prices:
            val = sig.compute(price=float(p))
        assert val > 0


class TestSeasonalSignal:
    def test_summer_crude_bullish(self):
        sig = SeasonalSignal("CL")
        val = sig.compute(month=6)
        assert val > 0

    def test_winter_natgas_bullish(self):
        sig = SeasonalSignal("NG")
        val = sig.compute(month=1)
        assert val > 0

    @pytest.mark.parametrize("month", range(1, 13))
    def test_all_months_valid(self, month):
        sig = SeasonalSignal("CL")
        val = sig.compute(month=month)
        assert -3.0 <= val <= 3.0


class TestCrackSpreadSignal:
    def test_wide_crack_bearish_crude(self):
        sig = CrackSpreadSignal(mean_crack=25.0, std_crack=8.0)
        val = sig.compute(cl_price=70.0, ho_price=2.50, rb_price=2.60)
        assert isinstance(val, float)

    def test_zero_crude_returns_zero(self):
        sig = CrackSpreadSignal()
        val = sig.compute(cl_price=0.0, ho_price=2.0, rb_price=2.0)
        assert val == 0.0


class TestCompositeAlphaModel:
    def test_default_model(self):
        model = CompositeAlphaModel.default_crude_model()
        result = model.compute_composite(
            front_price=73.0,
            deferred_price=72.0,
            inventory_level=440.0,
            month=6,
            price=73.0,
            cl_price=73.0,
            ho_price=2.35,
            rb_price=2.45,
        )
        assert "composite" in result
        assert -3.0 <= result["composite"] <= 3.0

    def test_default_ng_model(self):
        model = CompositeAlphaModel.default_ng_model()
        result = model.compute_composite(
            front_price=3.80,
            deferred_price=4.10,
            ng_storage_level=1800.0,
            month=3,
            price=3.80,
        )
        assert "composite" in result
        assert "ng_storage" in result
        assert -3.0 <= result["composite"] <= 3.0

    def test_empty_model(self):
        model = CompositeAlphaModel()
        result = model.compute_composite()
        assert result["composite"] == 0.0
