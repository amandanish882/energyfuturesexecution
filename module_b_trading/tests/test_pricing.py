"""Tests for futures pricing and risk analytics."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
import numpy as np

from module_a_curves.curve_bootstrapper import ForwardCurve, ForwardCurveBootstrapper, FuturesSettlement
from module_b_trading.futures_pricer import FuturesPosition, FuturesPricer, CONTRACT_SPECS
from module_b_trading.risk_analytics import RiskAnalytics


@pytest.fixture
def curve():
    times = [i/12 for i in range(1, 13)]
    prices = [72.50 + 0.30 * i for i in range(12)]
    return ForwardCurve(times, prices)


@pytest.fixture
def pricer(curve):
    return FuturesPricer(curve)


@pytest.fixture
def long_position():
    return FuturesPosition(
        ticker="CLZ26", product="CL", num_contracts=10,
        direction="long", entry_price=72.50, expiry_date="2026-11-20",
    )


class TestFuturesPosition:
    def test_long_direction_sign(self, long_position):
        assert long_position.direction_sign == 1

    def test_short_direction_sign(self):
        pos = FuturesPosition("CLZ26", "CL", 5, "short", 72.0, "2026-11-20")
        assert pos.direction_sign == -1

    def test_contract_size_auto(self, long_position):
        assert long_position.contract_size == 1000  # CL

    def test_invalid_direction(self):
        with pytest.raises(ValueError):
            FuturesPosition("CLZ26", "CL", 5, "flat", 72.0)

    def test_zero_contracts(self):
        with pytest.raises(ValueError):
            FuturesPosition("CLZ26", "CL", 0, "long", 72.0)


class TestFuturesPricer:
    def test_mark_to_market(self, pricer, long_position):
        mtm = pricer.mark_to_market(long_position)
        assert isinstance(mtm, float)

    def test_calendar_spread(self, pricer):
        value = pricer.calendar_spread_value("CL", 1/12, 6/12)
        assert isinstance(value, float)

    def test_crack_spread(self, pricer):
        value = pricer.crack_spread_value(
            cl_price=72.50, ho_price=2.35, rb_price=2.45,
        )
        assert isinstance(value, float)

    def test_portfolio_mtm(self, pricer, long_position):
        short_pos = FuturesPosition("CLF27", "CL", 5, "short", 73.00, "2027-01-20")
        df = pricer.portfolio_mtm([long_position, short_pos])
        assert len(df) == 2
        assert "mtm_usd" in df.columns

    def test_forward_price(self, pricer):
        price = pricer.forward_price(0.5)
        assert price > 0

    def test_none_curve_raises(self):
        with pytest.raises(ValueError):
            FuturesPricer(None)


class TestContractSpecs:
    @pytest.mark.parametrize("product", ["CL", "HO", "RB", "NG"])
    def test_specs_exist(self, product):
        assert product in CONTRACT_SPECS
        assert CONTRACT_SPECS[product]["contract_size"] > 0
