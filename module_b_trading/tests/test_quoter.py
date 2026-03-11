"""Tests for RFQ generation, win probability, and quote optimization."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest
import numpy as np
import pandas as pd

from module_b_trading.rfq_generator import RFQGenerator, PRODUCT_WEIGHTS, CLIENT_SEGMENTS
from module_b_trading.win_probability import WinProbabilityModel
from module_b_trading.quote_optimizer import QuoteOptimizer


class TestRFQGenerator:
    def test_generate_batch(self):
        gen = RFQGenerator(seed=42)
        df = gen.generate_batch(n=100)
        assert len(df) == 100
        assert "product" in df.columns
        assert "direction" in df.columns

    def test_products_in_range(self):
        gen = RFQGenerator(seed=42)
        df = gen.generate_batch(n=500)
        products = set(df["product"].unique())
        assert products.issubset({"CL", "HO", "RB", "NG"})

    def test_client_segments(self):
        gen = RFQGenerator(seed=42)
        df = gen.generate_batch(n=500)
        segments = set(df["client_segment"].unique())
        assert segments.issubset(set(CLIENT_SEGMENTS.keys()))

    def test_reproducible(self):
        df1 = RFQGenerator(seed=42).generate_batch(n=50)
        df2 = RFQGenerator(seed=42).generate_batch(n=50)
        assert df1["product"].tolist() == df2["product"].tolist()


class TestWinProbabilityModel:
    def test_predict_proba_shape(self):
        model = WinProbabilityModel()
        features = pd.DataFrame({
            "spread_bps": [5.0, 10.0],
            "num_contracts": [10, 50],
            "spread_sensitivity": [0.5, 0.8],
            "urgency": ["normal", "urgent"],
            "volatility": [0.25, 0.30],
            "product": ["CL", "NG"],
        })
        probs = model.predict_proba(features)
        assert len(probs) == 2
        assert all(0.0 <= p <= 1.0 for p in probs)

    def test_tighter_spread_higher_prob(self):
        model = WinProbabilityModel()
        tight = pd.DataFrame({
            "spread_bps": [1.0],
            "num_contracts": [10],
            "spread_sensitivity": [0.5],
            "urgency": ["normal"],
            "volatility": [0.25],
            "product": ["CL"],
        })
        wide = pd.DataFrame({
            "spread_bps": [20.0],
            "num_contracts": [10],
            "spread_sensitivity": [0.5],
            "urgency": ["normal"],
            "volatility": [0.25],
            "product": ["CL"],
        })
        p_tight = model.predict_proba(tight)[0]
        p_wide = model.predict_proba(wide)[0]
        assert p_tight > p_wide


class TestQuoteOptimizer:
    def test_optimize_single(self):
        opt = QuoteOptimizer()
        rfq = {
            "rfq_id": "RFQ-000001",
            "product": "CL",
            "direction": "buy",
            "num_contracts": 10,
            "spread_sensitivity": 0.5,
            "urgency": "normal",
        }
        result = opt.optimize_quote(rfq, mid_price=72.50)
        assert result.quote_price > 0
        assert result.win_probability > 0

    def test_estimate_costs(self):
        opt = QuoteOptimizer()
        costs = opt.estimate_costs("CL", 10)
        assert costs["total"] > 0
        assert costs["per_contract"] > 0

    def test_optimize_batch(self):
        gen = RFQGenerator(seed=42)
        rfqs = gen.generate_batch(n=20)
        opt = QuoteOptimizer()
        mids = {"CL": 72.50, "HO": 2.35, "RB": 2.45, "NG": 3.80}
        result = opt.optimize_batch(rfqs, mids)
        assert len(result) == 20
