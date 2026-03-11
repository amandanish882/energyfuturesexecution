"""RFQ (Request for Quote) generator for energy commodity futures.

Generates synthetic RFQ flow that mimics real market-making activity
across multiple energy products (WTI crude, heating oil, RBOB
gasoline, natural gas) and five client segments (producer, refiner,
airline, hedge fund, utility). Randomised attributes include product,
direction, contract size, spread type, urgency, and intraday timing.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd


PRODUCT_WEIGHTS = {
    "CL": 0.45,
    "HO": 0.15,
    "RB": 0.20,
    "NG": 0.20,
}

MONTH_CODES = "FGHJKMNQUVXZ"

CONTRACT_MONTHS = {
    "CL": list(MONTH_CODES),
    "HO": list(MONTH_CODES),
    "RB": list(MONTH_CODES),
    "NG": list(MONTH_CODES),
}

STRIP_DEPTH = {
    "CL": 12,
    "HO": 6,
    "RB": 6,
    "NG": 12,
}

CLIENT_SEGMENTS = {
    "producer": {
        "weight": 0.30,
        "direction_bias": "short",
        "avg_size": 50,
        "products": ["CL", "NG"],
        "spread_sensitivity": 0.6,
    },
    "refiner": {
        "weight": 0.25,
        "direction_bias": "long",
        "avg_size": 30,
        "products": ["CL", "HO", "RB"],
        "spread_sensitivity": 0.5,
    },
    "airline": {
        "weight": 0.10,
        "direction_bias": "long",
        "avg_size": 20,
        "products": ["CL", "HO"],
        "spread_sensitivity": 0.4,
    },
    "hedge_fund": {
        "weight": 0.20,
        "direction_bias": "neutral",
        "avg_size": 100,
        "products": ["CL", "HO", "RB", "NG"],
        "spread_sensitivity": 0.9,
    },
    "utility": {
        "weight": 0.15,
        "direction_bias": "long",
        "avg_size": 40,
        "products": ["NG"],
        "spread_sensitivity": 0.3,
    },
}


class RFQGenerator:
    """Generates synthetic RFQ flow for commodity futures market-making simulation.

    Samples client segments according to their configured probability
    weights, then draws product, direction, contract size, spread
    type, urgency, and intraday timestamp from per-segment
    distributions. Produces output in a structured DataFrame format
    that is directly consumable by QuoteOptimizer and MarkoutAnalyzer.

    Attributes:
        _rng: numpy RandomState instance used for all random draws,
            seeded for reproducibility.
        _rfqs_per_day: Default number of RFQs to generate per
            ``generate_batch`` call when no explicit count is given.
        _counter: Running integer counter used to create unique RFQ
            identifier strings.
    """

    def __init__(self, seed=42, rfqs_per_day=50):
        """Initialise the RFQ generator.

        Args:
            seed: Integer random seed for reproducible output.
                Defaults to 42.
            rfqs_per_day: Default number of RFQs produced by a single
                call to ``generate_batch`` when ``n`` is not supplied.
                Defaults to 50.
        """
        self._rng = np.random.RandomState(seed)
        self._rfqs_per_day = rfqs_per_day
        self._counter = 0

    def generate_batch(self, n=None, base_year=2026):
        """Generate a batch of synthetic RFQs and return them as a DataFrame.

        Produces ``n`` RFQ records by repeatedly sampling client
        segment, product, direction, size, urgency, and timestamp
        from the configured distributions. Approximately 15 % of RFQs
        are marked as calendar spreads, receiving a second leg ticker.

        Args:
            n: Number of RFQs to generate. If None, ``_rfqs_per_day``
                is used. Defaults to None.
            base_year: Four-digit integer year used when constructing
                ticker strings (e.g. 2026 produces tickers like
                ``'CLZ26'``). Defaults to 2026.

        Returns:
            pandas.DataFrame with one row per RFQ and columns:
            ``rfq_id``, ``timestamp``, ``client_segment``,
            ``product``, ``ticker``, ``direction``,
            ``num_contracts``, ``is_spread``, ``spread_leg2``,
            ``urgency``, and ``spread_sensitivity``. The
            ``product`` and ``client_segment`` columns are cast to
            ``str`` dtype.
        """
        n = n or self._rfqs_per_day
        rows = []

        for _ in range(n):
            self._counter += 1
            segment = self._pick_segment()
            seg_cfg = CLIENT_SEGMENTS[segment]
            product = self._pick_product(seg_cfg)
            ticker = self._make_ticker(product, base_year)
            direction = self._pick_direction(seg_cfg)
            size = max(1, int(self._rng.exponential(seg_cfg["avg_size"])))
            is_spread = self._rng.random() < 0.15
            spread_leg2 = ""
            if is_spread:
                spread_leg2 = self._make_ticker(product, base_year)

            urgency_draw = self._rng.random()
            urgency = "urgent" if urgency_draw < 0.1 else ("patient" if urgency_draw > 0.7 else "normal")

            hour = int(self._rng.normal(11.5, 1.5))
            hour = max(9, min(14, hour))
            minute = self._rng.randint(0, 60)

            rows.append({
                "rfq_id": f"RFQ-{self._counter:06d}",
                "timestamp": f"{base_year}-03-10 {hour:02d}:{minute:02d}:00",
                "client_segment": segment,
                "product": product,
                "ticker": ticker,
                "direction": direction,
                "num_contracts": size,
                "is_spread": is_spread,
                "spread_leg2": spread_leg2,
                "urgency": urgency,
                "spread_sensitivity": seg_cfg["spread_sensitivity"],
            })

        df = pd.DataFrame(rows)
        df["product"] = df["product"].astype(str)
        df["client_segment"] = df["client_segment"].astype(str)
        return df

    def _pick_segment(self):
        """Randomly select a client segment according to configured weights.

        Returns:
            String segment name (e.g. ``'producer'``, ``'hedge_fund'``)
            drawn from CLIENT_SEGMENTS using each segment's
            ``weight`` field as a probability.
        """
        segments = list(CLIENT_SEGMENTS.keys())
        weights = [CLIENT_SEGMENTS[s]["weight"] for s in segments]
        return str(self._rng.choice(segments, p=weights))

    def _pick_product(self, seg_cfg):
        """Randomly select a product available to a client segment.

        Normalises the global PRODUCT_WEIGHTS to the subset of
        products accessible to the given segment before sampling.

        Args:
            seg_cfg: Segment configuration dictionary from
                CLIENT_SEGMENTS, expected to contain a ``products``
                key listing available product codes.

        Returns:
            String product code (e.g. ``'CL'``, ``'NG'``) drawn
            from the segment's available products proportionally to
            their PRODUCT_WEIGHTS.
        """
        products = seg_cfg["products"]
        available_weights = np.array([PRODUCT_WEIGHTS.get(p, 0.1) for p in products])
        available_weights /= available_weights.sum()
        return str(self._rng.choice(products, p=available_weights))

    def _pick_direction(self, seg_cfg):
        """Randomly select a trade direction based on the segment's directional bias.

        Segments with a ``'long'`` bias buy 70 % of the time;
        ``'short'`` biased segments sell 70 % of the time; neutral
        segments trade 50/50.

        Args:
            seg_cfg: Segment configuration dictionary from
                CLIENT_SEGMENTS, expected to contain a
                ``direction_bias`` key with value ``'long'``,
                ``'short'``, or ``'neutral'``.

        Returns:
            String ``'buy'`` or ``'sell'``.
        """
        bias = seg_cfg["direction_bias"]
        if bias == "neutral":
            return "buy" if self._rng.random() < 0.5 else "sell"
        elif bias == "long":
            return "buy" if self._rng.random() < 0.7 else "sell"
        else:
            return "sell" if self._rng.random() < 0.7 else "buy"

    def _make_ticker(self, product, base_year):
        """Construct a CME-style futures ticker string for a product and year.

        Randomly selects a month index within the product's strip
        depth, looks up the corresponding CME month code, and appends
        the two-digit year suffix.

        Args:
            product: Product code string (e.g. ``'CL'``). Used to
                look up the strip depth from STRIP_DEPTH.
            base_year: Four-digit integer base year. The year suffix
                advances by one for each full 12-month offset.

        Returns:
            String ticker such as ``'CLZ26'`` or ``'NGF27'``.
        """
        depth = STRIP_DEPTH.get(product, 6)
        month_idx = self._rng.randint(0, min(depth, 12))
        month_code = MONTH_CODES[month_idx]
        year_suffix = str(base_year + month_idx // 12)[-2:]
        return f"{product}{month_code}{year_suffix}"

    def __repr__(self):
        """Return a concise string representation of the generator.

        Returns:
            String of the form ``RFQGenerator(rfqs_per_day=N)``
            showing the default daily RFQ volume.
        """
        return f"RFQGenerator(rfqs_per_day={self._rfqs_per_day})"
