"""Order execution simulator for energy commodity futures.

This module constructs synthetic Level 2 order books for NYMEX energy
futures and simulates realistic multi-slice order execution by walking
price levels, tracking slippage, and returning per-fill records as a
DataFrame.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_SPECS = {
    "CL": {"tick": 0.01, "tick_val": 10.0, "cs": 1000, "levels": 10},
    "HO": {"tick": 0.0001, "tick_val": 4.20, "cs": 42000, "levels": 8},
    "RB": {"tick": 0.0001, "tick_val": 4.20, "cs": 42000, "levels": 8},
    "NG": {"tick": 0.001, "tick_val": 10.0, "cs": 10000, "levels": 10},
}


class BookLevel:
    """A single price level in a Level 2 order book.

    Represents one resting limit order queue at a specific price,
    recording the aggregate quantity available at that price.

    Attributes:
        price: Price of this level in the native contract price units.
        size: Number of contracts available at this price level.
    """

    def __init__(self, price, size):
        self.price = price
        self.size = size


class L2Book:
    """Level 2 order book snapshot for a commodity futures contract.

    Stores bid and ask price levels with associated sizes, providing
    mid-price and spread calculations derived from the best bid and
    best ask. Levels are ordered best-to-worst (highest bid first,
    lowest ask first).

    Attributes:
        product: Commodity ticker symbol (e.g. ``"CL"``, ``"NG"``).
        bids: List of BookLevel objects on the buy side, ordered from
            highest to lowest price (best bid first).
        asks: List of BookLevel objects on the sell side, ordered from
            lowest to highest price (best ask first).
    """

    def __init__(self, product, bids=None, asks=None):
        self.product = product
        self.bids = bids if bids is not None else []
        self.asks = asks if asks is not None else []

    @property
    def mid_price(self):
        """Compute the mid-market price from the best bid and best ask.

        Returns:
            Average of the best bid price and best ask price as a float.
            Returns 0.0 if either side of the book is empty.
        """
        if self.bids and self.asks:
            return (self.bids[0].price + self.asks[0].price) / 2
        return 0.0

    @property
    def spread(self):
        """Compute the bid-ask spread from the best bid and best ask.

        Returns:
            Difference between the best ask price and the best bid price
            as a float. Returns 0.0 if either side of the book is empty.
        """
        if self.bids and self.asks:
            return self.asks[0].price - self.bids[0].price
        return 0.0


class Fill:
    """Record of a single partial or complete order fill.

    Captures the execution details for one price level consumed when
    walking the order book, including the signed slippage relative to
    the mid-price at the time of the fill.

    Attributes:
        price: Actual execution price in contract price units.
        size: Number of contracts filled at this price level.
        side: Order direction; ``"buy"`` or ``"sell"``.
        slippage: Signed price distance from mid-price. Positive for
            buys (paid above mid) and positive for sells (received
            below mid, i.e. mid minus fill price).
    """

    def __init__(self, price, size, side, slippage):
        self.price = price
        self.size = size
        self.side = side
        self.slippage = slippage


class OrderSimulator:
    """Simulates realistic order execution against synthetic Level 2 books.

    Generates randomised L2 order books for NYMEX energy futures using
    product-specific tick sizes and depth parameters, then walks those
    books to produce fill records with realistic price impact and
    slippage. Supports multi-slice execution with per-slice mid-price
    drift to mimic intraday price movement.

    Attributes:
        _rng: NumPy RandomState instance used for all stochastic
            components (book depth sizes and mid-price drift), seeded
            at construction for reproducibility.
    """

    def __init__(self, seed=42):
        """Initialise the order simulator with a fixed random seed.

        Args:
            seed: Integer seed passed to ``numpy.random.RandomState``
                to make all simulated book depths and price drifts
                fully reproducible. Defaults to 42.
        """
        self._rng = np.random.RandomState(seed)

    def generate_book(self, product, mid_price, depth_contracts=50):
        """Generate a synthetic Level 2 order book around a mid-price.

        Constructs bid and ask price levels using the product's tick size
        and configured number of depth levels. Level sizes are drawn from
        an exponential distribution whose mean decays with distance from
        the top of book, producing a realistic shape with thinner liquidity
        at outer levels.

        Args:
            product: Commodity ticker symbol (``"CL"``, ``"HO"``,
                ``"RB"``, or ``"NG"``). Falls back to ``"CL"`` specs
                if unrecognised.
            mid_price: Reference mid-market price around which bids and
                asks are centred.
            depth_contracts: Mean number of contracts at the best level
                of the exponential size distribution. Outer levels
                decay proportionally. Defaults to 50.

        Returns:
            L2Book containing bid and ask BookLevel lists, each of
            length equal to the product's configured depth (8 or 10
            levels), centred half a tick away from ``mid_price``.
        """
        spec = _SPECS.get(product, _SPECS["CL"])
        tick = spec["tick"]
        n_levels = spec["levels"]

        bids = []
        asks = []

        for i in range(n_levels):
            size = max(1, int(self._rng.exponential(depth_contracts / (1 + i * 0.3))))
            bid_price = round(mid_price - (i + 0.5) * tick, 6)
            bids.append(BookLevel(price=bid_price, size=size))

            size = max(1, int(self._rng.exponential(depth_contracts / (1 + i * 0.3))))
            ask_price = round(mid_price + (i + 0.5) * tick, 6)
            asks.append(BookLevel(price=ask_price, size=size))

        return L2Book(product=product, bids=bids, asks=asks)

    def walk_book(self, book, side, num_contracts):
        """Walk a Level 2 book to fill a given number of contracts.

        Consumes price levels from best to worst until the order is
        fully filled or the visible book is exhausted. If the visible
        depth is insufficient, a synthetic sweep fill is appended at
        twice the distance from mid to the worst visible level.

        Args:
            book: L2Book snapshot to execute against.
            side: Order direction; ``"buy"`` walks the ask side,
                ``"sell"`` walks the bid side.
            num_contracts: Number of contracts to fill.

        Returns:
            List of Fill objects, one per price level consumed plus
            an optional synthetic sweep fill if the book depth was
            insufficient to complete the order.
        """
        levels = book.asks if side == "buy" else book.bids
        mid = book.mid_price
        remaining = num_contracts
        fills = []

        for level in levels:
            if remaining <= 0:
                break
            fill_size = min(remaining, level.size)
            slippage = level.price - mid if side == "buy" else mid - level.price
            fills.append(Fill(
                price=level.price,
                size=fill_size,
                side=side,
                slippage=slippage,
            ))
            remaining -= fill_size

        if remaining > 0:
            worst = levels[-1].price if levels else mid
            extra_slip = abs(worst - mid) * 2
            sweep_price = mid + extra_slip if side == "buy" else mid - extra_slip
            fills.append(Fill(
                price=round(sweep_price, 6),
                size=remaining,
                side=side,
                slippage=extra_slip,
            ))

        return fills

    def simulate_execution(self, product, side, num_contracts,
                           mid_price, n_slices=5):
        """Simulate multi-slice execution of a futures order.

        Splits the parent order into child slices, generates a synthetic
        L2 order book for each slice with random mid-price drift, and
        walks the book to produce realistic fill records.

        Args:
            product: Commodity ticker symbol (``"CL"``, ``"HO"``,
                ``"RB"``, or ``"NG"``).
            side: Order direction; ``"buy"`` or ``"sell"``.
            num_contracts: Total number of contracts to execute.
            mid_price: Starting mid-market price used as the base for
                per-slice drift.
            n_slices: Number of child slices to split the order into.
                Each slice receives an equal share of the remaining
                contracts. Defaults to 5.

        Returns:
            DataFrame with one row per fill and columns:
            ``slice``, ``price``, ``size``, ``side``,
            ``slippage``, ``mid_at_entry``.
        """
        contracts_per_slice = max(1, num_contracts // n_slices)
        remaining = num_contracts
        all_fills = []

        for i in range(n_slices):
            if remaining <= 0:
                break

            slice_size = min(contracts_per_slice, remaining)
            drift = self._rng.normal(0, mid_price * 0.001)
            current_mid = mid_price + drift

            book = self.generate_book(product, current_mid)
            fills = self.walk_book(book, side, slice_size)

            for f in fills:
                all_fills.append({
                    "slice": i,
                    "price": f.price,
                    "size": f.size,
                    "side": f.side,
                    "slippage": f.slippage,
                    "mid_at_entry": current_mid,
                })

            remaining -= slice_size

        return pd.DataFrame(all_fills)

    def __repr__(self):
        """Return an unambiguous string representation of this simulator.

        Returns:
            The fixed string ``"OrderSimulator()"``.
        """
        return "OrderSimulator()"
