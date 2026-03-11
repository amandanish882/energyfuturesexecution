"""Quote optimizer for commodity futures market-making.

Searches a discrete grid of candidate bid-ask spreads for each
incoming RFQ and selects the spread that maximises expected P&L,
defined as E[PnL] = win_prob * revenue - (1 - win_prob) * opp_cost.
Supports single-RFQ optimisation and batch processing, and can
delegate win-probability estimation to an external model or fall
back to a built-in parametric approximation.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd


_LIQUID_PRODUCTS = {"CL", "NG"}

_BASE_SPREADS = {
    "CL": 0.01,
    "HO": 0.0003,
    "RB": 0.0003,
    "NG": 0.002,
}

_CONTRACT_SIZES = {
    "CL": 1000,
    "HO": 42000,
    "RB": 42000,
    "NG": 10000,
}


class QuoteResult:
    """Data container holding the output of quote optimisation for one RFQ.

    Captures the optimal spread levels, the resulting quote price,
    the estimated win probability at that spread, and the expected
    P&L so that the caller can act on or log the decision.

    Attributes:
        rfq_id: Unique identifier string for the RFQ.
        product: Product code string (e.g. ``'CL'``, ``'NG'``).
        direction: String ``'buy'`` or ``'sell'`` reflecting the
            client's requested direction.
        num_contracts: Number of contracts requested by the client.
        bid_spread: Optimal half-spread applied to the bid side in
            USD per unit.
        ask_spread: Optimal half-spread applied to the ask side in
            USD per unit.
        mid_price: Reference mid-market price in USD per unit.
        quote_price: Final quoted price in USD per unit after
            applying the half-spread.
        expected_pnl: Expected dollar P&L at the optimal spread,
            combining win probability and revenue less opportunity
            cost.
        win_probability: Estimated probability (0.0–1.0) of winning
            the RFQ at the optimal spread.
        edge_per_contract: Gross dollar edge per contract at the
            optimal spread (spread * contract_size).
    """

    def __init__(self, rfq_id, product, direction, num_contracts,
                 bid_spread, ask_spread, mid_price, quote_price,
                 expected_pnl, win_probability, edge_per_contract):
        """Initialise a QuoteResult with all optimisation outputs.

        Args:
            rfq_id: Unique identifier string for the RFQ.
            product: Product code string (e.g. ``'CL'``).
            direction: ``'buy'`` or ``'sell'`` from the client's
                perspective.
            num_contracts: Number of contracts requested.
            bid_spread: Half-spread applied to the bid in USD/unit.
            ask_spread: Half-spread applied to the ask in USD/unit.
            mid_price: Reference mid-market price in USD/unit.
            quote_price: Final quoted price in USD/unit.
            expected_pnl: Expected P&L in USD at the optimal spread.
            win_probability: Float in [0.0, 1.0] probability of
                winning the RFQ at the optimal spread.
            edge_per_contract: Gross dollar edge per contract at the
                optimal spread.
        """
        self.rfq_id = rfq_id
        self.product = product
        self.direction = direction
        self.num_contracts = num_contracts
        self.bid_spread = bid_spread
        self.ask_spread = ask_spread
        self.mid_price = mid_price
        self.quote_price = quote_price
        self.expected_pnl = expected_pnl
        self.win_probability = win_probability
        self.edge_per_contract = edge_per_contract


class QuoteOptimizer:
    """Optimises quoted spreads for commodity futures RFQs to maximise expected P&L.

    Evaluates a linearly spaced grid of candidate spreads between the
    product's base spread and a multiple thereof. For each candidate,
    the expected P&L is computed as:
        E[PnL] = win_prob * revenue - (1 - win_prob) * opp_cost

    The spread that yields the highest E[PnL] is selected and used
    to construct the final bid and ask prices. Win probability is
    either supplied by an external model or estimated via a built-in
    parametric approximation based on spread-to-base-spread ratio.

    Attributes:
        _win_model: Optional external win-probability model exposing
            a ``predict_proba(features)`` method. When None, the
            internal approximation is used.
        _grid_points: Number of evenly spaced candidate spreads to
            evaluate during the grid search.
        _max_mult: Maximum spread multiple of the product's base
            spread that the grid search will consider.
    """

    def __init__(self, win_model=None, grid_points=50, max_spread_multiple=3.0):
        """Initialise the quote optimizer.

        Args:
            win_model: Optional external model with a
                ``predict_proba(features: DataFrame) -> np.ndarray``
                interface. If None, the built-in parametric
                approximation is used. Defaults to None.
            grid_points: Number of candidate spreads in the search
                grid. Higher values improve precision at the cost of
                computation. Defaults to 50.
            max_spread_multiple: Upper bound of the search grid
                expressed as a multiple of the product's base spread.
                Defaults to 5.0.
        """
        self._win_model = win_model
        self._grid_points = grid_points
        self._max_mult = max_spread_multiple

    def optimize_quote(self, rfq, mid_price, volatility=0.25):
        """Find the expected-P&L-maximising quote for a single RFQ.

        Builds a grid of candidate spreads from the product's base
        spread up to ``max_spread_multiple`` times that value,
        evaluates expected P&L at each point, and selects the optimal
        spread. Constructs bid and ask prices by applying a half-spread
        to the mid-market price.

        Args:
            rfq: Dictionary with RFQ attributes. Expected keys:
                ``product`` (str), ``direction`` (``'buy'`` or
                ``'sell'``), ``num_contracts`` (int), ``rfq_id``
                (str), and optionally ``spread_sensitivity`` (float)
                and ``urgency`` (str).
            mid_price: Current mid-market price in USD per unit.
            volatility: Implied volatility used in the win-probability
                estimate (decimal, e.g. 0.25 for 25 %). Defaults to
                0.25.

        Returns:
            QuoteResult instance containing the optimal spread, quote
            price, win probability, expected P&L, and edge per
            contract.
        """
        product = rfq.get("product", "CL")
        direction = rfq.get("direction", "buy")
        n_contracts = rfq.get("num_contracts", 1)
        base_spread = _BASE_SPREADS.get(product, 0.01)
        contract_size = _CONTRACT_SIZES.get(product, 1000)

        spreads = np.linspace(base_spread, base_spread * self._max_mult, self._grid_points)
        best_epnl = -np.inf
        best_spread = base_spread

        for s in spreads:
            win_prob = self._estimate_win_prob(rfq, s, mid_price, volatility)
            revenue = s * n_contracts * contract_size
            opp_cost = 0.1 * base_spread * n_contracts * contract_size
            epnl = win_prob * revenue - (1 - win_prob) * opp_cost

            if epnl > best_epnl:
                best_epnl = epnl
                best_spread = s

        win_prob = self._estimate_win_prob(rfq, best_spread, mid_price, volatility)

        if direction == "buy":
            quote_price = mid_price - best_spread / 2
            bid_spread = best_spread / 2
            ask_spread = best_spread / 2
        else:
            quote_price = mid_price + best_spread / 2
            bid_spread = best_spread / 2
            ask_spread = best_spread / 2

        edge = best_spread * contract_size

        return QuoteResult(
            rfq_id=rfq.get("rfq_id", ""),
            product=product,
            direction=direction,
            num_contracts=n_contracts,
            bid_spread=bid_spread,
            ask_spread=ask_spread,
            mid_price=mid_price,
            quote_price=quote_price,
            expected_pnl=best_epnl,
            win_probability=win_prob,
            edge_per_contract=edge,
        )

    def optimize_batch(self, rfqs, mid_prices, volatilities=None):
        """Optimise quotes for a batch of RFQs and return a summary DataFrame.

        Iterates over every row in ``rfqs``, looks up the appropriate
        mid-market price and implied volatility, calls
        ``optimize_quote`` for each RFQ, and collects the results.

        Args:
            rfqs: pandas.DataFrame where each row represents one RFQ.
                Expected columns match those consumed by
                ``optimize_quote`` (``product``, ``direction``,
                ``num_contracts``, ``rfq_id``, etc.).
            mid_prices: Dictionary mapping product code strings to
                their current mid-market prices in USD (e.g.
                ``{'CL': 75.0, 'NG': 2.50}``).
            volatilities: Optional dictionary mapping product code
                strings to implied volatilities as decimals. If None,
                defaults of ``{'CL': 0.30, 'HO': 0.25, 'RB': 0.28,
                'NG': 0.45}`` are used.

        Returns:
            pandas.DataFrame with one row per RFQ and columns:
            ``rfq_id``, ``product``, ``direction``, ``num_contracts``,
            ``mid_price``, ``quote_price``, ``spread``,
            ``win_probability``, ``expected_pnl``, and
            ``edge_per_contract``.
        """
        vols = volatilities or {"CL": 0.30, "HO": 0.25, "RB": 0.28, "NG": 0.45}
        results = []

        for _, row in rfqs.iterrows():
            product = row.get("product", "CL")
            mid = mid_prices.get(product, 70.0)
            vol = vols.get(product, 0.30)
            result = self.optimize_quote(row.to_dict(), mid, vol)
            results.append({
                "rfq_id": result.rfq_id,
                "product": result.product,
                "direction": result.direction,
                "num_contracts": result.num_contracts,
                "mid_price": result.mid_price,
                "quote_price": result.quote_price,
                "spread": result.bid_spread + result.ask_spread,
                "win_probability": result.win_probability,
                "expected_pnl": result.expected_pnl,
                "edge_per_contract": result.edge_per_contract,
            })

        return pd.DataFrame(results)

    def _estimate_win_prob(self, rfq, spread, mid_price, volatility):
        """Estimate the probability of winning the RFQ at a given spread.

        If an external win model is configured, constructs a feature
        DataFrame and calls its ``predict_proba`` method. Otherwise,
        uses a linear approximation: starting from a base win
        probability of 0.8, the probability declines with the ratio
        of the proposed spread to the product base spread, weighted
        by the client's spread sensitivity.

        Args:
            rfq: Dictionary of RFQ attributes (same as accepted by
                ``optimize_quote``).
            spread: Candidate bid-ask spread in USD per unit to
                evaluate.
            mid_price: Current mid-market price in USD per unit, used
                to convert the spread to basis points when calling an
                external model.
            volatility: Implied volatility as a decimal, forwarded
                to the external model when present.

        Returns:
            Float in [0.05, 0.95] representing the estimated
            probability of winning the RFQ at the given spread.
        """
        if self._win_model is not None:
            features = pd.DataFrame([{
                "spread_bps": spread / mid_price * 10000 if mid_price > 0 else 0,
                "num_contracts": rfq.get("num_contracts", 1),
                "spread_sensitivity": rfq.get("spread_sensitivity", 0.5),
                "urgency": rfq.get("urgency", "normal"),
                "volatility": volatility,
                "product": rfq.get("product", "CL"),
            }])
            return float(self._win_model.predict_proba(features)[0])

        product = rfq.get("product", "CL")
        base = _BASE_SPREADS.get(product, 0.01)
        ratio = spread / base if base > 0 else 1.0
        sensitivity = rfq.get("spread_sensitivity", 0.5)
        base_prob = 0.8 - 0.15 * (ratio - 1) * sensitivity
        return float(np.clip(base_prob, 0.05, 0.95))

    def estimate_costs(self, product, num_contracts):
        """Estimate execution costs for a given product and trade size.

        Computes two cost components: a half-spread crossing cost
        and a market impact cost that grows with the square root of
        the number of contracts. Less liquid products (those not in
        ``_LIQUID_PRODUCTS``) attract a 2x impact multiplier.

        Args:
            product: Product code string (e.g. ``'CL'``, ``'HO'``).
                Unknown products use a base spread of 0.01 and a
                contract size of 1000.
            num_contracts: Number of contracts to be executed.

        Returns:
            Dictionary with keys:
                ``half_spread``: USD cost of crossing the half-spread
                    for the full position.
                ``market_impact``: USD estimated market impact cost.
                ``total``: Sum of half-spread and market impact costs.
                ``per_contract``: Total cost divided by the number of
                    contracts (0 if ``num_contracts`` is zero).
        """
        base = _BASE_SPREADS.get(product, 0.01)
        cs = _CONTRACT_SIZES.get(product, 1000)
        is_liquid = product in _LIQUID_PRODUCTS
        impact_mult = 1.0 if is_liquid else 2.0

        half_spread_cost = base / 2 * num_contracts * cs
        impact_cost = base * impact_mult * np.sqrt(num_contracts) * cs
        total = half_spread_cost + impact_cost

        return {
            "half_spread": half_spread_cost,
            "market_impact": impact_cost,
            "total": total,
            "per_contract": total / num_contracts if num_contracts > 0 else 0,
        }

    def __repr__(self):
        """Return a concise string representation of the optimizer.

        Returns:
            String of the form ``QuoteOptimizer(grid=N)`` where N is
            the number of grid points used in the spread search.
        """
        return f"QuoteOptimizer(grid={self._grid_points})"
