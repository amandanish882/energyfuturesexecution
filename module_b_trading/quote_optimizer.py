"""Quote optimizer for commodity futures market-making.

Searches a discrete grid of candidate bid-ask spreads for each
incoming RFQ and selects the spread that maximises expected P&L,
incorporating hedge slippage, transaction cost, size-dependent
unwind cost, and quadratic delta inventory-risk penalty:

    E[PnL] = win_prob * (revenue - hedge_slippage - txn_cost
                         - unwind_cost - inventory_penalty)

Supports alpha-informed directional skew that tilts bid/ask
asymmetrically based on per-product composite alpha signals,
single-RFQ optimisation and batch processing, and post-trade
markout P&L decomposition (edge, carry, curve move, hedge cost).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from module_c_execution.market_impact import AlmgrenChrissModel


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

_TXN_FEES = {  # per-contract exchange + clearing fees in USD
    "CL": 1.50,
    "HO": 1.50,
    "RB": 1.50,
    "NG": 1.50,
}


class QuoteResult:
    """Data container holding the output of quote optimisation for one RFQ.

    Captures the optimal spread levels (potentially asymmetric from
    alpha skew), the resulting quote price, the estimated win
    probability, and the expected P&L so that the caller can act on
    or log the decision.

    Attributes:
        rfq_id: Unique identifier string for the RFQ.
        product: Product code string (e.g. ``'CL'``, ``'NG'``).
        direction: String ``'buy'`` or ``'sell'`` reflecting the
            client's requested direction.
        num_contracts: Number of contracts requested by the client.
        bid_spread: Half-spread applied to the bid side in USD per
            unit, adjusted by alpha skew.
        ask_spread: Half-spread applied to the ask side in USD per
            unit, adjusted by alpha skew.
        mid_price: Reference mid-market price in USD per unit.
        quote_price: Final quoted price in USD per unit after
            applying the directional half-spread.
        expected_pnl: Expected dollar P&L at the optimal spread,
            combining win probability and revenue less hedge costs
            (slippage + txn + unwind + inventory penalty).
        win_probability: Estimated probability (0.0-1.0) of winning
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

        E[PnL] = win_prob * (revenue - hedge_slippage - txn_cost
                             - unwind_cost - inventory_penalty)

    where hedge costs capture real execution friction (half-spread
    crossing + per-contract exchange fees + sqrt-size market impact)
    and inventory penalty is a quadratic function of net delta
    position (Avellaneda-Stoikov style risk aversion).

    Supports alpha-informed directional skew that tilts bid/ask
    asymmetrically: bullish alpha tightens the bid and widens the ask,
    encouraging the desk to accumulate long positions.

    Attributes:
        _win_model: Optional external win-probability model exposing
            a ``predict_proba(features)`` method.
        _grid_points: Number of candidate spreads in the grid search.
        _max_mult: Maximum spread multiple of the product's base spread.
        _alpha_skews: Dict mapping product codes to alpha signal
            values in [-3, +3]. Positive = bullish.
        _current_delta: Dict mapping product codes to current net
            delta in contracts.
        _risk_lambda: Quadratic inventory penalty coefficient in
            $/contract^2.
    """

    def __init__(self, win_model=None, grid_points=50, max_spread_multiple=3.0,
                 alpha_skews=None, current_delta=None, risk_lambda=5e-12,
                 impact_model=None):
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
                Defaults to 3.0.
            alpha_skews: Optional dict mapping product codes to
                composite alpha signal values in [-3, +3]. Positive
                values are bullish, causing bid tightening and ask
                widening. Defaults to empty dict.
            current_delta: Optional dict mapping product codes to the
                desk's current net delta in contracts. Used for the
                quadratic inventory penalty. Defaults to empty dict.
            risk_lambda: Risk aversion coefficient for the quadratic
                inventory penalty, in dollars per dollar-delta-squared.
                Defaults to 5e-12.
            impact_model: Optional AlmgrenChrissModel instance for
                computing hedge slippage via temporary + permanent
                market impact. If None, one is created with default
                parameters.
        """
        self._win_model = win_model
        self._grid_points = grid_points
        self._max_mult = max_spread_multiple
        self._alpha_skews = alpha_skews or {}
        self._current_delta = current_delta or {}
        self._risk_lambda = risk_lambda
        self._impact_model = impact_model or AlmgrenChrissModel()

    def optimize_quote(self, rfq, mid_price, volatility=0.25):
        """Find the expected-P&L-maximising quote for a single RFQ.

        Builds a grid of candidate spreads from the product's base
        spread up to ``max_spread_multiple`` times that value. For each
        candidate the expected P&L is evaluated including hedge costs
        (slippage + txn + market impact) and a quadratic inventory
        penalty. The spread with highest E[PnL] is selected, then
        alpha-informed directional skew is applied to produce
        asymmetric bid/ask levels.

        Args:
            rfq: Dictionary with RFQ attributes. Expected keys:
                ``product`` (str), ``direction`` (``'buy'`` or
                ``'sell'``), ``num_contracts`` (int), ``rfq_id``
                (str), and optionally ``spread_sensitivity`` (float),
                ``urgency`` (str), ``client_segment`` (str).
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

        # --- Cost model ---
        hedge_costs = self.estimate_costs(product, n_contracts, mid_price=mid_price)
        hedge_cost = hedge_costs["total"]

        # Ensure the grid extends beyond breakeven so the optimizer
        # can find profitable quotes even for large orders.
        notional_per_unit = n_contracts * contract_size
        breakeven_spread = hedge_cost / notional_per_unit if notional_per_unit > 0 else base_spread
        grid_upper = max(base_spread * self._max_mult, breakeven_spread * 2.0)
        spreads = np.linspace(base_spread, grid_upper, self._grid_points)
        best_epnl = -np.inf
        best_spread = base_spread

        # --- Quadratic inventory penalty (Avellaneda-Stoikov) ---
        # Convert contract inventory to dollar-delta so penalty is
        # comparable across products (e.g. 10 CL ≠ 10 NG in risk).
        # dollar_delta = num_contracts * contract_size * mid_price
        current_contracts = self._current_delta.get(product, 0.0)
        dollar_per_contract = contract_size * mid_price
        current_dollar_delta = current_contracts * dollar_per_contract
        if direction == "buy":
            new_dollar_delta = current_dollar_delta + n_contracts * dollar_per_contract
        else:
            new_dollar_delta = current_dollar_delta - n_contracts * dollar_per_contract
        inv_penalty = self._risk_lambda * (new_dollar_delta**2 - current_dollar_delta**2)

        # --- Alpha-informed directional skew ---
        skew = self._alpha_skews.get(product, 0.0)
        skew_adjustment = np.clip(skew, -1.0, 1.0) * base_spread

        # --- Grid search over candidate spreads (skew-adjusted) ---
        best_quote_price = mid_price
        best_bid_spread = base_spread / 2
        best_ask_spread = base_spread / 2
        for s in spreads:
            # Apply additive skew to get the actual bid/ask the client sees
            bid_half = s / 2 - skew_adjustment
            ask_half = s / 2 + skew_adjustment

            if direction == "buy":
                quote_price = mid_price - bid_half
                effective_spread = bid_half
            else:
                quote_price = mid_price + ask_half
                effective_spread = ask_half

            # Win prob based on the actual skew-adjusted spread
            win_prob = self._estimate_win_prob(rfq, effective_spread, mid_price, volatility)
            # Revenue is what the desk earns on the skew-adjusted quote
            revenue = effective_spread * n_contracts * contract_size
            epnl = win_prob * (revenue - hedge_cost - inv_penalty)

            if epnl > best_epnl:
                best_epnl = epnl
                best_spread = s
                best_quote_price = quote_price
                best_bid_spread = bid_half
                best_ask_spread = ask_half

        win_prob = self._estimate_win_prob(
            rfq, best_bid_spread if direction == "buy" else best_ask_spread,
            mid_price, volatility,
        )
        quote_price = best_quote_price
        bid_spread = best_bid_spread
        ask_spread = best_ask_spread
        edge = (bid_spread if direction == "buy" else ask_spread) * contract_size

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
            ``mid_price``, ``quote_price``, ``bid_spread``,
            ``ask_spread``, ``spread``, ``win_probability``,
            ``expected_pnl``, and ``edge_per_contract``.
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
                "bid_spread": result.bid_spread,
                "ask_spread": result.ask_spread,
                "spread": result.bid_spread if result.direction == "buy" else result.ask_spread,
                "win_probability": result.win_probability,
                "expected_pnl": result.expected_pnl,
                "edge_per_contract": result.edge_per_contract,
            })

            # Update running delta so subsequent quotes reflect
            # accumulated inventory risk (not each quote from zero)
            n = result.num_contracts
            if result.direction == "buy":
                self._current_delta[product] = self._current_delta.get(product, 0.0) + n
            else:
                self._current_delta[product] = self._current_delta.get(product, 0.0) - n

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
                "product": rfq.get("product", "CL"),
                "client_segment": rfq.get("client_segment", "hedge_fund"),
                "volatility": volatility,
            }])
            return float(self._win_model.predict_proba(features)[0])

        product = rfq.get("product", "CL")
        base = _BASE_SPREADS.get(product, 0.01)
        ratio = spread / base if base > 0 else 1.0
        sensitivity = rfq.get("spread_sensitivity", 0.5)
        base_prob = 0.8 - 0.15 * (ratio - 1) * sensitivity
        return float(np.clip(base_prob, 0.05, 0.95))

    def estimate_costs(self, product, num_contracts, mid_price=None):
        """Estimate execution costs via Almgren-Chriss market impact model.

        Uses the Almgren-Chriss temporary + permanent impact framework
        from module_c_execution to compute hedge slippage, plus a fixed
        per-contract transaction fee.

        Args:
            product: Product code string (e.g. ``'CL'``, ``'HO'``).
            num_contracts: Number of contracts to be executed.
            mid_price: Optional mid-market price for the product. If
                None, the impact model uses its internal default.

        Returns:
            Dictionary with keys:
                ``hedge_slippage``: USD cost from temporary + permanent
                    market impact (Almgren-Chriss).
                ``txn_cost``: USD per-contract exchange + clearing fees.
                ``total``: Sum of all cost components.
                ``per_contract``: Total cost divided by the number of
                    contracts (0 if ``num_contracts`` is zero).
                ``temporary_impact``: Fractional temporary impact.
                ``permanent_impact``: Fractional permanent impact.
        """
        # Hedge cost uses only permanent + temporary impact
        # (deterministic costs). Timing risk is excluded because it
        # averages to zero across many trades — charging clients for
        # uncertainty would make quotes uncompetitively wide.
        hedge_impact = self._impact_model.estimate_impact(
            product, num_contracts, price=mid_price,
        )
        ref_price = mid_price or 70.0
        cs = _CONTRACT_SIZES.get(product, 1000)
        notional = ref_price * num_contracts * cs
        deterministic_bps = hedge_impact.permanent_bps + hedge_impact.temporary_bps
        hedge_slippage = deterministic_bps / 1e4 * notional

        # Unwind: cost of closing the hedge later (longer horizon,
        # so lower temporary impact but same permanent)
        unwind_impact = self._impact_model.estimate_impact(
            product, num_contracts, execution_horizon_min=120.0,
            price=mid_price,
        )
        unwind_bps = unwind_impact.permanent_bps + unwind_impact.temporary_bps
        unwind_cost = unwind_bps / 1e4 * notional

        txn_cost = _TXN_FEES.get(product, 1.50) * num_contracts
        total = hedge_slippage + unwind_cost + txn_cost

        return {
            "hedge_slippage": hedge_slippage,
            "unwind_cost": unwind_cost,
            "txn_cost": txn_cost,
            "total": total,
            "per_contract": total / num_contracts if num_contracts > 0 else 0,
            "temporary_bps": hedge_impact.temporary_bps,
            "permanent_bps": hedge_impact.permanent_bps,
        }

    def decompose_markout(self, result, markout_mid=None, holding_days=1, carry_bps=0.0):
        """Decompose post-trade P&L into edge, carry, curve move, hedge cost.

        Used for post-trade analysis to confirm systematic edge over
        flat pricing. Each component isolates a distinct source of
        P&L so the desk can attribute performance.

        Args:
            result: QuoteResult from optimize_quote.
            markout_mid: Mid price at markout time. If None, assumes
                no price movement.
            holding_days: Days position is held before unwind.
                Defaults to 1.
            carry_bps: Daily carry (roll yield) in basis points of
                notional. Defaults to 0.0.

        Returns:
            Dict with keys:
                ``edge``: Gross spread earned (bid_spread + ask_spread)
                    times position size.
                ``carry``: Roll yield earned while holding.
                ``curve_move``: MTM P&L from mid price movement.
                ``hedge_cost``: Negative of total hedging cost.
                ``total_pnl``: Sum of all components.
        """
        cs = _CONTRACT_SIZES.get(result.product, 1000)
        n = result.num_contracts

        # Edge: gross spread captured
        edge = (result.bid_spread + result.ask_spread) * n * cs

        # Carry: roll yield while holding
        notional = result.mid_price * n * cs
        carry = notional * carry_bps / 10000 * holding_days

        # Curve move: MTM from price change
        if markout_mid is not None:
            price_move = markout_mid - result.mid_price
            if result.direction == "buy":
                curve_move = price_move * n * cs
            else:
                curve_move = -price_move * n * cs
        else:
            curve_move = 0.0

        # Hedge cost: total execution friction
        costs = self.estimate_costs(result.product, n)
        hedge_cost = costs["total"]

        return {
            "edge": edge,
            "carry": carry,
            "curve_move": curve_move,
            "hedge_cost": -hedge_cost,
            "total_pnl": edge + carry + curve_move - hedge_cost,
        }

    def __repr__(self):
        """Return a concise string representation of the optimizer.

        Returns:
            String of the form ``QuoteOptimizer(grid=N)`` where N is
            the number of grid points used in the spread search.
        """
        return f"QuoteOptimizer(grid={self._grid_points})"
