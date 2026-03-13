"""Forward curve bootstrapper for commodity futures.

Provides classes and helpers for constructing commodity forward curves
from futures settlement prices. Supports log-linear and monotone-convex
interpolation methods and wraps a C++ cubic-spline kernel for performance.
"""

import numpy as np
import pandas as pd

from .interpolation import LogLinearInterpolator, MonotoneConvexInterpolator

import sys as _sys, pathlib as _pl
_cpp_dir = str(_pl.Path(__file__).resolve().parent.parent / "shared" / "cpp_kernel")
if _cpp_dir not in _sys.path:
    _sys.path.insert(0, _cpp_dir)
import commodities_cpp as _cpp


class FuturesSettlement:
    """Represents a single futures contract settlement record.

    Encapsulates all market data fields for one contract expiry, used
    as the primary input to ForwardCurveBootstrapper. Instances can be
    passed directly or as plain dicts throughout the bootstrapping pipeline.

    Attributes:
        product: Commodity ticker symbol (e.g. "CL", "HO", "RB", "NG").
        contract_code: Exchange contract code (e.g. "CLZ26").
        settlement: End-of-day settlement price in dollars per unit.
        time_to_expiry: Time to contract expiry expressed as a year fraction.
        volume: Daily traded volume for the contract. Defaults to 0.
        open_interest: Open interest count for the contract. Defaults to 0.
    """

    def __init__(self, product, contract_code, settlement, time_to_expiry,
                 volume=0, open_interest=0):
        self.product = product
        self.contract_code = contract_code
        self.settlement = settlement
        self.time_to_expiry = time_to_expiry
        self.volume = volume
        self.open_interest = open_interest


class ForwardCurve:
    """Represents a bootstrapped commodity forward price curve.

    Stores tenor-price node pairs and constructs an interpolator over
    pseudo discount factors (F(t)/F(0)) so that forward prices can be
    evaluated continuously. Provides convenience yield, roll yield, and
    calendar spread analytics as well as parallel-shift scenarios.

    Attributes:
        times: Array of year-fraction tenor points corresponding to contract
            expiries.
        forward_prices: Array of forward prices (in dollars per unit) at each
            tenor node.
        valuation_date: ISO-format date string used as the curve anchor.
        interpolation_method: Name of the interpolation scheme in use;
            either "log_linear" or "monotone_convex".
        product: Commodity ticker symbol (e.g. "CL", "HO").
        spot_price: Spot price anchor used for convenience-yield calculations;
            defaults to the first forward price when not supplied explicitly.
    """

    def __init__(self, times, forward_prices, valuation_date="2024-12-31",
                 interpolation_method="log_linear", product="CL",
                 spot_price=None):
        self.times = np.asarray(times, dtype=float)
        self.forward_prices = np.asarray(forward_prices, dtype=float)
        self.valuation_date = valuation_date
        self.interpolation_method = interpolation_method
        self.product = product
        self.spot_price = spot_price if spot_price is not None else float(forward_prices[0])

        # Build interpolator on pseudo discount factors F(t)/S
        pseudo_dfs = self.forward_prices / self.spot_price

        if interpolation_method == "monotone_convex":
            self._interp = MonotoneConvexInterpolator(self.times, pseudo_dfs)
        else:
            self._interp = LogLinearInterpolator(self.times, pseudo_dfs)

        self._base_price = self.spot_price

    def forward_price(self, t):
        """Return the interpolated forward price at a given tenor.

        For tenors at or before zero the spot price is returned directly.
        For positive tenors the price is recovered by scaling the base price
        by the interpolated pseudo discount factor.

        Args:
            t: Tenor expressed as a year fraction. Values <= 0 return the
                spot price.

        Returns:
            Forward price in dollars per unit as a float.
        """
        if t <= 0:
            return self.spot_price
        return self._base_price * self._interp(t)

    def forward_price_array(self, t_array):
        """Return forward prices evaluated at an array of tenors.

        Vectorised wrapper around forward_price that iterates over each
        element and collects results into a NumPy array.

        Args:
            t_array: Iterable of year-fraction tenors at which to evaluate
                the forward curve.

        Returns:
            NumPy array of forward prices corresponding to each input tenor.
        """
        return np.array([self.forward_price(t) for t in t_array])

    def convenience_yield(self, t, risk_free_rate=0.045, storage_cost=0.0):
        """Compute the implied convenience yield at a given tenor.

        Applies the cost-of-carry identity rearranged for the convenience
        yield: y = r + u - (1/T) * ln(F(T) / S). Returns zero when either
        the forward price or spot price is non-positive, or when t <= 0.

        Args:
            t: Tenor expressed as a year fraction. Values <= 0 return 0.0.
            risk_free_rate: Continuously compounded risk-free interest rate.
                Defaults to 0.045 (4.5%).
            storage_cost: Continuously compounded storage cost rate u.
                Defaults to 0.0.

        Returns:
            Implied convenience yield as a float. Returns 0.0 for degenerate
            inputs (non-positive prices or zero tenor).
        """
        if t <= 0:
            return 0.0
        F_t = self.forward_price(t)
        if F_t <= 0 or self.spot_price <= 0:
            return 0.0
        return risk_free_rate + storage_cost - (1.0 / t) * np.log(F_t / self.spot_price)

    def roll_yield(self, t1, t2):
        """Compute the annualized roll yield between two tenor points.

        Measures the return earned by rolling a long position from the
        far tenor t2 to the near tenor t1, expressed as an annualized
        rate. Returns 0.0 when the near forward price is non-positive or
        when the two tenors are effectively equal.

        Args:
            t1: Near tenor expressed as a year fraction.
            t2: Far tenor expressed as a year fraction. Must be greater than
                t1 by at least 1e-10 to avoid division by zero.

        Returns:
            Annualized roll yield as a float. Positive values indicate
            backwardation (near contracts are priced higher than far).
        """
        F1 = self.forward_price(t1)
        F2 = self.forward_price(t2)
        if F1 <= 0 or abs(t2 - t1) < 1e-10:
            return 0.0
        return -(F2 - F1) / F1 * (365.25 / ((t2 - t1) * 365.25))

    def calendar_spread(self, t1, t2):
        """Compute the calendar spread between two tenor points.

        Returns the difference F(t2) - F(t1). A positive value indicates
        contango (far contracts priced above near), while a negative value
        indicates backwardation.

        Args:
            t1: Near tenor expressed as a year fraction.
            t2: Far tenor expressed as a year fraction.

        Returns:
            Calendar spread in dollars per unit as a float.
        """
        return self.forward_price(t2) - self.forward_price(t1)

    def is_contango(self):
        """Determine whether the curve is in contango at the front end.

        Compares the second node price to the first node price. Returns
        False when fewer than two nodes are present.

        Returns:
            True if the second forward price exceeds the first (contango),
            False otherwise (backwardation or insufficient data).
        """
        if len(self.forward_prices) < 2:
            return False
        return bool(self.forward_prices[1] > self.forward_prices[0])

    def is_backwardation(self):
        """Determine whether the curve is in backwardation at the front end.

        Delegates to is_contango and returns the logical negation. A curve
        is in backwardation when near-dated contracts are priced above
        far-dated contracts.

        Returns:
            True if the curve is in backwardation, False if it is in contango.
        """
        return not self.is_contango()

    def shift(self, shift_usd):
        """Return a new ForwardCurve with all prices shifted by a fixed amount.

        Adds shift_usd to every forward price node and to the spot price,
        preserving all other curve attributes. Useful for parallel-shift
        scenario analysis and PnL attribution.

        Args:
            shift_usd: Dollar-per-unit shift to apply uniformly across all
                tenor nodes and the spot price. Can be negative.

        Returns:
            A new ForwardCurve instance with shifted prices. The original
            curve is not modified.
        """
        new_prices = self.forward_prices + shift_usd
        return ForwardCurve(
            self.times, new_prices, self.valuation_date,
            self.interpolation_method, self.product,
            self.spot_price + shift_usd,
        )

    def instantaneous_forward(self, t, dt=1 / 365):
        """Compute the instantaneous forward rate at a given tenor.

        Delegates to the underlying interpolator's forward method, which
        approximates the instantaneous forward using a finite difference
        over a small time step dt.

        Args:
            t: Tenor expressed as a year fraction at which to evaluate the
                instantaneous forward rate.
            dt: Finite-difference step size in years. Defaults to 1/365
                (one calendar day).

        Returns:
            Instantaneous forward rate as a float.
        """
        return self._interp.forward(t, dt)

    def __repr__(self):
        """Return a concise string representation of the curve.

        Summarises the key curve properties — product, valuation date,
        number of tenor nodes, maximum tenor, spot price, curve shape,
        and interpolation method — in a single readable line.

        Returns:
            A formatted string identifying the curve and its main attributes.
        """
        shape = "contango" if self.is_contango() else "backwardation"
        return (f"ForwardCurve(product={self.product}, val_date={self.valuation_date}, "
                f"nodes={len(self.times)}, max_T={self.times[-1]:.1f}Y, "
                f"spot={self.spot_price:.2f}, shape={shape}, "
                f"method={self.interpolation_method})")


class ForwardCurveBootstrapper:
    """Constructs a ForwardCurve from a collection of futures settlement prices.

    For commodity markets, futures settlement prices directly represent forward
    prices, so bootstrapping reduces to sorting contract expiries and passing
    the (tenor, price) pairs to a ForwardCurve interpolator. This is
    fundamentally simpler than interest-rate curve bootstrapping, which
    requires iterative stripping of coupon instruments.

    Attributes:
        interpolation_method: Interpolation scheme forwarded to ForwardCurve;
            either "log_linear" (default) or "monotone_convex".
    """

    def __init__(self, interpolation_method="log_linear", **kwargs):
        self.interpolation_method = interpolation_method

    def bootstrap(self, settlements, valuation_date="2024-12-31", product="CL",
                  spot_price=None):
        """Build a ForwardCurve from a list of settlement records.

        Accepts settlement records as either FuturesSettlement instances or
        plain dicts containing at minimum "time_to_expiry" and "settlement"
        keys. Records are sorted by ascending time to expiry before the curve
        is constructed.

        Args:
            settlements: List of FuturesSettlement objects or dicts, each
                representing one contract expiry. Dicts must contain
                "time_to_expiry" and "settlement" keys; "product" is optional.
            valuation_date: ISO-format date string used as the curve anchor.
                Defaults to "2024-12-31".
            product: Commodity ticker symbol used when the product cannot be
                inferred from a settlement record. Defaults to "CL".
            spot_price: Optional float spot price from a physical market
                assessment (e.g. EIA daily spot). When None, the front
                futures price is used as a proxy.

        Returns:
            A ForwardCurve constructed from the sorted settlement prices using
            the bootstrapper's interpolation method.

        Raises:
            ValueError: If the settlements list is empty.
        """
        if not settlements:
            raise ValueError("No settlements provided for bootstrapping")

        pairs = []
        for s in settlements:
            if isinstance(s, dict):
                t = s["time_to_expiry"]
                p = s["settlement"]
                prod = s.get("product", product)
            else:
                t = s.time_to_expiry
                p = s.settlement
                prod = s.product
            pairs.append((t, p))

        pairs.sort(key=lambda x: x[0])
        times = np.array([p[0] for p in pairs])
        prices = np.array([p[1] for p in pairs])

        return ForwardCurve(
            times, prices, valuation_date, self.interpolation_method,
            product, spot_price=spot_price if spot_price is not None else prices[0],
        )

    def validate(self, curve, settlements):
        """Reprice settlement contracts against a bootstrapped curve.

        Evaluates the curve at each settlement's tenor and computes the
        pricing error in both dollar and basis-point terms. Because the curve
        is bootstrapped directly from settlements the errors should be
        near-zero; this method is primarily useful for sanity-checking and
        detecting numerical issues.

        Args:
            curve: A ForwardCurve instance produced by bootstrap().
            settlements: List of FuturesSettlement objects or dicts used to
                build the curve. Accepts the same formats as bootstrap().

        Returns:
            A pandas DataFrame with one row per settlement and columns:
            "contract", "tenor", "market_price", "model_price", "error",
            and "error_bps".
        """
        results = []
        for s in settlements:
            if isinstance(s, dict):
                code = s.get("contract_code", "")
                tenor = s["time_to_expiry"]
                mkt_price = s["settlement"]
            else:
                code = s.contract_code
                tenor = s.time_to_expiry
                mkt_price = s.settlement

            model_price = curve.forward_price(tenor)
            error = model_price - mkt_price

            results.append({
                "contract": code,
                "tenor": tenor,
                "market_price": mkt_price,
                "model_price": model_price,
                "error": error,
                "error_bps": error / mkt_price * 10000 if mkt_price > 0 else 0,
            })

        return pd.DataFrame(results)


def build_forward_curve(date=None, product="CL", interpolation_method="log_linear",
                        eia_api_key=None):
    """Fetch settlement data and bootstrap a forward curve in a single call.

    Convenience wrapper that instantiates a CommodityDataLoader, retrieves
    the futures strip for the given date and product from the local cache or
    EIA API, and passes the results to ForwardCurveBootstrapper. Suitable
    for interactive use and scripting when fine-grained control over the
    loader or bootstrapper is not required.

    Args:
        date: ISO-format date string for which to build the curve. Defaults
            to None, which the data loader resolves to today's date.
        product: Commodity ticker symbol (e.g. "CL", "HO", "RB", "NG").
            Defaults to "CL".
        interpolation_method: Interpolation scheme to use; either
            "log_linear" (default) or "monotone_convex".
        eia_api_key: Optional EIA API key string. If None the loader falls
            back to the EIA_API_KEY environment variable.

    Returns:
        A ForwardCurve bootstrapped from the most recently cached or fetched
        settlement strip for the specified date and product.
    """
    from .data_loader import CommodityDataLoader

    loader = CommodityDataLoader(eia_api_key=eia_api_key)
    settlements = loader.get_strip_for_date(date, product)

    bootstrapper = ForwardCurveBootstrapper(interpolation_method=interpolation_method)
    val_date = date or "2024-12-31"
    return bootstrapper.bootstrap(settlements, val_date, product)
