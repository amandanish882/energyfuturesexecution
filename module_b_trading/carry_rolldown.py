"""Roll-yield and carry analytics for commodity futures.

Implements core carry decomposition formulas for commodity forward
curves. Key definitions used throughout this module:

    Roll Yield (annualised):
        RY = (F_near - F_far) / F_near * (365 / days_between)

    Convenience Yield:
        y = r + u - (1/T) * ln(F(T) / S)

    Total Carry:
        carry = roll_yield + convenience_yield  (net of storage)

where r is the risk-free rate, u is the storage cost rate, T is
tenor in years, F(T) is the forward price, and S is the spot price.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from module_a_curves.curve_bootstrapper import ForwardCurve


class RollYieldResult:
    """Data container holding the result of a single roll-yield calculation.

    Stores all inputs and derived outputs for one adjacent-contract
    pair so that results can be inspected, compared, or assembled
    into a DataFrame by the caller.

    Attributes:
        product: Product code string (e.g. ``'CL'``, ``'NG'``).
        front_tenor: Time to expiry of the front contract in years.
        back_tenor: Time to expiry of the back contract in years.
        front_price: Forward price of the front contract in USD.
        back_price: Forward price of the back contract in USD.
        roll_yield_ann: Annualised roll yield expressed as a decimal
            (e.g. 0.12 for 12 %).
        carry_days: Number of calendar days between the two tenors.
        regime: String indicating the curve regime:
            ``'backwardation'`` when front > back, else ``'contango'``.
    """

    def __init__(self, product, front_tenor, back_tenor, front_price, back_price,
                 roll_yield_ann, carry_days, regime):
        """Initialise a RollYieldResult with all computed fields.

        Args:
            product: Product code string (e.g. ``'CL'``).
            front_tenor: Time to expiry of the front contract in years.
            back_tenor: Time to expiry of the back contract in years.
            front_price: Forward price of the front contract in USD.
            back_price: Forward price of the back contract in USD.
            roll_yield_ann: Annualised roll yield as a decimal.
            carry_days: Integer number of calendar days between the
                two tenors.
            regime: String ``'backwardation'`` or ``'contango'``.
        """
        self.product = product
        self.front_tenor = front_tenor
        self.back_tenor = back_tenor
        self.front_price = front_price
        self.back_price = back_price
        self.roll_yield_ann = roll_yield_ann
        self.carry_days = carry_days
        self.regime = regime


class RollYieldCalculator:
    """Computes roll yield and carry analytics from a bootstrapped forward curve.

    Provides methods to calculate annualised roll yield for adjacent
    contract pairs, extract implied convenience yields, rank the best
    roll trades, and compute total carry (roll yield plus convenience
    yield). All price inputs are drawn from the ForwardCurve passed
    at construction time.

    Attributes:
        _curve: ForwardCurve instance providing forward prices at
            arbitrary tenors.
        _r: Risk-free interest rate used in the convenience-yield
            formula, expressed as a decimal (e.g. 0.045 for 4.5 %).
        _u: Annual storage cost rate expressed as a decimal
            (e.g. 0.03 for 3 %).
    """

    def __init__(self, forward_curve, risk_free_rate=0.045, storage_cost_rate=0.03):
        """Initialise the roll-yield calculator.

        Args:
            forward_curve: ForwardCurve instance used to look up
                forward prices at requested tenors.
            risk_free_rate: Annual risk-free rate used in the
                convenience-yield formula, as a decimal. Defaults
                to 0.045 (4.5 %).
            storage_cost_rate: Annual physical storage cost rate as
                a decimal. Defaults to 0.03 (3 %).
        """
        self._curve = forward_curve
        self._r = risk_free_rate
        self._u = storage_cost_rate

    def roll_yield(self, front_t, back_t, product="CL"):
        """Calculate the annualised roll yield between two tenor points.

        Retrieves forward prices at ``front_t`` and ``back_t`` from
        the underlying curve, computes the day count between them,
        and annualises the fractional price difference.

        Args:
            front_t: Time to expiry of the front (near) contract in
                years (e.g. 0.083 for one month).
            back_t: Time to expiry of the back (far) contract in
                years. Must be greater than ``front_t``.
            product: Product code string used to label the result.
                Defaults to ``'CL'``.

        Returns:
            RollYieldResult instance containing the front and back
            prices, annualised roll yield, day count, and curve
            regime string.

        Raises:
            ValueError: If the front price is negligibly close to
                zero (absolute value < 1e-6).
        """
        f1 = self._curve.forward_price(front_t)
        f2 = self._curve.forward_price(back_t)

        if abs(f1) < 1e-6:
            raise ValueError(f"Front price near zero at t={front_t}")

        days_between = int((back_t - front_t) * 365)
        days_between = max(days_between, 1)

        ry_ann = (f1 - f2) / f1 * (365 / days_between)
        regime = "backwardation" if f1 > f2 else "contango"

        return RollYieldResult(
            product=product,
            front_tenor=front_t,
            back_tenor=back_t,
            front_price=f1,
            back_price=f2,
            roll_yield_ann=ry_ann,
            carry_days=days_between,
            regime=regime,
        )

    def convenience_yield(self, tenor, spot=None):
        """Extract the implied convenience yield at a given tenor.

        Applies the cost-of-carry identity to back out the convenience
        yield y from the observable forward price F(T) and the spot
        price S:
            y = r + u - (1/T) * ln(F(T) / S)

        Args:
            tenor: Time to expiry in years at which to evaluate the
                convenience yield. Must be positive.
            spot: Optional spot price in USD. If None, the forward
                price at the shortest available tenor in the curve is
                used as a proxy for spot.

        Returns:
            Float representing the annualised implied convenience
            yield as a decimal. Returns 0.0 if the tenor is
            non-positive or if either the spot or forward price is
            non-positive.
        """
        if tenor <= 0:
            return 0.0

        if spot is None:
            spot = self._curve.forward_price(self._curve.times[0])

        fwd = self._curve.forward_price(tenor)
        if spot <= 0 or fwd <= 0:
            return 0.0

        y = self._r + self._u - (1.0 / tenor) * np.log(fwd / spot)
        return float(y)

    def convenience_yield_curve(self, spot=None):
        """Compute the implied convenience yield at every tenor in the curve.

        Iterates over all positive tenors defined in the underlying
        ForwardCurve and evaluates ``convenience_yield`` at each one,
        collecting the results alongside the corresponding forward
        prices.

        Args:
            spot: Optional spot price in USD used as the S value in
                the convenience-yield formula. If None, the price at
                the shortest available tenor is used as a proxy.

        Returns:
            pandas.DataFrame with columns ``tenor``,
            ``forward_price``, and ``convenience_yield``, one row
            per positive tenor in the curve.
        """
        rows = []
        for t in self._curve.times:
            if t <= 0:
                continue
            cy = self.convenience_yield(t, spot)
            rows.append({
                "tenor": t,
                "forward_price": self._curve.forward_price(t),
                "convenience_yield": cy,
            })
        return pd.DataFrame(rows)

    def roll_yield_matrix(self, product="CL"):
        """Compute roll yield for every adjacent contract pair in the curve.

        Filters the curve's tenor list to positive values, then calls
        ``roll_yield`` for each consecutive (front, back) pair and
        assembles the results into a DataFrame.

        Args:
            product: Product code string used to label each
                RollYieldResult. Defaults to ``'CL'``.

        Returns:
            pandas.DataFrame with columns ``front_tenor``,
            ``back_tenor``, ``front_price``, ``back_price``,
            ``roll_yield_ann``, ``carry_days``, and ``regime``,
            one row per adjacent tenor pair.
        """
        times = [t for t in self._curve.times if t > 0]
        rows = []
        for i in range(len(times) - 1):
            result = self.roll_yield(times[i], times[i + 1], product)
            rows.append({
                "front_tenor": result.front_tenor,
                "back_tenor": result.back_tenor,
                "front_price": result.front_price,
                "back_price": result.back_price,
                "roll_yield_ann": result.roll_yield_ann,
                "carry_days": result.carry_days,
                "regime": result.regime,
            })
        return pd.DataFrame(rows)

    def best_roll_trades(self, product="CL", top_n=3):
        """Identify the highest roll-yield trades across the curve.

        Builds the full roll-yield matrix, sorts rows by the absolute
        value of the annualised roll yield in descending order, and
        returns the top-ranked pairs. Both backwardated and contango
        segments are considered, allowing long or short roll
        strategies.

        Args:
            product: Product code string forwarded to
                ``roll_yield_matrix``. Defaults to ``'CL'``.
            top_n: Maximum number of trade candidates to return.
                Defaults to 3.

        Returns:
            pandas.DataFrame containing the top ``top_n`` rows of the
            roll-yield matrix sorted by descending absolute roll
            yield, with a reset integer index. Returns an empty
            DataFrame if the curve has fewer than two tenors.
        """
        matrix = self.roll_yield_matrix(product)
        if len(matrix) == 0:
            return matrix
        ranked = matrix.reindex(
            matrix["roll_yield_ann"].abs().sort_values(ascending=False).index
        )
        return ranked.head(top_n).reset_index(drop=True)

    def total_carry(self, front_t, back_t):
        """Compute total carry as the sum of roll yield and convenience yield.

        Calls ``roll_yield`` for the specified tenor pair to obtain
        the annualised roll yield, then calls ``convenience_yield``
        at the front tenor to obtain the implied convenience yield,
        and returns their sum.

        Args:
            front_t: Time to expiry of the front contract in years.
            back_t: Time to expiry of the back contract in years.

        Returns:
            Float representing the total annualised carry as a
            decimal (roll yield + convenience yield).

        Raises:
            ValueError: If the front price at ``front_t`` is
                negligibly small (propagated from ``roll_yield``).
        """
        ry = self.roll_yield(front_t, back_t)
        cy = self.convenience_yield(front_t)
        return ry.roll_yield_ann + cy

    def __repr__(self):
        """Return a concise string representation of the calculator.

        Returns:
            String of the form ``RollYieldCalculator(r=X.XXX, u=X.XXX)``
            showing the configured risk-free rate and storage cost rate.
        """
        return f"RollYieldCalculator(r={self._r:.3f}, u={self._u:.3f})"
