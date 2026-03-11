"""Seasonal forward curve model for energy commodities.

Decomposes an observed forward curve into three additive components:

    F(t) = trend(t) + seasonal(t) + residual(t)

The seasonal component is represented by a truncated Fourier series:

    f_seasonal(t) = a0
                  + a1*sin(2*pi*t) + a2*cos(2*pi*t)
                  + a3*sin(4*pi*t) + a4*cos(4*pi*t)

where t is measured in calendar years from the valuation date. The
convenience yield implied by the base curve is extracted as a piecewise
constant function between contract expiries:

    y_i  for t in [T_{i-1}, T_i)

Default seasonal amplitudes for CL, HO, RB, and NG are pre-calibrated
from historical EIA seasonal demand patterns and stored in
_DEFAULT_SEASONAL_PARAMS.
"""

import datetime

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .curve_bootstrapper import ForwardCurve


# Default seasonal amplitudes by product (calibrated to historical averages)
# Source: EIA seasonal analysis of petroleum product demand patterns
_DEFAULT_SEASONAL_PARAMS = {
    "CL": {"a1": -1.5, "a2": 0.8, "a3": 0.3, "a4": -0.2},
    "HO": {"a1": -0.08, "a2": 0.05, "a3": 0.02, "a4": -0.01},
    "RB": {"a1": 0.06, "a2": -0.04, "a3": 0.015, "a4": 0.005},
    "NG": {"a1": -0.20, "a2": 0.15, "a3": -0.05, "a4": 0.03},
}


class SeasonalForwardCurve:
    """Seasonal decomposition of a commodity forward curve.

    Wraps a bootstrapped ForwardCurve and overlays a Fourier-series seasonal
    model, decomposing the observed curve into trend, seasonal, and residual
    components. Seasonal parameters can be used at their pre-calibrated
    defaults or refined by calling calibrate() against a live settlement strip.

    Attributes:
        base_curve: The underlying ForwardCurve whose prices are decomposed.
        product: Commodity ticker symbol (e.g. "CL", "HO"). Inherited from
            base_curve when not supplied explicitly.
        valuation_date: ISO-format date string or date object used to anchor
            the seasonal calendar. Inherited from base_curve when not supplied.
        seasonal_params: NumPy array of five Fourier coefficients
            [a0, a1, a2, a3, a4] controlling the seasonal shape.
        convenience_yields: NumPy array of piecewise constant convenience
            yields extracted at each tenor node of the base curve.
    """

    def __init__(self, base_curve, product=None, valuation_date=None):
        self.base_curve = base_curve
        self.product = product or base_curve.product
        self.valuation_date = valuation_date or base_curve.valuation_date

        val = self._to_date(self.valuation_date)
        self._val_date = val
        self._month_frac = (val.month - 1 + val.day / 30.0) / 12.0

        default = _DEFAULT_SEASONAL_PARAMS.get(self.product, _DEFAULT_SEASONAL_PARAMS["CL"])
        self.seasonal_params = np.array([
            0.0,
            default["a1"], default["a2"],
            default["a3"], default["a4"],
        ])

        self.convenience_yields = self._extract_convenience_yields()

    def calibrate(self, times, prices, regularization=1.0):
        """Calibrate seasonal Fourier parameters to an observed futures strip.

        Minimises a penalised least-squares objective combining the repricing
        error across all provided tenor-price pairs with a Tikhonov
        regularisation term that anchors the solution near the prior
        (default) seasonal parameters. Optimisation uses L-BFGS-B.

        Args:
            times: Array-like of year-fraction tenors corresponding to each
                observed price. Must be non-empty.
            prices: Array-like of observed forward prices aligned with times.
            regularization: Non-negative scalar controlling the strength of
                the L2 penalty on deviations from the prior parameters.
                Larger values keep the calibrated parameters closer to the
                pre-set defaults. Defaults to 1.0.
        """
        times = np.asarray(times)
        prices = np.asarray(prices)

        if len(times) == 0:
            return

        prior = self.seasonal_params.copy()

        def objective(params):
            self.seasonal_params = params
            model_prices = np.array([self._seasonal_adjusted_price(t) for t in times])
            reprice_err = np.sum((model_prices - prices) ** 2)
            reg_err = regularization * np.sum((params - prior) ** 2)
            return reprice_err + reg_err

        result = minimize(
            objective,
            self.seasonal_params,
            method="L-BFGS-B",
            options={"maxiter": 2000, "ftol": 1e-14},
        )

        self.seasonal_params = result.x

    def seasonal_component(self, t):
        """Evaluate the Fourier seasonal component at a given tenor.

        Computes the two-harmonic Fourier series using the calibrated
        parameters, adjusted to align t=0 with the current calendar month
        so that the seasonal pattern tracks the correct time of year.

        Args:
            t: Tenor expressed as a year fraction measured from the
                valuation date.

        Returns:
            Seasonal price adjustment in dollars per unit as a float.
        """
        t_cal = t + self._month_frac
        a = self.seasonal_params
        return (a[0]
                + a[1] * np.sin(2 * np.pi * t_cal)
                + a[2] * np.cos(2 * np.pi * t_cal)
                + a[3] * np.sin(4 * np.pi * t_cal)
                + a[4] * np.cos(4 * np.pi * t_cal))

    def trend_component(self, t):
        """Compute the trend component of the forward curve at a given tenor.

        The trend is defined as the residual after subtracting the seasonal
        component from the observed forward price: trend(t) = F(t) - seasonal(t).

        Args:
            t: Tenor expressed as a year fraction measured from the
                valuation date.

        Returns:
            Trend price level in dollars per unit as a float.
        """
        return self.base_curve.forward_price(t) - self.seasonal_component(t)

    def extract_seasonal_pattern(self, n_points=12):
        """Extract the seasonal adjustment for each calendar month.

        Computes the seasonal component at the approximate year-fraction
        corresponding to each month, expressed relative to the current
        valuation date. Useful for tabular display and downstream analysis.

        Args:
            n_points: Unused legacy parameter retained for interface
                compatibility. The method always returns 12 rows, one
                per calendar month.

        Returns:
            A pandas DataFrame with 12 rows and columns: "month" (1-12),
            "month_name" (abbreviated month name), "seasonal_adjustment"
            (dollar adjustment), and "time_years" (year fraction used).
        """
        rows = []
        for month in range(1, 13):
            t_month = (month - 1 - self._val_date.month + 1) / 12.0
            if t_month < 0:
                t_month += 1.0
            seasonal_adj = self.seasonal_component(t_month)
            rows.append({
                "month": month,
                "month_name": datetime.date(2000, month, 1).strftime("%b"),
                "seasonal_adjustment": seasonal_adj,
                "time_years": t_month,
            })
        return pd.DataFrame(rows)

    def compare_actual_vs_seasonal(self):
        """Compare the observed forward curve against the seasonal model.

        Evaluates the actual forward price, seasonal component, and implied
        trend component on a 500-point grid spanning from the first to the
        last (or one-year, whichever is shorter) tenor. Also computes the
        maximum absolute seasonal adjustment and the monthly pattern table.

        Returns:
            A dict with three keys:
                "forward_comparison": pandas DataFrame with columns "time",
                    "actual_forward", "seasonal_component", and
                    "trend_component" on a fine tenor grid.
                "max_difference": Maximum absolute value of the seasonal
                    component across the tenor grid, as a float.
                "seasonal_pattern": pandas DataFrame from
                    extract_seasonal_pattern() with monthly seasonal
                    adjustments.
        """
        t_grid = np.linspace(
            self.base_curve.times[0],
            min(self.base_curve.times[-1], 1.0),
            500,
        )

        actual = [self.base_curve.forward_price(t) for t in t_grid]
        seasonal = [self.seasonal_component(t) for t in t_grid]
        trend = [a - s for a, s in zip(actual, seasonal)]

        comparison = pd.DataFrame({
            "time": t_grid,
            "actual_forward": actual,
            "seasonal_component": seasonal,
            "trend_component": trend,
        })

        return {
            "forward_comparison": comparison,
            "max_difference": max(abs(s) for s in seasonal),
            "seasonal_pattern": self.extract_seasonal_pattern(),
        }

    def plot_seasonal_decomposition(self, ax=None):
        """Plot the seasonal decomposition of the forward curve.

        Draws three lines on the provided (or newly created) Axes: the actual
        forward prices, the trend component, and the seasonal component shifted
        to the mean forward level for visual comparison. Vertical dashed lines
        mark each tenor node of the base curve up to the plotted maximum.

        Args:
            ax: Optional matplotlib Axes object on which to draw. If None a
                new figure and Axes are created with figsize=(12, 6).

        Returns:
            The matplotlib Axes object containing the plot, allowing further
            customisation by the caller.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))

        max_t = min(1.5, self.base_curve.times[-1])
        t_grid = np.linspace(self.base_curve.times[0], max_t, 500)

        actual = [self.base_curve.forward_price(t) for t in t_grid]
        ax.plot(t_grid, actual, label="Actual Forward", linewidth=2)

        trend = [self.trend_component(t) for t in t_grid]
        ax.plot(t_grid, trend, label="Trend", linewidth=1.5, linestyle="--")

        base_level = np.mean(actual)
        seasonal = [base_level + self.seasonal_component(t) for t in t_grid]
        ax.plot(t_grid, seasonal, label="Seasonal (shifted)", linewidth=1.5, alpha=0.7)

        for t in self.base_curve.times:
            if t <= max_t:
                ax.axvline(t, color="gray", linestyle="--", alpha=0.3, linewidth=0.8)

        ax.set_xlabel("Time (years)")
        ax.set_ylabel(f"Price ($/{'bbl' if self.product == 'CL' else 'unit'})")
        ax.set_title(f"Seasonal Decomposition: {self.product} Forward Curve")
        ax.legend()
        ax.grid(True, alpha=0.3)

        return ax

    def _seasonal_adjusted_price(self, t):
        """Compute the model price combining the base curve and seasonal signal.

        Used internally by calibrate() as the objective's model prediction.
        Adds a damped seasonal component (scaled by 0.1) to the base curve's
        forward price to avoid over-fitting during optimisation.

        Args:
            t: Tenor expressed as a year fraction measured from the
                valuation date.

        Returns:
            Model-predicted forward price in dollars per unit as a float.
        """
        return self.base_curve.forward_price(t) + self.seasonal_component(t) * 0.1

    def _extract_convenience_yields(self):
        """Extract implied convenience yields at each tenor node of the base curve.

        Applies the cost-of-carry identity y = r - (1/t) * ln(F(t)/S) at
        each positive tenor node using a fixed risk-free rate of 4.5% and
        the spot price from the base curve. Nodes at t=0 or with non-positive
        prices receive a yield of zero.

        Returns:
            NumPy array of implied convenience yields, one per tenor node in
            the base curve, in the same order as base_curve.times.
        """
        n = len(self.base_curve.times)
        yields = np.zeros(n)
        r = 0.045
        S = self.base_curve.spot_price

        for i in range(n):
            t = self.base_curve.times[i]
            F = self.base_curve.forward_prices[i]
            if t > 0 and F > 0 and S > 0:
                yields[i] = r - (1.0 / t) * np.log(F / S)

        return yields

    @staticmethod
    def _to_date(d):
        """Normalise a date-like value to a datetime.date object.

        Accepts ISO-format strings, datetime.datetime instances, and
        datetime.date objects, converting each to a plain datetime.date.

        Args:
            d: A date value as a str in ISO format ("YYYY-MM-DD"), a
                datetime.datetime instance, or a datetime.date instance.

        Returns:
            A datetime.date object corresponding to the input value.
        """
        if isinstance(d, str):
            return datetime.date.fromisoformat(d)
        if isinstance(d, datetime.datetime):
            return d.date()
        return d
