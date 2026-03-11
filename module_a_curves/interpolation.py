"""Interpolation methods for commodity forward curves.

Provides two interpolator classes used by ForwardCurve to evaluate prices
continuously between bootstrapped tenor nodes:

    LogLinearInterpolator   -- piecewise linear in log-space, equivalent to
                               constant continuously compounded forward rates
                               between nodes.
    MonotoneConvexInterpolator -- PCHIP spline on instantaneous zero rates,
                               ensuring monotone discount factors and smooth
                               instantaneous forward curves.

Both classes follow the same interface: construct with (times, pseudo_dfs),
call as a function to get a discount factor, and use .forward() for the
instantaneous forward rate.
"""

import numpy as np
from scipy.interpolate import PchipInterpolator


class LogLinearInterpolator:
    """Interpolates discount factors log-linearly between tenor nodes.

    Stores the natural logarithm of each discount factor and interpolates
    linearly in log-space, which is equivalent to assuming a piecewise
    constant continuously compounded forward rate between nodes. Beyond the
    last node the flat zero rate from the final segment is extrapolated.

    Attributes:
        times: NumPy array of year-fraction tenor nodes.
        dfs: NumPy array of discount factors at each tenor node.
        log_dfs: NumPy array of natural logarithms of the discount factors,
            used as the interpolation target.
    """

    def __init__(self, times, dfs):
        self.times = np.asarray(times, dtype=float)
        self.dfs = np.asarray(dfs, dtype=float)
        self.log_dfs = np.log(self.dfs)

    def __call__(self, t):
        """Evaluate the interpolated discount factor at a given tenor.

        Returns 1.0 for t <= 0. For t beyond the last node, extrapolates
        using the flat zero rate implied by the final node. Otherwise
        performs log-linear interpolation between the bracketing nodes.

        Args:
            t: Tenor expressed as a year fraction. Values <= 0 return 1.0.

        Returns:
            Interpolated discount factor as a float in the range (0, 1].
        """
        if t <= 0:
            return 1.0
        if t >= self.times[-1]:
            zero_last = -self.log_dfs[-1] / self.times[-1]
            return np.exp(-zero_last * t)
        return np.exp(np.interp(t, self.times, self.log_dfs))

    def discount_factors(self, t_array):
        """Evaluate the interpolated discount factor at an array of tenors.

        Vectorised wrapper that calls __call__ for each element and
        collects results into a NumPy array.

        Args:
            t_array: Iterable of year-fraction tenors.

        Returns:
            NumPy array of discount factors corresponding to each input tenor.
        """
        return np.array([self(t) for t in t_array])

    def forward(self, t, dt=1 / 365):
        """Compute the instantaneous forward rate at a given tenor.

        Approximates the instantaneous forward using a finite difference of
        the log discount factor over a small step dt:
            f(t) = -(ln D(t+dt) - ln D(t)) / dt.

        Args:
            t: Tenor expressed as a year fraction. Values <= 0 are clamped
                to dt before evaluation.
            dt: Finite-difference step size in years. Defaults to 1/365
                (one calendar day).

        Returns:
            Instantaneous forward rate as a float.
        """
        if t <= 0:
            t = dt
        log_d1 = np.log(self(t))
        log_d2 = np.log(self(t + dt))
        return -(log_d2 - log_d1) / dt


class MonotoneConvexInterpolator:
    """Monotone-convex interpolator using PCHIP splines on zero rates.

    Converts discount factors to continuously compounded zero rates and fits
    a Piecewise Cubic Hermite Interpolating Polynomial (PCHIP) through the
    resulting (tenor, zero-rate) pairs. PCHIP preserves monotonicity of the
    input data, preventing spurious oscillations in the implied forward curve.
    Beyond the last calibrated tenor the zero rate is held flat at its
    terminal value.

    Attributes:
        times: NumPy array of year-fraction tenor nodes (including any
            prepended t=0 anchor).
        dfs: NumPy array of discount factors at each input node.
    """

    def __init__(self, times, dfs):
        self.times = np.asarray(times, dtype=float)
        self.dfs = np.asarray(dfs, dtype=float)

        mask = self.times > 1e-10
        t_pos = self.times[mask]
        df_pos = self.dfs[mask]
        zero_rates = -np.log(df_pos) / t_pos

        if t_pos[0] > 0.01:
            t_interp = np.concatenate([[0.0], t_pos])
            z_interp = np.concatenate([[zero_rates[0]], zero_rates])
        else:
            t_interp = t_pos
            z_interp = zero_rates

        self._pchip = PchipInterpolator(t_interp, z_interp, extrapolate=True)
        self._max_t = t_pos[-1]
        self._last_zero = zero_rates[-1]

    def __call__(self, t):
        """Evaluate the interpolated discount factor at a given tenor.

        Returns 1.0 for t <= 0. Beyond the last calibrated tenor the
        terminal zero rate is held flat (constant extrapolation). Within
        the calibrated range the PCHIP spline is evaluated and converted
        back to a discount factor.

        Args:
            t: Tenor expressed as a year fraction. Values <= 0 return 1.0.

        Returns:
            Interpolated discount factor as a float in the range (0, 1].
        """
        if t <= 0:
            return 1.0
        if t > self._max_t:
            return np.exp(-self._last_zero * t)
        zero_rate = float(self._pchip(t))
        return np.exp(-zero_rate * t)

    def discount_factors(self, t_array):
        """Evaluate the interpolated discount factor at an array of tenors.

        Vectorised wrapper that calls __call__ for each element and
        collects results into a NumPy array.

        Args:
            t_array: Iterable of year-fraction tenors.

        Returns:
            NumPy array of discount factors corresponding to each input tenor.
        """
        return np.array([self(t) for t in t_array])

    def zero_rate(self, t):
        """Return the interpolated continuously compounded zero rate.

        Evaluates the PCHIP spline at t to obtain the zero rate. At t <= 0
        the spline is evaluated at 0.0. Beyond the last calibrated tenor the
        terminal zero rate is returned.

        Args:
            t: Tenor expressed as a year fraction.

        Returns:
            Continuously compounded zero rate as a float.
        """
        if t <= 0:
            return float(self._pchip(0.0))
        if t > self._max_t:
            return self._last_zero
        return float(self._pchip(t))

    def forward(self, t, dt=1 / 365):
        """Compute the instantaneous forward rate at a given tenor.

        Uses the analytical derivative of the PCHIP spline for the zero
        rate r(t) and applies the identity f(t) = r(t) + t * r'(t).
        Beyond the last calibrated tenor the terminal zero rate is returned
        as a flat forward. Values at or below zero are clamped to dt.

        Args:
            t: Tenor expressed as a year fraction. Values <= 0 are clamped
                to dt before evaluation.
            dt: Finite-difference step parameter included for interface
                compatibility with LogLinearInterpolator; the analytical
                derivative is used instead of a finite difference. Defaults
                to 1/365.

        Returns:
            Instantaneous forward rate as a float.
        """
        if t <= 0:
            t = dt
        if t > self._max_t:
            return self._last_zero
        r = float(self._pchip(t))
        r_prime = float(self._pchip(t, 1))
        return r + t * r_prime
