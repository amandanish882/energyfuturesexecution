"""Alpha signal generators for energy commodity markets.

Provides a suite of individual alpha signal classes and a composite
model that blends them into a single weighted score. All signals
follow a common convention: the returned value is a z-score clipped
to the range [-3, +3], where positive values indicate a bullish
view (go long) and negative values indicate a bearish view (go short).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from module_a_curves.curve_bootstrapper import ForwardCurve
from module_a_curves.data_loader import CommodityDataLoader


class AlphaSignal:
    """Abstract base class for all alpha signal generators.

    Defines the common interface that every concrete signal must
    implement. Subclasses override ``compute`` to produce a
    z-score in the range [-3, +3] representing directional
    conviction, and may call the shared ``_clip`` helper to
    enforce that bound.
    """

    def compute(self, **kwargs):
        """Compute the signal value as a z-score in [-3, +3].

        Subclasses override this method to implement their specific
        signal logic. The default implementation returns a neutral
        value of 0.0 (no view).

        Args:
            **kwargs: Arbitrary keyword arguments consumed by the
                concrete subclass implementation.

        Returns:
            Float in [-3.0, +3.0]. Positive values are bullish;
            negative values are bearish.
        """
        return 0.0

    @staticmethod
    def _clip(value):
        """Clip a raw signal value to the canonical [-3, +3] range.

        Args:
            value: Raw float signal value before clipping.

        Returns:
            Float equal to ``value`` clamped to [-3.0, +3.0].
        """
        return float(np.clip(value, -3.0, 3.0))


class TermStructureSignal(AlphaSignal):
    """Alpha signal derived from the shape of the futures term structure.

    Computes the percentage spread between the front and second
    contract, standardises it against recent history, and returns
    a z-score. Backwardation (front > deferred) produces a positive
    (bullish) signal; contango (front < deferred) produces a negative
    (bearish) signal.

    Attributes:
        _window: Number of historical observations used to compute
            the rolling mean and standard deviation for z-score
            normalisation.
        _history: Running list of observed spread-percentage values.
    """

    def __init__(self, lookback_std_window=60):
        """Initialise the term-structure signal.

        Args:
            lookback_std_window: Number of past observations to
                include when calculating the rolling standard
                deviation used for z-score normalisation. Defaults
                to 60.
        """
        self._window = lookback_std_window
        self._history = []

    def compute(self, forward_curve=None, front_price=None, deferred_price=None, **kwargs):
        """Compute the term-structure signal as a clipped z-score.

        Accepts either a ForwardCurve object (from which the first two
        tenor points are extracted) or explicit front and deferred
        prices. The raw spread percentage is stored in history before
        being normalised. When fewer than five observations are
        available the signal falls back to a simple scaled spread.

        Args:
            forward_curve: Optional ForwardCurve instance. If provided,
                the first two available tenors are used as front and
                deferred prices respectively.
            front_price: Optional float representing the front-month
                futures price. Used only when ``forward_curve`` is None.
            deferred_price: Optional float representing the deferred
                futures price. Used only when ``forward_curve`` is None.
            **kwargs: Additional keyword arguments are ignored.

        Returns:
            Float in [-3.0, +3.0]. Positive values indicate
            backwardation (bullish); negative values indicate contango
            (bearish). Returns 0.0 if inputs are insufficient or
            degenerate.
        """
        if forward_curve is not None:
            times = forward_curve.times
            if len(times) < 2:
                return 0.0
            f1 = forward_curve.forward_price(times[0])
            f2 = forward_curve.forward_price(times[1])
        elif front_price is not None and deferred_price is not None:
            f1, f2 = front_price, deferred_price
        else:
            return 0.0

        if abs(f1) < 1e-6:
            return 0.0

        spread_pct = -(f2 - f1) / f1
        self._history.append(spread_pct)

        if len(self._history) < 5:
            return self._clip(spread_pct * 10)

        arr = np.array(self._history[-self._window:])
        mu, sigma = arr.mean(), arr.std()
        if sigma < 1e-8:
            return 0.0
        z = (spread_pct - mu) / sigma
        return self._clip(z)


class InventorySignal(AlphaSignal):
    """Alpha signal derived from EIA crude-oil inventory deviations.

    Measures how far current crude-oil inventories deviate from
    their seasonal norm, standardises the deviation over a rolling
    60-observation window, and returns the negated z-score so that
    above-normal inventories are bearish (negative) and below-normal
    inventories are bullish (positive).

    Attributes:
        _seasonal: Dictionary mapping month integer (1–12) to the
            expected seasonal average inventory level in million
            barrels.
        _history: Running list of observed seasonal-deviation
            percentages used for rolling normalisation.
    """

    def __init__(self, seasonal_avg=None):
        """Initialise the inventory signal.

        Args:
            seasonal_avg: Optional dictionary mapping month integers
                (1–12) to seasonal average inventory levels in
                million barrels. If None, a hard-coded default
                seasonal pattern is used.
        """
        self._seasonal = seasonal_avg or {
            1: 435.0, 2: 445.0, 3: 455.0, 4: 460.0, 5: 465.0, 6: 460.0,
            7: 450.0, 8: 440.0, 9: 435.0, 10: 430.0, 11: 435.0, 12: 438.0,
        }
        self._history = []

    def compute(self, inventory_level=0.0, month=1, **kwargs):
        """Compute the inventory signal as a clipped z-score.

        Calculates the percentage deviation of the observed inventory
        level from the seasonal norm for the given month, appends it
        to history, and returns the negated z-score over the rolling
        window. When fewer than five observations are available a
        simple scaled shortfall is returned instead.

        Args:
            inventory_level: Current crude-oil inventory level in
                million barrels, as reported by the EIA. Defaults
                to 0.0.
            month: Integer (1–12) representing the current calendar
                month, used to look up the seasonal average. Defaults
                to 1 (January).
            **kwargs: Additional keyword arguments are ignored.

        Returns:
            Float in [-3.0, +3.0]. Positive values indicate
            below-seasonal inventories (bullish); negative values
            indicate above-seasonal inventories (bearish). Returns
            0.0 when the rolling standard deviation is negligible.
        """
        seasonal_norm = self._seasonal.get(month, 445.0)
        deviation = inventory_level - seasonal_norm
        deviation_pct = deviation / seasonal_norm

        self._history.append(deviation_pct)

        if len(self._history) < 5:
            return self._clip(-deviation_pct * 10)

        arr = np.array(self._history[-60:])
        mu, sigma = arr.mean(), arr.std()
        if sigma < 1e-8:
            return 0.0
        z = (deviation_pct - mu) / sigma
        return self._clip(-z)  # above normal = bearish


class MomentumSignal(AlphaSignal):
    """Price momentum signal based on a fast vs. slow moving-average crossover.

    Tracks a rolling price history and computes the percentage by
    which the fast moving average exceeds the slow moving average.
    A positive crossover is scaled into a bullish z-score; a negative
    crossover produces a bearish signal.

    Attributes:
        _fast: Number of periods in the fast moving-average window.
        _slow: Number of periods in the slow moving-average window.
        _prices: Running list of observed prices used to compute
            both moving averages.
    """

    def __init__(self, fast_window=5, slow_window=20):
        """Initialise the momentum signal.

        Args:
            fast_window: Number of periods for the fast (short-term)
                moving average. Defaults to 5.
            slow_window: Number of periods for the slow (long-term)
                moving average. Defaults to 20.
        """
        self._fast = fast_window
        self._slow = slow_window
        self._prices = []

    def compute(self, price=0.0, **kwargs):
        """Compute the momentum signal as a clipped z-score.

        Appends the latest price to history and returns 0.0 until
        enough observations exist to fill the slow window. Once
        sufficient data is available, the crossover percentage is
        scaled by a factor of 50 to map it to the z-score range and
        then clipped.

        Args:
            price: The most recent futures price observation.
                Defaults to 0.0.
            **kwargs: Additional keyword arguments are ignored.

        Returns:
            Float in [-3.0, +3.0]. Positive values indicate upward
            price momentum (bullish); negative values indicate
            downward momentum (bearish). Returns 0.0 while fewer
            than ``slow_window`` prices have been observed or when
            the slow moving average is negligibly small.
        """
        self._prices.append(price)
        if len(self._prices) < self._slow:
            return 0.0

        arr = np.array(self._prices)
        fast_ma = arr[-self._fast:].mean()
        slow_ma = arr[-self._slow:].mean()
        if abs(slow_ma) < 1e-6:
            return 0.0

        crossover = (fast_ma - slow_ma) / slow_ma
        z = crossover * 50
        return self._clip(z)


class SeasonalSignal(AlphaSignal):
    """Alpha signal capturing known seasonal demand patterns in energy markets.

    Returns a pre-calibrated seasonal bias score for the current
    calendar month. The bias table encodes historical seasonal
    tendencies for each supported product (WTI crude, RBOB gasoline,
    heating oil, natural gas) and is clipped to the standard
    [-3, +3] z-score range.

    Attributes:
        _product: Product code string (e.g. ``'CL'``, ``'RB'``,
            ``'HO'``, ``'NG'``) used to select the bias table.
    """

    _SEASONAL_BIAS = {
        "CL": {1: -0.5, 2: -0.8, 3: -0.3, 4: 0.5, 5: 1.0, 6: 1.2,
               7: 0.8, 8: 0.3, 9: -0.5, 10: -0.8, 11: -0.3, 12: 0.2},
        "RB": {1: -1.0, 2: -0.5, 3: 0.5, 4: 1.5, 5: 2.0, 6: 1.5,
               7: 1.0, 8: 0.5, 9: -0.5, 10: -1.0, 11: -1.5, 12: -1.5},
        "HO": {1: 1.5, 2: 1.0, 3: 0.5, 4: -0.5, 5: -1.0, 6: -1.0,
               7: -0.5, 8: 0.0, 9: 0.5, 10: 1.0, 11: 1.5, 12: 2.0},
        "NG": {1: 1.5, 2: 1.0, 3: 0.0, 4: -1.0, 5: -1.5, 6: -0.5,
               7: 0.5, 8: 1.0, 9: -0.5, 10: -1.0, 11: 0.5, 12: 1.5},
    }

    def __init__(self, product="CL"):
        """Initialise the seasonal signal for a given product.

        Args:
            product: Energy product code. Must be one of ``'CL'``
                (WTI crude), ``'RB'`` (RBOB gasoline), ``'HO'``
                (heating oil), or ``'NG'`` (natural gas). Unknown
                products fall back to the ``'CL'`` bias table.
                Defaults to ``'CL'``.
        """
        self._product = product

    def compute(self, month=1, **kwargs):
        """Compute the seasonal signal for the current calendar month.

        Looks up the pre-calibrated seasonal bias for the configured
        product and month, then clips the result to [-3, +3].

        Args:
            month: Integer (1–12) representing the current calendar
                month. Defaults to 1 (January).
            **kwargs: Additional keyword arguments are ignored.

        Returns:
            Float in [-3.0, +3.0] representing the seasonal
            directional bias. Positive values are seasonally bullish;
            negative values are seasonally bearish.
        """
        bias_table = self._SEASONAL_BIAS.get(self._product, self._SEASONAL_BIAS["CL"])
        signal = bias_table.get(month, 0.0)
        return self._clip(signal)


class CrackSpreadSignal(AlphaSignal):
    """Alpha signal derived from the 3:2:1 crack spread deviation.

    Computes the 3:2:1 refinery crack spread from WTI crude, RBOB
    gasoline, and heating oil prices and standardises it against a
    fixed long-run mean and standard deviation. Because a wide crack
    spread is bearish for crude (refiners sell products, buy crude,
    pushing crude supply higher), the raw z-score is negated before
    being returned.

    Attributes:
        _mean: Long-run average crack spread in USD per barrel.
        _std: Long-run standard deviation of the crack spread in
            USD per barrel.
        _history: Running list of observed crack-spread values.
    """

    def __init__(self, mean_crack=25.0, std_crack=8.0):
        """Initialise the crack-spread signal.

        Args:
            mean_crack: Expected long-run average of the 3:2:1 crack
                spread in USD per barrel. Defaults to 25.0.
            std_crack: Expected long-run standard deviation of the
                3:2:1 crack spread in USD per barrel. Defaults to 8.0.
        """
        self._mean = mean_crack
        self._std = std_crack
        self._history = []

    def compute(self, cl_price=0.0, ho_price=0.0, rb_price=0.0, **kwargs):
        """Compute the crack-spread signal as a clipped z-score.

        Calculates the 3:2:1 crack spread as:
            crack = (2 * rb_price * 42 + 1 * ho_price * 42 - 3 * cl_price) / 3

        Standardises against the fixed mean and standard deviation and
        negates the z-score so that a wide crack (above mean) is
        bearish for crude.

        Args:
            cl_price: WTI crude oil price in USD per barrel. A value
                of zero or less causes the method to return 0.0.
                Defaults to 0.0.
            ho_price: Heating oil price in USD per gallon.
                Defaults to 0.0.
            rb_price: RBOB gasoline price in USD per gallon.
                Defaults to 0.0.
            **kwargs: Additional keyword arguments are ignored.

        Returns:
            Float in [-3.0, +3.0]. Negative values indicate a wide
            crack spread (bearish crude); positive values indicate a
            narrow crack spread (bullish crude). Returns 0.0 when
            ``cl_price`` is non-positive or the standard deviation is
            negligible.
        """
        if cl_price <= 0:
            return 0.0
        crack = (2 * rb_price * 42 + 1 * ho_price * 42 - 3 * cl_price) / 3
        self._history.append(crack)

        if self._std < 1e-6:
            return 0.0
        z = (crack - self._mean) / self._std
        return self._clip(-z)  # wide cracks = bearish crude


class CompositeAlphaModel:
    """Blends multiple alpha signals into a single weighted composite score.

    Maintains a registry of named AlphaSignal instances, each
    associated with a scalar weight. When ``compute_composite`` is
    called, every registered signal is evaluated with the same
    keyword arguments and the results are combined via a
    weighted average. The final composite is clipped to [-3, +3].

    Attributes:
        _signals: Dictionary mapping signal name strings to
            ``(AlphaSignal, float)`` tuples of (signal, weight).
    """

    def __init__(self, signals=None):
        """Initialise the composite alpha model.

        Args:
            signals: Optional dictionary mapping signal name strings
                to ``(AlphaSignal, float)`` tuples. If None an empty
                registry is created and signals must be added via
                ``add_signal``.
        """
        self._signals = signals or {}

    def add_signal(self, name, signal, weight=1.0):
        """Register a signal with the composite model.

        Args:
            name: Unique string identifier for the signal (e.g.
                ``'momentum'``, ``'inventory'``).
            signal: An AlphaSignal instance whose ``compute`` method
                will be called with the composite's keyword arguments.
            weight: Scalar weight applied to this signal when forming
                the weighted average. Defaults to 1.0.
        """
        self._signals[name] = (signal, weight)

    def compute_composite(self, **kwargs):
        """Compute all individual signals and their weighted composite.

        Iterates over every registered signal, calls ``compute``
        with the provided keyword arguments, and accumulates a
        weighted sum. The composite is normalised by the sum of
        absolute weights and then clipped to [-3, +3].

        Args:
            **kwargs: Keyword arguments forwarded verbatim to each
                registered signal's ``compute`` method (e.g.
                ``forward_curve``, ``inventory_level``, ``month``,
                ``price``, ``cl_price``, etc.).

        Returns:
            Dictionary mapping each registered signal name to its
            individual float score, plus a ``'composite'`` key
            holding the overall weighted z-score in [-3.0, +3.0].
        """
        results = {}
        weighted_sum = 0.0
        total_weight = 0.0

        for name, (signal, weight) in self._signals.items():
            val = signal.compute(**kwargs)
            results[name] = val
            weighted_sum += val * weight
            total_weight += abs(weight)

        composite = weighted_sum / total_weight if total_weight > 0 else 0.0
        results["composite"] = AlphaSignal._clip(composite)
        return results

    @staticmethod
    def default_crude_model():
        """Create a pre-configured composite alpha model for WTI crude oil.

        Assembles a CompositeAlphaModel with five signals and their
        associated weights:

            - term_structure (TermStructureSignal): 30 %
            - inventory (InventorySignal): 25 %
            - momentum (MomentumSignal): 20 %
            - seasonal (SeasonalSignal for 'CL'): 15 %
            - crack_spread (CrackSpreadSignal): 10 %

        Returns:
            A fully configured CompositeAlphaModel instance ready
            for use with WTI crude oil market data.
        """
        model = CompositeAlphaModel()
        model.add_signal("term_structure", TermStructureSignal(), weight=0.30)
        model.add_signal("inventory", InventorySignal(), weight=0.25)
        model.add_signal("momentum", MomentumSignal(), weight=0.20)
        model.add_signal("seasonal", SeasonalSignal("CL"), weight=0.15)
        model.add_signal("crack_spread", CrackSpreadSignal(), weight=0.10)
        return model
