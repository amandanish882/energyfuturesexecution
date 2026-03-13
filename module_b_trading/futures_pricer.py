"""Futures pricer: values energy commodity futures positions against a forward curve.

Provides contract specification constants, a FuturesPosition data
class, and a FuturesPricer engine that computes mark-to-market values,
calendar spread values, crack spread values, and full portfolio P&L
summaries using a bootstrapped ForwardCurve.
"""

import sys
from pathlib import Path
from datetime import date, datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from module_a_curves.curve_bootstrapper import ForwardCurve


CONTRACT_SPECS = {
    "CL": {"name": "WTI Crude Oil", "contract_size": 1000, "unit": "bbl",
           "tick_size": 0.01, "tick_value": 10.0},
    "HO": {"name": "Heating Oil", "contract_size": 42000, "unit": "gal",
           "tick_size": 0.0001, "tick_value": 4.20},
    "RB": {"name": "RBOB Gasoline", "contract_size": 42000, "unit": "gal",
           "tick_size": 0.0001, "tick_value": 4.20},
    "NG": {"name": "Natural Gas", "contract_size": 10000, "unit": "MMBtu",
           "tick_size": 0.001, "tick_value": 10.0},
}

MONTH_CODE = {
    "F": 1, "G": 2, "H": 3, "J": 4, "K": 5, "M": 6,
    "N": 7, "Q": 8, "U": 9, "V": 10, "X": 11, "Z": 12,
}


class FuturesPosition:
    """Represents a single commodity futures position.

    Stores all static attributes of a position — product, size,
    direction, entry price, and contract specification — that are
    needed by FuturesPricer to compute mark-to-market values and
    sensitivities.

    Attributes:
        ticker: Exchange ticker string including month and year codes
            (e.g. ``'CLZ26'``).
        product: Two-to-three-character product code (e.g. ``'CL'``,
            ``'NG'``).
        num_contracts: Number of contracts held (positive integer).
        direction: String ``'long'`` or ``'short'`` indicating the
            trade direction.
        entry_price: Trade execution price in USD per contract unit.
        expiry_date: Optional expiry date string for record-keeping.
        contract_size: Number of physical units per contract (e.g.
            1000 barrels for CL). Derived from CONTRACT_SPECS when
            not explicitly provided.
    """

    def __init__(self, ticker, product, num_contracts, direction="long",
                 entry_price=0.0, expiry_date="", contract_size=0.0):
        """Initialise a FuturesPosition.

        Args:
            ticker: Exchange ticker string including the month and
                year codes (e.g. ``'CLZ26'``). The third character
                is used by FuturesPricer to identify the expiry tenor.
            product: Product code string (e.g. ``'CL'``, ``'HO'``,
                ``'RB'``, ``'NG'``). Used to look up the default
                contract specification.
            num_contracts: Positive integer number of contracts.
            direction: ``'long'`` for a bought position or ``'short'``
                for a sold position. Defaults to ``'long'``.
            entry_price: Trade execution price in USD per contract
                unit. Defaults to 0.0.
            expiry_date: Optional expiry date string used for
                labelling only. Defaults to ``''``.
            contract_size: Physical units per contract. When 0.0 the
                value is looked up from CONTRACT_SPECS using
                ``product``; unknown products default to 1000.
                Defaults to 0.0.

        Raises:
            ValueError: If ``num_contracts`` is not positive.
            ValueError: If ``direction`` is not ``'long'`` or
                ``'short'``.
        """
        if num_contracts <= 0:
            raise ValueError(f"num_contracts must be positive, got {num_contracts}")
        if direction not in ("long", "short"):
            raise ValueError(f"Direction must be 'long' or 'short', got '{direction}'")

        self.ticker = ticker
        self.product = product
        self.num_contracts = num_contracts
        self.direction = direction
        self.entry_price = entry_price
        self.expiry_date = expiry_date

        if contract_size == 0.0:
            spec = CONTRACT_SPECS.get(product)
            self.contract_size = spec["contract_size"] if spec else 1000
        else:
            self.contract_size = contract_size

    @property
    def direction_sign(self):
        """Return +1 for a long position or -1 for a short position.

        Returns:
            Integer +1 when ``direction`` is ``'long'``, or -1 when
            ``direction`` is ``'short'``.
        """
        return 1 if self.direction == "long" else -1

    def __repr__(self):
        """Return a concise string representation of the position.

        Returns:
            String of the form
            ``FuturesPosition(LONG Nx TICKER @ PRICE)`` or
            ``FuturesPosition(SHORT Nx TICKER @ PRICE)``.
        """
        dir_label = "LONG" if self.direction == "long" else "SHORT"
        return (
            f"FuturesPosition({dir_label} {self.num_contracts}x "
            f"{self.ticker} @ {self.entry_price:.2f})"
        )


class FuturesPricer:
    """Prices commodity futures positions using a bootstrapped forward curve.

    Provides mark-to-market valuation for individual positions and
    full portfolios, along with calendar spread and crack spread
    valuation helpers. All prices are sourced from the ForwardCurve
    passed at construction time; tenor matching from ticker codes is
    handled internally.

    Attributes:
        _curve: ForwardCurve instance used to retrieve forward prices
            at requested tenors.
    """

    def __init__(self, forward_curve):
        """Initialise the futures pricer with a forward curve.

        Args:
            forward_curve: ForwardCurve instance providing forward
                prices at any requested tenor. Must not be None.

        Raises:
            ValueError: If ``forward_curve`` is None.
        """
        if forward_curve is None:
            raise ValueError("ForwardCurve must not be None")
        self._curve = forward_curve

    def mark_to_market(self, position):
        """Compute the mark-to-market P&L for a single futures position.

        Computes the exact tenor from the curve's valuation date to
        the contract's expiry, retrieves the interpolated forward price,
        and calculates the unrealised P&L as:
            MTM = (F_current - entry_price) * contracts
                  * contract_size * direction_sign

        Args:
            position: FuturesPosition instance whose ticker is used to
                identify the best-matching tenor in the forward curve.

        Returns:
            Float representing the mark-to-market P&L in USD.
            Positive values indicate an unrealised gain; negative
            values indicate an unrealised loss.
        """
        t = self._find_tenor(position)
        current_price = self._curve.forward_price(t)
        mtm = ((current_price - position.entry_price)
               * position.num_contracts
               * position.contract_size
               * position.direction_sign)
        return mtm

    def calendar_spread_value(self, product, front_month_t, back_month_t, num_spreads=1):
        """Value a calendar spread position (long deferred, short front).

        Retrieves the forward prices at both tenors and computes the
        spread value as:
            value = (F_back - F_front) * num_spreads * contract_size

        Args:
            product: Product code string used to look up the contract
                size from CONTRACT_SPECS. Unknown products default to
                the CL specification.
            front_month_t: Time to expiry of the front (short) leg
                in years.
            back_month_t: Time to expiry of the back (long) leg in
                years.
            num_spreads: Number of calendar spread units to value.
                Defaults to 1.

        Returns:
            Float representing the total USD value of the calendar
            spread. Positive values indicate the back month is at a
            premium (contango); negative values indicate the front
            month is at a premium (backwardation).
        """
        spec = CONTRACT_SPECS.get(product, CONTRACT_SPECS["CL"])
        F1 = self._curve.forward_price(front_month_t)
        F2 = self._curve.forward_price(back_month_t)
        spread = F2 - F1
        return spread * num_spreads * spec["contract_size"]

    def crack_spread_value(self, cl_price, ho_price, rb_price, num_cracks=1):
        """Value a 3:2:1 refinery crack spread.

        Computes the per-barrel crack spread from input product prices
        and scales by the number of crack units and the crude contract
        size (1000 barrels):
            product_revenue = 2 * rb_price * 42 + 1 * ho_price * 42
            crack_per_bbl   = (product_revenue - 3 * cl_price) / 3
                                   (implicit in dollar terms)
            value           = crack_per_bbl * num_cracks * 1000

        Args:
            cl_price: WTI crude oil price in USD per barrel.
            ho_price: Heating oil price in USD per gallon.
            rb_price: RBOB gasoline price in USD per gallon.
            num_cracks: Number of 3:2:1 crack spread units to value.
                Defaults to 1.

        Returns:
            Float representing the total USD value of the crack
            spread position.
        """
        product_revenue = 2 * rb_price * 42 + 1 * ho_price * 42
        crude_cost = 3 * cl_price
        crack_per_bbl = product_revenue - crude_cost
        return crack_per_bbl * num_cracks * 1000

    def portfolio_mtm(self, positions):
        """Mark-to-market every position in a portfolio and return a summary.

        Iterates over the list of positions, computes the forward
        price and MTM P&L for each, and assembles the results with
        contract metadata into a single DataFrame.

        Args:
            positions: Iterable of FuturesPosition instances to value.

        Returns:
            pandas.DataFrame with one row per position and columns:
            ``ticker``, ``product``, ``direction``, ``contracts``,
            ``entry_price``, ``current_price``, ``mtm_usd``,
            ``mtm_per_contract``, ``contract_size``, and ``unit``.
        """
        rows = []
        for pos in positions:
            t = self._find_tenor(pos)
            current = self._curve.forward_price(t)
            mtm = self.mark_to_market(pos)
            spec = CONTRACT_SPECS.get(pos.product, CONTRACT_SPECS["CL"])
            rows.append({
                "ticker": pos.ticker,
                "product": pos.product,
                "direction": pos.direction,
                "contracts": pos.num_contracts,
                "entry_price": pos.entry_price,
                "current_price": current,
                "mtm_usd": mtm,
                "mtm_per_contract": mtm / pos.num_contracts if pos.num_contracts > 0 else 0,
                "contract_size": spec["contract_size"],
                "unit": spec["unit"],
            })
        return pd.DataFrame(rows)

    def forward_price(self, t):
        """Return the forward price at a given tenor from the underlying curve.

        Args:
            t: Tenor in years at which to retrieve the forward price.

        Returns:
            Float forward price in USD at tenor ``t``.
        """
        return self._curve.forward_price(t)

    def _find_tenor(self, position):
        """Compute the year-fraction tenor from the curve's valuation date
        to the position's contract expiry, and return it directly so that
        ``forward_price(t)`` can interpolate the price at the exact tenor.

        Extracts the month code and two-digit year from the ticker
        (e.g. ``'CLZ26'`` → December 2026), approximates the contract
        expiry as the 20th of the **preceding** calendar month (CME
        energy futures expire ~20th of month before delivery), and
        computes the day-count fraction relative to the curve's
        valuation date.

        Args:
            position: FuturesPosition instance whose ``ticker``
                attribute is inspected for the month/year code.

        Returns:
            Float tenor in years.  Falls back to ``times[0]`` when no
            positive tenor is available, or 0.25 when ``times`` is empty.
        """
        ticker = position.ticker
        times = self._curve.times

        # Parse valuation date from the curve
        try:
            val_date = date.fromisoformat(str(self._curve.valuation_date))
        except (ValueError, TypeError):
            val_date = None

        # Try to extract month code and year from ticker
        # e.g. "CLZ26" -> month_char='Z', year_str='26'
        if len(ticker) >= 4 and val_date is not None:
            month_char = ticker[2].upper()
            month_num = MONTH_CODE.get(month_char)
            year_str = ticker[3:]

            if month_num is not None and year_str.isdigit():
                year = 2000 + int(year_str)

                # CME energy futures expire ~20th of month before delivery
                # e.g. CLK26 (May delivery) expires ~Apr 20
                expiry_month = month_num - 1 if month_num > 1 else 12
                expiry_year = year if month_num > 1 else year - 1
                expiry_date = date(expiry_year, expiry_month, 20)

                target_t = (expiry_date - val_date).days / 365.0

                # If the contract has already expired, clamp to a small
                # positive tenor so we still return a sensible price
                if target_t <= 0:
                    target_t = 1.0 / 365.0

                return target_t

        # Fallback: return the first non-zero tenor
        for t in times:
            if abs(t) > 1e-6:
                return t
        return times[0] if len(times) > 0 else 0.25

    def __repr__(self):
        """Return a concise string representation of the pricer.

        Returns:
            String of the form ``FuturesPricer(curve=<ForwardCurve>)``
            identifying the underlying forward curve.
        """
        return f"FuturesPricer(curve={self._curve})"
