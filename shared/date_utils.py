"""Date utility functions for trading calendar operations.

Provides business-day arithmetic, year-fraction conversion under
standard day-count conventions, periodic schedule generation, and
business-day validation, all built on NumPy's busday infrastructure
and pandas Timestamp interoperability.
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd


def business_days_between(start, end):
    """Count the number of business days from start up to but not including end.

    Delegates to ``numpy.busday_count``, which uses the standard
    Monday-to-Friday definition of business days with no holiday
    calendar applied.

    Args:
        start: Start date as a ``datetime.date``, ``numpy.datetime64``,
            or any value accepted by ``numpy.busday_count``.
        end: End date (exclusive) in the same format as ``start``.

    Returns:
        Integer count of business days in the half-open interval
        [start, end).
    """
    return int(np.busday_count(start, end))


def add_business_days(start, n):
    """Return the date that is exactly n business days after the start date.

    Uses ``numpy.busday_offset`` with ``roll="forward"`` so that if
    the computed landing date falls on a weekend it is advanced to the
    following Monday.

    Args:
        start: Starting date as a ``datetime.date``, ``numpy.datetime64``,
            or any value accepted by ``numpy.busday_offset``.
        n: Number of business days to advance. May be negative to move
            backwards.

    Returns:
        ``datetime.date`` representing the date n business days from
        ``start``.
    """
    result = np.busday_offset(start, n, roll="forward")
    return pd.Timestamp(result).date()


def year_fraction(start, end, basis="act365"):
    """Convert a date range to a decimal year fraction under a specified day-count convention.

    Supports three standard conventions used in commodity and fixed-income
    markets. The ``"bus252"`` basis uses ``business_days_between`` for the
    numerator, while the actual-day bases use the raw calendar day count.

    Args:
        start: Start date as a ``datetime.date`` or compatible type.
        end: End date as a ``datetime.date`` or compatible type.
        basis: Day-count convention identifier. Accepted values are:

            - ``"act365"`` -- Actual days divided by 365 (default).
            - ``"act360"`` -- Actual days divided by 360.
            - ``"bus252"`` -- Business days divided by 252.

    Returns:
        Float representing the elapsed time as a fraction of one year
        under the specified convention.

    Raises:
        ValueError: If ``basis`` is not one of the three recognised
            convention strings.
    """
    if basis == "act365":
        return (end - start).days / 365.0
    elif basis == "act360":
        return (end - start).days / 360.0
    elif basis == "bus252":
        return business_days_between(start, end) / 252.0
    else:
        raise ValueError(f"Unknown day-count basis: {basis}")


def generate_schedule(start, end, freq_months=1):
    """Generate a list of dates from start to end at a fixed monthly interval.

    Advances the current date by ``freq_months`` calendar months on each
    iteration, capping the day component at 28 to avoid month-end
    overflow. The resulting list includes both the start and end dates
    if they align with the schedule.

    Args:
        start: First date in the schedule as a ``datetime.date``.
        end: Last permissible date in the schedule as a ``datetime.date``.
            Dates are included up to and including this value.
        freq_months: Number of calendar months between successive dates.
            Defaults to 1 (monthly).

    Returns:
        List of ``datetime.date`` objects at approximately
        ``freq_months``-month intervals from ``start`` to ``end``
        inclusive.
    """
    dates = []
    current = start
    while current <= end:
        dates.append(current)
        month = current.month + freq_months
        year = current.year + (month - 1) // 12
        month = (month - 1) % 12 + 1
        day = min(current.day, 28)
        current = date(year, month, day)
    return dates


def is_business_day(d):
    """Determine whether a given date falls on a business day.

    Uses the standard Monday-to-Friday definition via
    ``numpy.is_busday`` with no holiday calendar.

    Args:
        d: Date to test as a ``datetime.date``, ``numpy.datetime64``,
            or any value accepted by ``numpy.is_busday``.

    Returns:
        ``True`` if ``d`` is a Monday through Friday (a business day),
        ``False`` if it falls on a Saturday or Sunday.
    """
    return bool(np.is_busday(d))


def next_business_day(d):
    """Return the given date if it is a business day, otherwise advance to the next one.

    Applies ``numpy.busday_offset`` with an offset of zero and
    ``roll="forward"``, which leaves weekday dates unchanged and moves
    Saturday to Monday and Sunday to Monday.

    Args:
        d: Input date as a ``datetime.date``, ``numpy.datetime64``,
            or any value accepted by ``numpy.busday_offset``.

    Returns:
        ``datetime.date`` equal to ``d`` if it is already a business
        day, or the nearest following business day otherwise.
    """
    result = np.busday_offset(d, 0, roll="forward")
    return pd.Timestamp(result).date()
