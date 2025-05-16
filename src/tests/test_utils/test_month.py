"""
Tests for the month utility functions in the cvx.simulator package.

This module contains tests for the month utility functions, which are used
for creating performance tables that show returns by month and year. The tests
verify that the monthlytable function correctly creates tables with both
compounded and cumulative aggregation methods.
"""

import pandas as pd

from cvx.simulator.utils.month import Aggregate, monthlytable


def test_month_compounded(nav: pd.Series) -> None:
    """
    Test that the monthlytable function works with compounded returns.

    This test creates a monthly performance table using the COMPOUND aggregation
    method, which aggregates returns by multiplying (1+r) values. The test prints
    the resulting table for visual inspection.

    Parameters
    ----------
    nav : pd.Series
        Series of portfolio NAV values over time, provided by a fixture
    """
    table = monthlytable(nav.pct_change().fillna(0.0), Aggregate.COMPOUND)
    print(table)


def test_month_cumulative(nav: pd.Series) -> None:
    """
    Test that the monthlytable function works with cumulative returns.

    This test creates a monthly performance table using the CUMULATIVE aggregation
    method, which aggregates returns by simple addition. The test prints the
    resulting table for visual inspection.

    Parameters
    ----------
    nav : pd.Series
        Series of portfolio NAV values over time, provided by a fixture
    """
    table = monthlytable(nav.pct_change().fillna(0.0), Aggregate.CUMULATIVE)
    print(table)
