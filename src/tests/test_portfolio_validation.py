"""Tests for the validation checks in the Portfolio class.

This module contains tests for the validation checks in the Portfolio class's
__post_init__ method, specifically testing error conditions for non-monotonic
and non-unique indices, as well as mismatches between prices and units.
"""

import pandas as pd
import pytest

from cvxsimulator import Portfolio


def test_non_unique_prices_index():
    """Test that Portfolio raises an error for non-unique prices index.

    This test verifies that the Portfolio correctly raises a ValueError when
    initialized with a prices DataFrame that has duplicate index values.
    """
    # Create DataFrames with duplicate index in prices
    dates1 = pd.DatetimeIndex(["2020-01-01", "2020-01-02", "2020-01-02"])
    dates2 = pd.DatetimeIndex(["2020-01-01", "2020-01-02"])
    prices = pd.DataFrame(index=dates1, data={"A": [1, 2, 3]})
    units = pd.DataFrame(index=dates2, data={"A": [10, 20]})

    # Verify that initializing a Portfolio with these DataFrames raises a ValueError
    with pytest.raises(ValueError, match="`prices` index must be unique"):
        Portfolio(prices=prices, units=units, aum=1000)


def test_non_monotonic_units_index():
    """Test that Portfolio raises an error for non-monotonic units index.

    This test verifies that the Portfolio correctly raises a ValueError when
    initialized with a units DataFrame that has a non-monotonically increasing index.
    """
    # Create DataFrames with non-monotonic index in units
    dates1 = pd.DatetimeIndex(["2020-01-01", "2020-01-02", "2020-01-03"])
    dates2 = pd.DatetimeIndex(["2020-01-01", "2020-01-03", "2020-01-02"])
    prices = pd.DataFrame(index=dates1, data={"A": [1, 2, 3]})
    units = pd.DataFrame(index=dates2, data={"A": [10, 20, 30]})

    # Verify that initializing a Portfolio with these DataFrames raises a ValueError
    with pytest.raises(ValueError, match="`units` index must be monotonic increasing"):
        Portfolio(prices=prices, units=units, aum=1000)


def test_non_unique_units_index():
    """Test that Portfolio raises an error for non-unique units index.

    This test verifies that the Portfolio correctly raises a ValueError when
    initialized with a units DataFrame that has duplicate index values.
    """
    # Create DataFrames with duplicate index in units
    dates1 = pd.DatetimeIndex(["2020-01-01", "2020-01-02", "2020-01-03"])
    dates2 = pd.DatetimeIndex(["2020-01-01", "2020-01-02", "2020-01-02"])
    prices = pd.DataFrame(index=dates1, data={"A": [1, 2, 3]})
    units = pd.DataFrame(index=dates2, data={"A": [10, 20, 30]})

    # Verify that initializing a Portfolio with these DataFrames raises a ValueError
    with pytest.raises(ValueError, match="`units` index must be unique"):
        Portfolio(prices=prices, units=units, aum=1000)


def test_units_index_not_in_prices():
    """Test that Portfolio raises an error when units index contains dates not in prices.

    This test verifies that the Portfolio correctly raises a ValueError when
    initialized with a units DataFrame that has index values not present in the prices DataFrame.
    """
    # Create DataFrames with units index containing dates not in prices
    dates1 = pd.DatetimeIndex(["2020-01-01", "2020-01-02"])
    dates2 = pd.DatetimeIndex(["2020-01-01", "2020-01-03"])
    prices = pd.DataFrame(index=dates1, data={"A": [1, 2]})
    units = pd.DataFrame(index=dates2, data={"A": [10, 30]})

    # Verify that initializing a Portfolio with these DataFrames raises a ValueError
    with pytest.raises(ValueError, match="`units` index contains dates not present in `prices`"):
        Portfolio(prices=prices, units=units, aum=1000)


def test_units_assets_not_in_prices():
    """Test that Portfolio raises an error when units contains assets not in prices.

    This test verifies that the Portfolio correctly raises a ValueError when
    initialized with a units DataFrame that has columns not present in the prices DataFrame.
    """
    # Create DataFrames with units containing assets not in prices
    dates = pd.DatetimeIndex(["2020-01-01", "2020-01-02"])
    prices = pd.DataFrame(index=dates, data={"A": [1, 2]})
    units = pd.DataFrame(index=dates, data={"A": [10, 20], "B": [30, 40]})

    # Verify that initializing a Portfolio with these DataFrames raises a ValueError
    with pytest.raises(ValueError, match="`units` contains assets not present in `prices`"):
        Portfolio(prices=prices, units=units, aum=1000)
