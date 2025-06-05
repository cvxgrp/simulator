"""
Tests for the rescale utility functions in the cvx.simulator package.

This module contains tests for the internal helper functions in the rescale module,
which are used for converting between returns and prices.
"""

import polars as pl

from cvx.simulator.utils.rescale import _rescale_pl


def test_rescale_pl():
    """
    Test that the _rescale_pl function correctly converts a polars Series of returns to prices.

    This test creates a Series with returns, applies the _rescale_pl function to it,
    and verifies that the result starts at 1.0 and has the correct values.
    """
    # Create a polars Series of returns
    returns = pl.Series([0.05, 0.03, -0.02, 0.01])

    # Apply the _rescale_pl function
    prices = _rescale_pl(returns)

    # Verify the first value is 1.0
    assert prices[0] == 1.0

    # Verify the values match the expected values
    # The actual calculation in _rescale_pl is (1+r).cum_prod() / first
    expected_values = [1.0, 1.03, 1.0094, 1.0194940000000001]

    # Get the actual values
    actual_values = prices.to_list()

    # Check length
    assert len(actual_values) == len(expected_values)

    # Check values with tolerance
    for i, (expected, actual) in enumerate(zip(expected_values, actual_values)):
        assert abs(actual - expected) < 1e-6, f"Value at index {i} differs: expected {expected}, got {actual}"
