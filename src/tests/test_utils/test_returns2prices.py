"""
Tests for the rescale utility functions in the cvx.simulator package.

This module contains tests for the rescale utility functions, which are used
for converting between returns and prices. The tests verify that the returns2prices
function correctly converts returns to prices while preserving the return structure.
"""

import pandas as pd
import polars as pl

from cvx.simulator.utils.rescale import returns2prices, returns2prices_pl


def test_prices(prices: pd.DataFrame) -> None:
    """
    Test that the returns2prices function correctly converts returns to prices.

    This test converts a DataFrame of prices to returns, then uses returns2prices
    to convert back to prices. It verifies that the resulting price DataFrame has
    the same index, columns, and percentage changes as the original prices, even
    though the absolute price levels will be different (normalized to start at 1.0).

    Parameters
    ----------
    prices : pd.DataFrame
        DataFrame of asset prices with dates as index and assets as columns,
        provided by a fixture
    """
    returns = prices.pct_change().fillna(0.0)

    prices_rescaled = returns2prices(returns)

    pd.testing.assert_index_equal(prices.index, prices_rescaled.index)
    pd.testing.assert_index_equal(prices.columns, prices_rescaled.columns)
    pd.testing.assert_frame_equal(prices.pct_change(), prices_rescaled.pct_change())


def test_returns2prices() -> None:
    """
    Test both pandas and polars implementations of returns2prices.

    This test creates identical DataFrames in pandas and polars,
    converts them to prices using their respective functions,
    and verifies that the results are equivalent.
    """
    # Test pandas implementation
    returns_pd = pd.DataFrame({"A": [0.05, 0.03, -0.02, 0.01], "B": [0.02, 0.01, 0.03, -0.01]})
    prices_pd = returns2prices(returns_pd)

    # Test polars implementation
    returns_pl = pl.DataFrame({"A": [0.05, 0.03, -0.02, 0.01], "B": [0.02, 0.01, 0.03, -0.01]})
    prices_pl = returns2prices_pl(returns_pl)

    # Convert polars result to pandas for comparison
    prices_pl_as_pd = prices_pl.to_pandas()

    # Verify that both implementations produce the same results
    pd.testing.assert_frame_equal(
        prices_pd.reset_index(drop=True), prices_pl_as_pd.reset_index(drop=True), check_dtype=False
    )


def test_returns2prices_pl() -> None:
    """
    Test the polars implementation of returns2prices.

    This test verifies that the polars implementation correctly
    converts returns to prices and maintains the expected properties.
    """
    # Create test data
    returns_pl = pl.DataFrame({"A": [0.05, 0.03, -0.02, 0.01], "B": [0.02, 0.01, 0.03, -0.01]})

    # Convert to prices
    prices_pl = returns2prices_pl(returns_pl)

    # Verify first row is 1.0 for all columns (starting prices)
    for col in prices_pl.columns:
        assert prices_pl[col][0] == 1.0

    # Verify the price changes match the returns
    # For each period: (price_t / price_{t-1}) - 1 should equal return_t
    for col in returns_pl.columns:
        # Calculate price changes
        price_changes = (prices_pl[col][1:] / prices_pl[col][:-1]) - 1

        # Compare with original returns (skip first row since there's no previous price)
        # We need to skip the first return when comparing with price changes
        original_returns = returns_pl[col][1:]

        # Convert to lists for easier comparison
        price_changes_list = price_changes.to_list()
        original_returns_list = original_returns.to_list()

        # Check each value with a small tolerance for floating point differences
        for i in range(len(price_changes_list)):
            assert abs(price_changes_list[i] - original_returns_list[i]) < 1e-10
