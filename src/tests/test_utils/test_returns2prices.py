"""
Tests for the rescale utility functions in the cvx.simulator package.

This module contains tests for the rescale utility functions, which are used
for converting between returns and prices. The tests verify that the returns2prices
function correctly converts returns to prices while preserving the return structure.
"""

import pandas as pd

from cvx.simulator.utils.rescale import returns2prices


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
