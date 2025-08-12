"""Tests for the validation checks in the Builder class.

This module contains tests for the validation checks in the Builder class's
__post_init__ method, specifically testing error conditions for non-monotonic
and non-unique indices.
"""

import pandas as pd
import pytest

from cvxsimulator import Builder


def test_non_monotonic_index():
    """Test that Builder raises an error for non-monotonic index.

    This test verifies that the Builder correctly raises a ValueError when
    initialized with a prices DataFrame that has a non-monotonically increasing index.
    """
    # Create a DataFrame with a non-monotonic index
    dates = pd.DatetimeIndex(["2020-01-01", "2020-01-03", "2020-01-02"])
    prices = pd.DataFrame(index=dates, data={"A": [1, 2, 3]})

    # Verify that initializing a Builder with this DataFrame raises a ValueError
    with pytest.raises(ValueError, match="Index must be monotonically increasing"):
        Builder(prices=prices)


def test_non_unique_index():
    """Test that Builder raises an error for non-unique index.

    This test verifies that the Builder correctly raises a ValueError when
    initialized with a prices DataFrame that has duplicate index values.
    """
    # Create a DataFrame with duplicate index values
    dates = pd.DatetimeIndex(["2020-01-01", "2020-01-02", "2020-01-02"])
    prices = pd.DataFrame(index=dates, data={"A": [1, 2, 3]})

    # Verify that initializing a Builder with this DataFrame raises a ValueError
    with pytest.raises(ValueError, match="Index must have unique values"):
        Builder(prices=prices)
