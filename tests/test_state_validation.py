"""Tests for the validation checks in the State class.

This module contains tests for the validation checks in the State class,
specifically testing error conditions in the weights property.
"""

from unittest.mock import PropertyMock, patch

import pandas as pd
import pytest

from cvxsimulator import State


def test_nav_not_equal_aum():
    """Test that State.weights raises an error when nav != aum.

    This test verifies that the State.weights property correctly raises a ValueError
    when the net asset value (nav) is not equal to the assets under management (aum).
    """
    # Create a State instance
    state = State()

    # Set up prices and positions
    prices = pd.Series({"A": 100, "B": 200})
    position = pd.Series({"A": 1, "B": 1})

    # Set the prices and position
    state.prices = prices
    state.position = position

    # Set aum to 400
    state.aum = 400

    # Patch the nav property to return a different value (300) than aum (400)
    with patch.object(State, "nav", new_callable=PropertyMock) as mock_nav:
        mock_nav.return_value = 300.0

        # Verify that accessing the weights property raises a ValueError
        with pytest.raises(ValueError, match="300.0 != 400"):
            _ = state.weights
