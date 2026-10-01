"""Tests for the validation checks in the State class.

This module contains tests for the validation checks in the State class,
specifically testing error conditions in the weights property.
"""

import pandas as pd
import pytest

from cvx.simulator import State


@pytest.mark.parametrize("aum", [0.0, float("nan"), float("inf")])
def test_weights_undefined_for_degenerate_nav(aum):
    """Test that State.weights raises an error when the NAV is zero or not finite.

    Weights are fractions of NAV, so a zero or non-finite NAV leaves them
    undefined. The default State has an AUM of 0.0, which is the case a
    caller most easily reaches by forgetting to set it.
    """
    state = State()
    state.prices = pd.Series({"A": 100.0, "B": 200.0})
    state.position = pd.Series({"A": 1.0, "B": 1.0})
    state.aum = aum

    with pytest.raises(ValueError, match="weights are undefined for a NAV of"):
        _ = state.weights
