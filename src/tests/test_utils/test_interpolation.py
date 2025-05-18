"""
Tests for the interpolation utility functions in the cvx.simulator package.

This module contains tests for the interpolation utility functions, which are used
for filling missing values in time series data. The tests verify that the interpolate
function correctly fills missing values and that the valid function correctly
identifies series with no missing values in the middle.
"""

import numpy as np
import pandas as pd

from cvx.simulator.utils.interpolation import interpolate, valid


def test_interpolate() -> None:
    """
    Test that the interpolate function correctly fills missing values.

    This test creates a Series with NaN values at the beginning, middle, and end,
    applies the interpolate function to it, and verifies that the result is valid
    according to the valid function (i.e., it has no NaN values between the first
    and last valid indices).

    The test series has this pattern:
    [NaN, NaN, 2, 3, NaN, NaN, 4, 5, NaN, NaN, 6, NaN, NaN]

    After interpolation, the middle NaNs should be filled, but the NaNs at the
    beginning and end should remain.
    """
    ts = pd.Series(
        data=[
            np.nan,
            np.nan,
            2,
            3,
            np.nan,
            np.nan,
            4,
            5,
            np.nan,
            np.nan,
            6,
            np.nan,
            np.nan,
        ]
    )
    a = interpolate(ts)
    assert valid(a)
