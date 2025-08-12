"""Tests for the interpolation utility functions in the cvx.simulator package.

This module contains tests for the interpolation utility functions, which are used
for filling missing values in time series data. The tests verify that the interpolate
function correctly fills missing values and that the valid function correctly
identifies series with no missing values in the middle.
"""

import numpy as np
import pandas as pd
import polars as pl

from cvxsimulator.utils import interpolate, valid


def test_interpolate_pandas() -> None:
    """Test that the interpolate function correctly fills missing values in a pandas Series.

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


def test_interpolate_polars() -> None:
    """Test that the interpolate function correctly fills missing values in a polars Series.

    This test creates a Series with null values at the beginning, middle, and end,
    applies the interpolate function to it, and verifies that the result is valid
    according to the valid function (i.e., it has no null values between the first
    and last valid indices).

    The test series has this pattern:
    [null, null, 2, 3, null, null, 4, 5, null, null, 6, null, null]

    After interpolation, the middle nulls should be filled, but the nulls at the
    beginning and end should remain.
    """
    data = [
        None,
        None,
        2,
        3,
        None,
        None,
        4,
        5,
        None,
        None,
        6,
        None,
        None,
    ]
    ts = pl.Series(data)
    a = interpolate(ts)
    assert valid(a)


def test_valid_pandas() -> None:
    """Test that the valid function correctly identifies pandas Series with no missing values in the middle."""
    # Series with NaNs only at beginning and end - should be valid
    ts1 = pd.Series([np.nan, 1, 2, 3, np.nan])
    assert valid(ts1)

    # Series with NaN in the middle - should not be valid
    ts2 = pd.Series([1, 2, np.nan, 4, 5])
    assert not valid(ts2)


def test_valid_polars() -> None:
    """Test that the valid function correctly identifies polars Series with no missing values in the middle."""
    # Series with nulls only at beginning and end - should be valid
    ts1 = pl.Series([None, 1, 2, 3, None])
    assert valid(ts1)

    # Series with null in the middle - should not be valid
    ts2 = pl.Series([1, 2, None, 4, 5])
    assert not valid(ts2)
