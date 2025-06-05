"""
Tests for the interpolation utility functions in the cvx.simulator package.

This module contains tests for the interpolation utility functions, which are used
for filling missing values in time series data. The tests verify that the interpolate
function correctly fills missing values and that the valid function correctly
identifies series with no missing values in the middle.
"""

import numpy as np
import pandas as pd
import polars as pl

from cvx.simulator.utils.interpolation import (
    interpolate,
    interpolate_df_pl,
    interpolate_pl,
    valid,
    valid_df_pl,
    valid_pl,
)


def test_interpolate_pandas() -> None:
    """
    Test that the interpolate function correctly fills missing values in a pandas Series.

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
    """
    Test that the interpolate function correctly fills missing values in a polars Series.

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
    a = interpolate_pl(ts)
    assert valid_pl(a)


def test_valid_pandas() -> None:
    """
    Test that the valid function correctly identifies pandas Series with no missing values in the middle.
    """
    # Series with NaNs only at beginning and end - should be valid
    ts1 = pd.Series([np.nan, 1, 2, 3, np.nan])
    assert valid(ts1)

    # Series with NaN in the middle - should not be valid
    ts2 = pd.Series([1, 2, np.nan, 4, 5])
    assert not valid(ts2)


def test_valid_polars() -> None:
    """
    Test that the valid function correctly identifies polars Series with no missing values in the middle.
    """
    # Series with nulls only at beginning and end - should be valid
    ts1 = pl.Series([None, 1, 2, 3, None])
    assert valid_pl(ts1)

    # Series with null in the middle - should not be valid
    ts2 = pl.Series([1, 2, None, 4, 5])
    assert not valid_pl(ts2)


def test_interpolate_df_pl() -> None:
    """
    Test that the interpolate_df_pl function correctly fills missing values in a polars DataFrame.

    This test creates a DataFrame with null values at the beginning, middle, and end of different columns,
    applies the interpolate_df_pl function to it, and verifies that the result has no null values
    between the first and last valid indices for each column.
    """
    # Create a test DataFrame with nulls in different positions
    df = pl.DataFrame(
        {
            "A": [1, None, None, 4, 5],  # Nulls in the middle
            "B": [None, 2, None, 4, None],  # Nulls at beginning, middle, and end
            "C": [None, None, 3, 4, 5],  # Nulls at beginning
            "D": [1, 2, 3, None, None],  # Nulls at end
        }
    )

    # Apply interpolate_df_pl
    result = interpolate_df_pl(df)

    # Verify each column is valid
    for col in result.columns:
        assert valid_pl(result[col])

    # Check specific values
    # Column A: [1, 1, 1, 4, 5]
    assert result["A"].to_list() == [1, 1, 1, 4, 5]

    # Column B: [None, 2, 2, 4, None]
    assert result["B"].to_list()[0] is None
    assert result["B"].to_list()[1:4] == [2, 2, 4]
    assert result["B"].to_list()[4] is None

    # Column C: [None, None, 3, 4, 5]
    assert result["C"].to_list()[0:2] == [None, None]
    assert result["C"].to_list()[2:5] == [3, 4, 5]

    # Column D: [1, 2, 3, None, None]
    assert result["D"].to_list()[0:3] == [1, 2, 3]
    assert result["D"].to_list()[3:5] == [None, None]


def test_valid_df_pl() -> None:
    """
    Test that the valid_df_pl function correctly identifies polars DataFrames with no missing values in the middle.
    """
    # DataFrame with nulls only at beginning and end of each column - should be valid
    df1 = pl.DataFrame(
        {
            "A": [None, 1, 2, 3, None],  # Nulls only at beginning and end
            "B": [None, 2, 3, 4, None],  # Nulls only at beginning and end
        }
    )
    assert valid_df_pl(df1)

    # DataFrame with null in the middle of one column - should not be valid
    df2 = pl.DataFrame(
        {
            "A": [1, 2, None, 4, 5],  # Null in the middle
            "B": [1, 2, 3, 4, 5],  # No nulls
        }
    )
    assert not valid_df_pl(df2)

    # DataFrame with nulls only at beginning and end of one column and no nulls in another - should be valid
    df3 = pl.DataFrame(
        {
            "A": [None, 1, 2, 3, None],  # Nulls only at beginning and end
            "B": [1, 2, 3, 4, 5],  # No nulls
        }
    )
    assert valid_df_pl(df3)
