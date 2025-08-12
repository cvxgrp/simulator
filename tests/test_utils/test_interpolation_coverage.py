"""Tests for the interpolation utility functions to achieve 100% coverage.

This module contains additional tests for the interpolation utility functions
to ensure 100% test coverage, including tests for edge cases and tests with
date columns.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import polars as pl

from cvxsimulator.utils import (
    interpolate,
    interpolate_df_pl,
    valid,
    valid_pl,
)


def test_interpolate_all_nan() -> None:
    """Test that the interpolate function correctly handles a pandas Series with all NaN values.

    This test creates a Series with all NaN values, applies the interpolate function to it,
    and verifies that the result is unchanged.
    """
    # Create a pandas Series with all NaN values
    ts = pd.Series([np.nan, np.nan, np.nan])

    # Apply interpolate
    result = interpolate(ts)

    # Verify that the result is unchanged
    pd.testing.assert_series_equal(result, ts)


def test_valid_with_polars_series() -> None:
    """Test that the valid function correctly handles a polars Series.

    This test creates a polars Series, passes it to the valid function,
    and verifies that the result is the same as calling valid_pl directly.
    """
    # Create a polars Series with nulls only at beginning and end
    ts1 = pl.Series([None, 1, 2, 3, None])

    # Create a polars Series with null in the middle
    ts2 = pl.Series([1, 2, None, 4, 5])

    # Verify that valid calls valid_pl for polars Series
    assert valid(ts1) == valid_pl(ts1)
    assert valid(ts2) == valid_pl(ts2)


def test_valid_pl_with_zero_or_one_non_null() -> None:
    """Test that the valid_pl function correctly handles a polars Series with 0 or 1 non-null values.

    This test creates polars Series with 0 or 1 non-null values, applies the valid_pl function to them,
    and verifies that the result is True.
    """
    # Create a polars Series with all null values
    ts1 = pl.Series([None, None, None])

    # Create a polars Series with one non-null value
    ts2 = pl.Series([None, 1, None])

    # Verify that valid_pl returns True for both cases
    assert valid_pl(ts1)
    assert valid_pl(ts2)


def test_interpolate_with_date_index() -> None:
    """Test that the interpolate function correctly handles a pandas Series with a date index.

    This test creates a Series with a date index and NaN values in the middle,
    applies the interpolate function to it, and verifies that the result is valid.
    """
    # Create a date index
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(5)]

    # Create a pandas Series with a date index and NaN values in the middle
    ts = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0], index=dates)

    # Apply interpolate
    result = interpolate(ts)

    # Verify that the result is valid
    assert valid(result)

    # Verify that the NaN value was filled correctly
    assert result[dates[2]] == 2.0  # Should be filled with the previous value


def test_interpolate_df_pl_with_date_column() -> None:
    """Test that the interpolate_df_pl function correctly handles a polars DataFrame with a date column.

    This test creates a DataFrame with a date column and null values in other columns,
    applies the interpolate_df_pl function to it, and verifies that the result is valid.
    """
    # Create a date column
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(5)]

    # Create a polars DataFrame with a date column and null values in other columns
    dframe = pl.DataFrame({"date": dates, "A": [1.0, 2.0, None, 4.0, 5.0], "B": [None, 2.0, 3.0, None, 5.0]})

    # Apply interpolate_df_pl
    result = interpolate_df_pl(dframe)

    # Verify that the date column is unchanged
    assert result["date"].to_list() == dates

    # Verify that the other columns are valid
    assert valid_pl(result["A"])
    assert valid_pl(result["B"])

    # Verify that the null values were filled correctly
    assert result["A"].to_list() == [1.0, 2.0, 2.0, 4.0, 5.0]
    assert result["B"].to_list() == [None, 2.0, 3.0, 3.0, 5.0]
