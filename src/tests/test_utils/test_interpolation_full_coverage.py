"""Tests for achieving 100% coverage of the interpolation utility functions.

This module contains additional tests for the interpolation utility functions
to ensure 100% test coverage, focusing on functions and edge cases that are
not covered by the existing tests.
"""

import pandas as pd
import polars as pl

from cvxsimulator.utils import (
    interpolate,
    interpolate_df_pl,
    interpolate_pl,
    valid,
    valid_df_pl,
    valid_pl,
)


def test_valid_df_pl_all_valid():
    """Test that valid_df_pl returns True when all columns in the DataFrame are valid."""
    # Create a DataFrame with valid columns
    dframe = pl.DataFrame(
        {
            "A": [None, 1, 2, 3, None],  # Nulls only at beginning and end
            "B": [None, 2, 3, 4, None],  # Nulls only at beginning and end
            "C": [1, 2, 3, 4, 5],  # No nulls
        }
    )

    # Verify that valid_df_pl returns True
    assert valid_df_pl(dframe)


def test_valid_df_pl_one_invalid():
    """Test that valid_df_pl returns False when at least one column in the DataFrame is invalid."""
    # Create a DataFrame with one invalid column
    dframe = pl.DataFrame(
        {
            "A": [None, 1, 2, 3, None],  # Nulls only at beginning and end
            "B": [1, 2, None, 4, 5],  # Null in the middle
            "C": [1, 2, 3, 4, 5],  # No nulls
        }
    )

    # Verify that valid_df_pl returns False
    assert not valid_df_pl(dframe)


def test_interpolate_df_pl_with_different_types():
    """Test that interpolate_df_pl correctly handles a DataFrame with different types of columns."""
    # Create a DataFrame with different types of columns
    dframe = pl.DataFrame(
        {
            "int": [1, None, None, 4, 5],
            "float": [1.0, None, None, 4.0, 5.0],
            "str": ["a", None, None, "d", "e"],
            "bool": [True, None, None, False, True],
        }
    )

    # Apply interpolate_df_pl
    result = interpolate_df_pl(dframe)

    # Verify that all columns are valid
    for col in result.columns:
        assert valid_pl(result[col])

    # Verify that the values are filled correctly
    assert result["int"].to_list() == [1, 1, 1, 4, 5]
    assert result["float"].to_list() == [1.0, 1.0, 1.0, 4.0, 5.0]
    assert result["str"].to_list() == ["a", "a", "a", "d", "e"]
    assert result["bool"].to_list() == [True, True, True, False, True]


def test_interpolate_df_pl_empty_dataframe():
    """Test that interpolate_df_pl correctly handles an empty DataFrame."""
    # Create an empty DataFrame
    dframe = pl.DataFrame()

    # Apply interpolate_df_pl
    result = interpolate_df_pl(dframe)

    # Verify that the result is an empty DataFrame
    assert result.shape == (0, 0)


def test_valid_df_pl_empty_dataframe():
    """Test that valid_df_pl correctly handles an empty DataFrame."""
    # Create an empty DataFrame
    dframe = pl.DataFrame()

    # Verify that valid_df_pl returns True for an empty DataFrame
    assert valid_df_pl(dframe)


def test_interpolate_pl_with_one_value():
    """Test that interpolate_pl correctly handles a Series with only one non-null value."""
    # Create a Series with only one non-null value
    ts = pl.Series([None, 1, None])

    # Apply interpolate_pl
    result = interpolate_pl(ts)

    # Verify that the result is unchanged
    assert result.to_list() == [None, 1, None]


def test_valid_pl_with_all_nulls():
    """Test that valid_pl correctly handles a Series with all null values."""
    # Create a Series with all null values
    ts = pl.Series([None, None, None])

    # Verify that valid_pl returns True
    assert valid_pl(ts)


def test_interpolate_empty_pandas_series():
    """Test that interpolate correctly handles an empty pandas Series."""
    # Create an empty pandas Series
    ts = pd.Series([])

    # Apply interpolate
    result = interpolate(ts)

    # Verify that the result is an empty Series
    assert len(result) == 0


def test_valid_empty_pandas_series():
    """Test that valid correctly handles an empty pandas Series."""
    # Create an empty pandas Series
    ts = pd.Series([])

    # Verify that valid returns True for an empty Series
    assert valid(ts)


def test_interpolate_pl_empty_series():
    """Test that interpolate_pl correctly handles an empty polars Series."""
    # Create an empty polars Series
    ts = pl.Series([], dtype=pl.Float64)

    # Apply interpolate_pl
    result = interpolate_pl(ts)

    # Verify that the result is an empty Series
    assert len(result) == 0


def test_valid_pl_empty_series():
    """Test that valid_pl correctly handles an empty polars Series."""
    # Create an empty polars Series
    ts = pl.Series([], dtype=pl.Float64)

    # Verify that valid_pl returns True for an empty Series
    assert valid_pl(ts)
