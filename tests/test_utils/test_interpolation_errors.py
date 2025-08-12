"""Tests for error handling in the interpolation utility functions.

This module contains tests for the error handling in the interpolation utility functions,
which are used for filling missing values in time series data.
"""

import polars as pl
import pytest

from cvxsimulator.utils import interpolate, valid


def test_interpolate_invalid_type():
    """Test that interpolate raises TypeError when input is not a pandas or polars Series.

    This test calls interpolate with a list and verifies that it raises a TypeError.
    """
    # Call interpolate with a list
    with pytest.raises(TypeError, match="Expected pd.Series or pl.Series, got <class 'list'>"):
        interpolate([1, 2, 3])


def test_valid_invalid_type():
    """Test that valid raises TypeError when input is not a pandas or polars Series.

    This test calls valid with a list and verifies that it raises a TypeError.
    """
    # Call valid with a list
    with pytest.raises(TypeError, match="Expected pd.Series or pl.Series, got <class 'list'>"):
        valid([1, 2, 3])


def test_interpolate_polars_with_nulls():
    """Test that interpolate correctly handles a polars Series with nulls.

    This test creates a polars Series with nulls in the middle, applies interpolate,
    and verifies that the nulls are filled correctly.
    """
    # Create a polars Series with nulls in the middle
    ts = pl.Series([1.0, None, None, 4.0])

    # Apply interpolate
    result = interpolate(ts)

    # Verify that the nulls are filled correctly
    assert result.to_list() == [1.0, 1.0, 1.0, 4.0]


def test_interpolate_polars_all_nulls():
    """Test that interpolate correctly handles a polars Series with all nulls.

    This test creates a polars Series with all null values, applies interpolate,
    and verifies that the result is unchanged.
    """
    # Create a polars Series with all null values
    ts = pl.Series([None, None, None])

    # Apply interpolate
    result = interpolate(ts)

    # Verify that the result is unchanged
    assert result.to_list() == [None, None, None]
