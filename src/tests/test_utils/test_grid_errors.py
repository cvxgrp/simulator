"""
Tests for error handling in the grid utility functions.

This module contains tests for the error handling in the grid utility functions,
which are used for resampling time series data to coarser time grids.
"""

import polars as pl
import pytest

from cvx.simulator.utils.grid import iron_frame


def test_iron_frame_polars_no_datetime():
    """
    Test that iron_frame raises ValueError when a polars DataFrame has no datetime column.

    This test creates a polars DataFrame with no datetime columns and verifies that
    calling iron_frame on it raises a ValueError.
    """
    # Create a polars DataFrame with no datetime columns
    df = pl.DataFrame({"A": [1, 2, 3, 4], "B": [5, 6, 7, 8]})

    # Verify that calling iron_frame raises a ValueError
    with pytest.raises(ValueError, match="Polars DataFrame must have at least one datetime column to use as index"):
        iron_frame(df, rule="ME")


def test_iron_frame_invalid_type():
    """
    Test that iron_frame raises TypeError when input is not a pandas or polars DataFrame.

    This test calls iron_frame with a list and verifies that it raises a TypeError.
    """
    # Call iron_frame with a list
    with pytest.raises(TypeError, match="Expected pd.DataFrame or pl.DataFrame, got <class 'list'>"):
        iron_frame([1, 2, 3], rule="ME")
