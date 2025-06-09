"""Tests for Polars DataFrame functionality in the cvx.simulator package.

This module contains tests for working with Polars DataFrames, including iteration,
index extraction, window operations, and conversion between Polars and Pandas.
These tests verify that the package can correctly handle Polars DataFrames as an
alternative to Pandas DataFrames.
"""

import pandas as pd
import polars as pl

from cvxsimulator.builder import polars2pandas


def test_iteration(prices_pl: pl.DataFrame):
    """Test iteration over rows in a Polars DataFrame.

    This test verifies that a Polars DataFrame can be iterated over row by row
    using the rows() method with named=True.

    Parameters
    ----------
    prices_pl : pl.DataFrame
        A Polars DataFrame fixture containing price data

    """
    # iterate
    for row in prices_pl.rows(named=True):
        print(row)


def test_index(prices_pl: pl.DataFrame):
    """Test extraction of the date index from a Polars DataFrame.

    This test verifies that the date column can be extracted from a Polars DataFrame
    and converted to a list.

    Parameters
    ----------
    prices_pl : pl.DataFrame
        A Polars DataFrame fixture containing price data with a 'date' column

    """
    index = prices_pl["date"].to_list()
    print(index)


def test_index_history(prices_pl: pl.DataFrame):
    """Test building a history of dates from a Polars DataFrame.

    This test verifies that we can extract the date column from a Polars DataFrame,
    convert it to a list, and then build a growing history of dates by slicing
    the list up to each point in time.

    Parameters
    ----------
    prices_pl : pl.DataFrame
        A Polars DataFrame fixture containing price data with a 'date' column

    """
    index = prices_pl["date"].to_list()

    for i in range(len(index)):
        up_to_now = index[: i + 1]
        print(up_to_now)


def test_index_window(prices_pl: pl.DataFrame):
    """Test creating a sliding window of dates from a Polars DataFrame.

    This test verifies that we can extract the date column from a Polars DataFrame,
    convert it to a list, and then create a sliding window of dates of a fixed size
    that moves forward in time.

    Parameters
    ----------
    prices_pl : pl.DataFrame
        A Polars DataFrame fixture containing price data with a 'date' column

    """
    index = prices_pl["date"].to_list()

    window_size = 3

    for i in range(len(index)):
        window = index[max(0, i - window_size + 1) : i + 1]
        print(window)


def test_polars2pandas(prices_pl: pl.DataFrame, prices: pd.DataFrame):
    """Test the polars2pandas function.

    This test verifies that the polars2pandas function correctly converts a Polars DataFrame
    to a Pandas DataFrame with the 'date' column set as the index.

    Parameters
    ----------
    prices_pl : pl.DataFrame
        A Polars DataFrame fixture containing price data
    prices : pd.DataFrame
        A Pandas DataFrame fixture containing the same price data

    """
    # Convert the Polars DataFrame to a Pandas DataFrame
    result = polars2pandas(prices_pl)

    # Check that the result is a Pandas DataFrame
    assert isinstance(result, pd.DataFrame)

    # Check that the index name is 'date'
    assert result.index.name == "date"

    # Check that the index is a DatetimeIndex
    assert isinstance(result.index, pd.DatetimeIndex)

    # Check that the columns match (excluding the 'date' column from the Polars DataFrame)
    assert set(result.columns) == set(prices.columns)

    # Check that the data values match
    # Note: We're comparing the values directly, not the DataFrames themselves,
    # to avoid issues with index types, etc.
    for col in result.columns:
        assert result[col].equals(prices[col])
