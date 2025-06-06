import pandas as pd
import polars as pl

from cvx.simulator.builder import polars2pandas


def test_iteration(prices_pl: pl.DataFrame):
    # iterate
    for row in prices_pl.rows(named=True):
        row


def test_index(prices_pl: pl.DataFrame):
    index = prices_pl["date"].to_list()
    print(index)


def test_index_history(prices_pl: pl.DataFrame):
    index = prices_pl["date"].to_list()

    for i in range(len(index)):
        up_to_now = index[: i + 1]
        print(up_to_now)


def test_index_window(prices_pl: pl.DataFrame):
    index = prices_pl["date"].to_list()

    window_size = 3

    for i in range(len(index)):
        window = index[max(0, i - window_size + 1) : i + 1]
        print(window)


def test_polars2pandas(prices_pl: pl.DataFrame, prices: pd.DataFrame):
    """
    Test the polars2pandas function.

    This test verifies that the polars2pandas function correctly converts a Polars DataFrame
    to a Pandas DataFrame with the 'date' column set as the index.

    Args:
        prices_pl: A Polars DataFrame fixture containing price data
        prices: A Pandas DataFrame fixture containing the same price data
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
