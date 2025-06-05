"""
Tests for the grid utility functions in the cvx.simulator package.

This module contains tests for the grid utility functions, which are used for
resampling time series data to coarser time grids. The tests verify that the
iron_frame function correctly resamples data according to specified rules.
"""

from __future__ import annotations

import pandas as pd
import polars as pl

from cvx.simulator.utils.grid import iron_frame


def test_iron_frame_pandas(prices: pd.DataFrame) -> None:
    """
    Test that the iron_frame function correctly resamples pandas DataFrame to month-end frequency.

    This test verifies that when applying the iron_frame function with a "ME"
    (month-end) rule to a prices DataFrame, there are exactly 27 days with
    significant price changes, which corresponds to the expected number of
    month-end rebalancing points.

    Parameters
    ----------
    prices : pd.DataFrame
        DataFrame of asset prices with dates as index and assets as columns
    """
    # compute the coarse grid
    frame = iron_frame(prices, rule="ME")

    # fish for days the prices in the frame are changing
    price_change = frame.diff().abs().sum(axis=1)
    # only take into account significant changes
    ts = price_change[price_change > 1.0]
    # verify that there are 27 days with significant changes
    assert len(ts) == 27


def test_iron_frame_polars(prices: pd.DataFrame) -> None:
    """
    Test that the iron_frame function correctly resamples polars DataFrame to month-end frequency.

    This test verifies that when applying the iron_frame function with a "ME"
    (month-end) rule to a prices DataFrame converted to polars, there are exactly 27 days with
    significant price changes, which corresponds to the expected number of
    month-end rebalancing points.

    Parameters
    ----------
    prices : pd.DataFrame
        DataFrame of asset prices with dates as index and assets as columns
    """
    # Reset the index to make the datetime a column
    prices_reset = prices.reset_index()

    # Convert pandas DataFrame to polars
    pl_prices = pl.from_pandas(prices_reset)

    # compute the coarse grid
    frame = iron_frame(pl_prices, rule="ME")

    # Convert back to pandas for easier testing
    pd_frame = frame.to_pandas()

    # Set the index back to the datetime column
    pd_frame = pd_frame.set_index("index")

    # fish for days the prices in the frame are changing
    price_change = pd_frame.diff().abs().sum(axis=1)
    # only take into account significant changes
    ts = price_change[price_change > 1.0]
    # verify that there are 27 days with significant changes
    assert len(ts) == 27
