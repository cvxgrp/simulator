"""
Tests for the grid utility functions in the cvx.simulator package.

This module contains tests for the grid utility functions, which are used for
resampling time series data to coarser time grids. The tests verify that the
iron_frame function correctly resamples data according to specified rules.
"""

from __future__ import annotations

import pandas as pd

from cvx.simulator.utils.grid import iron_frame


def test_iron_frame(prices: pd.DataFrame) -> None:
    """
    Test that the iron_frame function correctly resamples data to month-end frequency.

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
