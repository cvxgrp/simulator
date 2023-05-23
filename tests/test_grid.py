import pytest
import pandas as pd
import numpy as np

from cvx.simulator.builder import builder
from cvx.simulator.grid import resample_index, project_frame_to_grid


def test_resample_index(prices):
    """
    Test the resample index function
    :param prices: the prices frame (fixture)
    """
    # first day using a monthly rule
    grid = resample_index(prices.index, rule="M")

    # check that all the dates in the grid are in the prices index
    for t in grid:
        assert t in prices.index

    # verify 28 days are in the grid
    assert len(grid) == 28

    # verify that the first day of the grid is the first day of the prices
    grid = resample_index(prices.index[4:], rule="M")
    assert len(grid) == 28
    assert grid[0] == prices.index[4]


def test_project_frame_to_grid(prices):
    """
    Test the project frame to grid function
    :param prices:
    :return:
    """
    # compute the coarse grid
    grid = resample_index(prices.index[4:], rule="M")
    # the frame is only changing rows on days of the grid
    frame = project_frame_to_grid(prices, grid=grid)

    # fish for days the prices in the frame are changing
    price_change = frame.diff().abs().sum(axis=1)
    # only take into account significant changes
    ts = price_change[price_change > 1.0]
    # verify that the days are the grid
    assert set(ts.index) == set(grid[1:])




def test_portfolio_resampling(prices):
    """
    Test the portfolio resampling
    :param prices: the prices frame (fixture)
    """
    # build a portfolio with only one asset
    b = builder(prices[["A"]])

    # compute a grid and rebalance only on those particular days
    grid = resample_index(prices.index, rule="M")

    # change the position only at days that are in the grid
    for times, _ in b:
        # if the last day is in the grid
        if times[-1] in grid:
            # change the position
            b[times[-1]] = pd.Series({"A": np.random.rand(1)})
        else:
            # new position is the old position
            # This may look ineffective but we could use
            # trading costs for just holding short positions etc.
            b[times[-1]] = b[times[-2]]

    # build the portfolio
    portfolio = b.build()

    assert portfolio.stocks["A"].tail(10).std() == pytest.approx(0.0, abs=1e-12)
