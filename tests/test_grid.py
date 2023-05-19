import pytest
import pandas as pd
import numpy as np

from cvx.simulator.builder import builder
from cvx.simulator.grid import resample_index, project_frame_to_grid


def test_resample_index(prices):
    x = resample_index(prices.index, rule="M")
    for t in x:
        assert t in prices.index

    assert len(x) == 28

    # first day in the middle of a month
    x = resample_index(prices.index[4:], rule="M")
    assert len(x) == 28
    assert x[0] == prices.index[4]


def test_project_frame_to_grid(prices):
    grid = resample_index(prices.index[4:], rule="M")
    frame = project_frame_to_grid(prices, grid=grid)
    a = frame.diff().sum(axis=1)
    assert a.tail(5).sum() == 0.0



def test_portfolio_resampling(prices):
    # construct a portfolio with only one asset
    b = builder(prices[["A"]])
    assert set(b.assets) == {"A"}

    # compute a grid and rebalance only on those particular days
    grid = resample_index(prices.index, rule="M")

    # change the position only at days that are in the grid
    for times, _ in b:
        if times[-1] in grid:
            b[times[-1]] = pd.Series({"A": np.random.rand(1)})
        else:
            # new position is the old position
            # This may look ineffective but we could use
            # trading costs for just holding short positions etc.
            b[times[-1]] = b[times[-2]]

    portfolio = b.build()
    print(portfolio.stocks)

    assert portfolio.stocks["A"].tail(10).std() == pytest.approx(0.0, abs=1e-12)
