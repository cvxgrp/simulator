import numpy as np
import pandas as pd
import pytest

from cvx.simulator import FuturesBuilder


@pytest.fixture()
def builder(prices):
    # create some returns from prices
    returns = prices.pct_change().fillna(0.0)
    # create the builder from the returns
    builder = FuturesBuilder.from_returns(returns)
    return builder


def test_init_from_returns(builder, prices):
    # the rescaled prices start at 1.0...
    pd.testing.assert_series_equal(builder.prices["A"] * 1673.78, prices["A"])


def test_set_cashposition(builder):
    """
    Test that we can set the cashposition in the iteration over states
    """
    for time, state in builder:
        # set the cashposition for each future to 100k
        builder.cashposition = 1e5 * np.ones(len(state.assets))

    portfolio = builder.build()
    pd.testing.assert_frame_equal(
        portfolio.cashposition,
        pd.DataFrame(index=portfolio.index, columns=portfolio.assets, data=1e5),
    )

    # check the computation of our nav is correct
    profit = (portfolio.cashposition.shift(1) * portfolio.returns).sum(axis=1)
    pd.testing.assert_series_equal(portfolio.nav, profit.cumsum() + portfolio.aum)
