import pandas as pd

from cvx.simulator.utils.rescale import returns2prices


def test_prices(prices):
    returns = prices.pct_change().fillna(0.0)

    prices_rescaled = returns2prices(returns)

    pd.testing.assert_index_equal(prices.index, prices_rescaled.index)
    pd.testing.assert_index_equal(prices.columns, prices_rescaled.columns)
    pd.testing.assert_frame_equal(prices.pct_change(), prices_rescaled.pct_change())
