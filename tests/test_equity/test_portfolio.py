import pandas as pd
import pytest

from cvx.simulator.builder import Builder


def test_portfolio(prices):
    builder = Builder(prices=prices[["A", "B"]], initial_cash=1e6)

    for _, _ in builder:
        # hold one share in both assets
        builder.position = [1, 1]

    portfolio = builder.build()
    pd.testing.assert_series_equal(
        portfolio.nav,
        prices[["A", "B"]].sum(axis=1) + 1e6 - prices[["A", "B"]].iloc[0].sum(),
    )

    portfolio.cashflow.iloc[0] = pytest.approx(-975014.24)
