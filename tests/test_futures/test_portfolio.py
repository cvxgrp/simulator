import pandas as pd

from cvx.simulator.equity.builder import EquityBuilder
from cvx.simulator.futures.portfolio import FuturesPortfolio


def test_portfolio_cumulated(prices):
    builder = EquityBuilder(prices=prices[["A", "B"]], initial_cash=1e6)

    for _, _ in builder:
        # hold one share in both assets
        builder.cashposition = [1e5, 4e5]

    portfolio = FuturesPortfolio(
        aum=1e6, prices=prices[["A", "B"]], units=builder.units
    )

    # shift the cashposition forward such that we multiply the returns with the previous cashposition
    profit = (portfolio.cashposition.shift(1) * portfolio.returns).sum(axis=1)
    pd.testing.assert_series_equal(portfolio.nav, profit.cumsum() + portfolio.aum)
