import pandas as pd

from cvx.simulator.futures.builder import FuturesBuilder
from cvx.simulator.futures.portfolio import FuturesPortfolio


def test_portfolio_cumulated(prices):
    builder = FuturesBuilder(prices=prices[["A", "B"]], aum=1e6)

    for _, _ in builder:
        # hold one share in both assets
        builder.cashposition = [1e5, 4e5]

    portfolio = FuturesPortfolio(
        aum=1e6, prices=prices[["A", "B"]], units=builder.units
    )

    # shift the cashposition forward such that we multiply the returns with the previous cashposition
    profit = (portfolio.cashposition.shift(1) * portfolio.returns).sum(axis=1)
    pd.testing.assert_series_equal(portfolio.nav, profit.cumsum() + portfolio.aum)


def test_from_cash_position_prices(prices):
    cashpos = pd.DataFrame(index=prices.index, columns=prices.columns, data=1e5)

    portfolio = FuturesPortfolio.from_cashpos_prices(
        prices=prices, cashposition=cashpos, aum=1e6
    )

    profit = (portfolio.cashposition.shift(1) * portfolio.returns).sum(axis=1)
    pd.testing.assert_series_equal(portfolio.nav, profit.cumsum() + portfolio.aum)


def test_from_cash_returns(prices):
    returns = prices.pct_change().fillna(0.0)
    cashpos = pd.DataFrame(index=prices.index, columns=prices.columns, data=1e5)

    portfolio = FuturesPortfolio.from_cashpos_returns(
        returns=returns, cashposition=cashpos, aum=1e6
    )

    profit = (portfolio.cashposition.shift(1) * portfolio.returns).sum(axis=1)
    pd.testing.assert_series_equal(portfolio.nav, profit.cumsum() + portfolio.aum)
