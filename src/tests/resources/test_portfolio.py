"""
Tests for the Portfolio class factory methods in the cvx.simulator package.

This module contains tests for the Portfolio class factory methods, which are used
for creating portfolios from different types of input data. The tests verify that
portfolios can be correctly created from cash positions and prices, cash positions
and returns, and through the Builder class.
"""

import pandas as pd
import pytest

from cvx.simulator.builder import Builder
from cvx.simulator.portfolio import Portfolio


def test_portfolio_cumulated(prices: pd.DataFrame) -> None:
    """
    Test building a portfolio with the Builder class and accounting for trading costs.

    This test creates a portfolio using the Builder class, sets cash positions for
    two assets, accounts for trading costs (5 bps of traded value), and verifies
    that the resulting portfolio has the expected NAV.

    Parameters
    ----------
    prices : pd.DataFrame
        DataFrame of asset prices with dates as index and assets as columns,
        provided by a fixture
    """
    builder = Builder(prices=prices[["A", "B"]].tail(10), initial_aum=1e6)

    for t, state in builder:
        # hold one share in both assets
        builder.cashposition = [1e5, 4e5]

        # costs are 5 bps of the traded value
        costs = 0.0005 * (state.trades.abs() * state.prices).sum()

        # reduce the available aum by the costs
        builder.aum = state.aum - costs

    portfolio = builder.build()

    pd.testing.assert_series_equal(portfolio.nav, builder.aum)

    # assert sharpe(portfolio.nav.pct_change().dropna()) == pytest.approx(3.891806571531769, abs=1e-3)
    assert portfolio.nav.iloc[-1] == pytest.approx(1015576.0104632963, abs=1e-3)


def test_from_cash_position_prices(prices: pd.DataFrame) -> None:
    """
    Test creating a portfolio from cash positions and prices.

    This test creates a portfolio using the Portfolio.from_cashpos_prices factory
    method, which takes cash positions and prices as input. It then verifies that
    the portfolio's NAV is correctly calculated from the cumulative profit plus
    the initial AUM.

    Parameters
    ----------
    prices : pd.DataFrame
        DataFrame of asset prices with dates as index and assets as columns,
        provided by a fixture
    """
    cashpos = pd.DataFrame(index=prices.index, columns=prices.columns, data=1e5)

    portfolio = Portfolio.from_cashpos_prices(prices=prices, cashposition=cashpos, aum=1e6)

    profit = (portfolio.cashposition.shift(1) * portfolio.returns).sum(axis=1)
    pd.testing.assert_series_equal(portfolio.nav, profit.cumsum() + portfolio.aum, check_names=False)


def test_from_cash_returns(prices: pd.DataFrame) -> None:
    """
    Test creating a portfolio from cash positions and returns.

    This test creates a portfolio using the Portfolio.from_cashpos_returns factory
    method, which takes cash positions and returns as input. It converts returns
    to prices internally and then creates the portfolio. The test verifies that
    the portfolio's NAV is correctly calculated from the cumulative profit plus
    the initial AUM.

    Parameters
    ----------
    prices : pd.DataFrame
        DataFrame of asset prices with dates as index and assets as columns,
        provided by a fixture
    """
    returns = prices.pct_change().fillna(0.0)
    cashpos = pd.DataFrame(index=prices.index, columns=prices.columns, data=1e5)

    portfolio = Portfolio.from_cashpos_returns(returns=returns, cashposition=cashpos, aum=1e6)

    profit = (portfolio.cashposition.shift(1) * portfolio.returns).sum(axis=1)
    pd.testing.assert_series_equal(portfolio.nav, profit.cumsum() + portfolio.aum, check_names=False)

    print(portfolio.weights)
