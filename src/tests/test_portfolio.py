"""
Tests for the Portfolio class in the cvx.simulator package.

This module contains tests for the Portfolio class, which represents a portfolio
of assets with methods for calculating various metrics (NAV, profit, drawdown, etc.)
and analyzing performance. The tests verify that the Portfolio correctly calculates
these metrics and provides the expected visualization capabilities.
"""

import pandas as pd
import pytest

from cvx.simulator.portfolio import Portfolio

# from cvx.simulator.utils.metric import sharpe


@pytest.fixture()
def portfolio(prices: pd.DataFrame, nav: pd.Series) -> Portfolio:
    """
    Create a Portfolio fixture for testing.

    This fixture creates a Portfolio instance with the provided price data,
    unit positions of 1.0 for all assets, and the provided NAV series.

    Parameters
    ----------
    prices : pd.DataFrame
        DataFrame of asset prices with dates as index and assets as columns
    nav : pd.Series
        Series of portfolio NAV values over time

    Returns
    -------
    Portfolio
        A Portfolio instance initialized with the provided data
    """
    units = pd.DataFrame(index=prices.index, columns=prices.columns, data=1.0)
    return Portfolio(prices=prices, units=units, aum=nav)


def test_assets(portfolio: Portfolio, prices: pd.DataFrame) -> None:
    """
    Test that the portfolio assets match the price data columns.

    This test verifies that the Portfolio.assets property correctly returns
    all assets from the price data.

    Parameters
    ----------
    portfolio : Portfolio
        The Portfolio fixture to test
    prices : pd.DataFrame
        The price data fixture used to initialize the Portfolio
    """
    assert set(portfolio.assets) == set(prices.columns)


def test_index(portfolio: Portfolio) -> None:
    """
    Test that the portfolio index matches the price data index.

    This test verifies that the Portfolio.index property correctly returns
    the time index from the price data, and that it has the expected length.

    Parameters
    ----------
    portfolio : Portfolio
        The Portfolio fixture to test
    """
    assert len(portfolio.index) == 602
    pd.testing.assert_index_equal(portfolio.index, portfolio.prices.index)


def test_prices(portfolio: Portfolio, prices: pd.DataFrame) -> None:
    """
    Test that the portfolio prices match the input price data.

    This test verifies that the Portfolio.prices property correctly returns
    the price data that was used to initialize the Portfolio.

    Parameters
    ----------
    portfolio : Portfolio
        The Portfolio fixture to test
    prices : pd.DataFrame
        The price data fixture used to initialize the Portfolio
    """
    pd.testing.assert_frame_equal(portfolio.prices, prices)


def test_turnover(portfolio: Portfolio) -> None:
    """
    Test that the turnover calculations are correct.

    This test verifies that:
    1. trades_currency is correctly calculated as trades_units * prices
    2. turnover is correctly calculated as the absolute value of trades_currency

    Parameters
    ----------
    portfolio : Portfolio
        The Portfolio fixture to test
    """
    v = portfolio.trades_units * portfolio.prices
    pd.testing.assert_frame_equal(v, portfolio.trades_currency)
    pd.testing.assert_frame_equal(v.abs(), portfolio.turnover)


def test_turnover_relative(portfolio: Portfolio) -> None:
    """
    Test that the relative turnover calculations are correct.

    This test verifies that turnover_relative is correctly calculated as
    trades_currency divided by NAV.

    Parameters
    ----------
    portfolio : Portfolio
        The Portfolio fixture to test
    """
    v = portfolio.trades_units * portfolio.prices
    pd.testing.assert_frame_equal(v.div(portfolio.nav, 0), portfolio.turnover_relative)


def test_get(portfolio: Portfolio) -> None:
    """
    Test that the __getitem__ method works correctly.

    This test verifies that the Portfolio class can be indexed by time
    to retrieve the positions (units) at that time point, and that the
    result is a pandas Series.

    Parameters
    ----------
    portfolio : Portfolio
        The Portfolio fixture to test
    """
    for time in portfolio.index:
        w = portfolio[time]
        assert isinstance(w, pd.Series)


def test_units(portfolio: Portfolio) -> None:
    """
    Test that the units property returns the correct position data.

    This test verifies that the Portfolio.units property correctly returns
    the position data that was used to initialize the Portfolio (all 1.0 in this case).

    Parameters
    ----------
    portfolio : Portfolio
        The Portfolio fixture to test
    """
    stocks = pd.DataFrame(index=portfolio.index, columns=portfolio.assets, data=1.0)
    pd.testing.assert_frame_equal(portfolio.units, stocks)


def test_returns(portfolio: Portfolio) -> None:
    """
    Test that the returns property calculates asset returns correctly.

    This test verifies that the Portfolio.returns property correctly calculates
    the percentage change in prices for each asset.

    Parameters
    ----------
    portfolio : Portfolio
        The Portfolio fixture to test
    """
    pd.testing.assert_frame_equal(portfolio.returns, portfolio.prices.pct_change())


def test_cashposition(portfolio: Portfolio) -> None:
    """
    Test that the cashposition property calculates position values correctly.

    This test verifies that the Portfolio.cashposition property correctly calculates
    the cash value of each position by multiplying units by prices.

    Parameters
    ----------
    portfolio : Portfolio
        The Portfolio fixture to test
    """
    pd.testing.assert_frame_equal(portfolio.cashposition, portfolio.prices * portfolio.units)


def test_sharpe(portfolio: Portfolio) -> None:
    assert portfolio.data.stats.sharpe(periods=252)["NAV"] == pytest.approx(-0.1021095912448208)


# def test_plotly_aggregate(portfolio: Portfolio) -> None:
#     """
#     Test that the snapshot method works with aggregated data.
#
#     This test verifies that the Portfolio.snapshot method correctly generates
#     a plotly figure when the aggregate parameter is set to True.
#
#     Parameters
#     ----------
#     portfolio : Portfolio
#         The Portfolio fixture to test
#     """
#     benchmark = pd.Series(index=portfolio.index, data=1e6)
#     fig = portfolio.snapshot(benchmark=benchmark, aggregate=True)
#     fig.show()
#
#
# def test_plotly_no_aggregate(portfolio: Portfolio) -> None:
#     """
#     Test that the snapshot method works with non-aggregated data.
#
#     This test verifies that the Portfolio.snapshot method correctly generates
#     a plotly figure when the aggregate parameter is set to False (default).
#
#     Parameters
#     ----------
#     portfolio : Portfolio
#         The Portfolio fixture to test
#     """
#     benchmark = pd.Series(index=portfolio.index, data=1e6)
#     fig = portfolio.snapshot(benchmark=benchmark)
#     fig.show()


def test_drawdown(portfolio: Portfolio) -> None:
    """
    Test that the drawdown and highwater properties calculate correctly.

    This test verifies that:
    1. The highwater property correctly calculates the expanding maximum of NAV
    2. The drawdown property correctly calculates 1 - (NAV / highwater)

    Parameters
    ----------
    portfolio : Portfolio
        The Portfolio fixture to test
    """
    pd.testing.assert_series_equal(
        portfolio.highwater,
        portfolio.nav.expanding(min_periods=1).max(),
        check_names=False,
    )

    drawdown = 1.0 - portfolio.nav / portfolio.highwater
    pd.testing.assert_series_equal(portfolio.drawdown, drawdown, check_names=False)


def test_monotonic() -> None:
    """
    Test that Portfolio initialization fails with non-monotonic index.

    This test verifies that the Portfolio class correctly raises an AssertionError
    when initialized with price data that has a non-monotonic index.
    """
    prices = pd.DataFrame(index=[2, 1], columns=["A"])
    with pytest.raises(AssertionError):
        Portfolio(prices=prices, units=prices, aum=1e6)


# def test_snapshot(portfolio: Portfolio) -> None:
#     """
#     Test that the snapshot method works with a zero benchmark.
#
#     This test verifies that the Portfolio.snapshot method correctly generates
#     a plotly figure when provided with a benchmark series of zeros.
#
#     Parameters
#     ----------
#     portfolio : Portfolio
#         The Portfolio fixture to test
#     """
#     xxx = pd.Series(index=portfolio.index, data=0.0)
#     fig = portfolio.snapshot(benchmark=xxx)
#     fig.show()


# def test_snapshot_no_benchmark(portfolio: Portfolio) -> None:
#     """
#     Test that the snapshot method works without a benchmark.
#
#     This test verifies that the Portfolio.snapshot method correctly generates
#     a plotly figure when no benchmark is provided.
#
#     Parameters
#     ----------
#     portfolio : Portfolio
#         The Portfolio fixture to test
#     """
#     fig = portfolio.snapshot()
#     fig.show()


# def test_snapshot_log_axis(portfolio: Portfolio) -> None:
#     """
#     Test that the snapshot method works with a logarithmic scale.
#
#     This test verifies that the Portfolio.snapshot method correctly generates
#     a plotly figure with a logarithmic y-axis when log_scale is set to True.
#
#     Parameters
#     ----------
#     portfolio : Portfolio
#         The Portfolio fixture to test
#     """
#     xxx = pd.Series(index=portfolio.index, data=10.0)
#     fig = portfolio.snapshot(log_scale=True, benchmark=xxx)
#     fig.show()


def test_equity(portfolio):
    """
    Test that the equity of the portfolio is the same as the prices * units
    :param portfolio: the portfolio object (fixture)
    """
    pd.testing.assert_frame_equal(portfolio.equity, portfolio.prices * portfolio.units)


def test_profit(portfolio):
    """
    Test that the profit is computed correctly
    :param portfolio: the portfolio object (fixture)
    """
    pd.testing.assert_series_equal(
        portfolio.profit,
        portfolio.equity.sum(axis=1).diff().fillna(0.0),
        check_names=False,
    )


# def test_profit_metrics(portfolio):
#     """
#     Test that the profit is computed correctly
#     :param portfolio: the portfolio object (fixture)
#     """
#     assert portfolio.profit.mean() == pytest.approx(-5.801328903654639)
#     assert portfolio.profit.std() == pytest.approx(839.8620124674756)
#     assert portfolio.profit.sum() == pytest.approx(-3492.4000000000033)
#     # assert sharpe(portfolio.profit, n=252) == pytest.approx(-0.10965282385614909)
#     # profit is replacing NaNs with 0?!
#     assert portfolio.sharpe() == pytest.approx(-0.1038600869081656)


def test_snapshot(portfolio: Portfolio) -> None:
    fig = portfolio.snapshot()
    fig.show()
