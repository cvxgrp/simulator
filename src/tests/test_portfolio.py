"""
Tests for the Portfolio class in the cvx.simulator package.

This module contains tests for the Portfolio class, which represents a portfolio
of assets with methods for calculating various metrics (NAV, profit, drawdown, etc.)
and analyzing performance. The tests verify that the Portfolio correctly calculates
these metrics and provides the expected visualization capabilities.
"""

import pandas as pd
import polars as pl
import polars.testing as pdt
import pytest

from cvx.simulator.portfolio import Portfolio


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


def test_prices_polars(prices: pd.DataFrame, prices_pl: pl.DataFrame):
    prices = pl.from_pandas(prices.reset_index())
    pdt.assert_frame_equal(prices_pl, prices)


def test_prices_pandas(prices: pd.DataFrame, prices_pl: pl.DataFrame):
    prices_pl = prices_pl.to_pandas().set_index("date")
    pd.testing.assert_frame_equal(prices_pl, prices)


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
    assert portfolio.index == portfolio.prices.index.to_list()


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
    stocks.index.name = "date"
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
    assert portfolio.sharpe(periods=252) == pytest.approx(-0.1021095912448208)


def test_monotonic() -> None:
    """
    Test that Portfolio initialization fails with non-monotonic index.

    This test verifies that the Portfolio class correctly raises an AssertionError
    when initialized with price data that has a non-monotonic index.
    """
    prices = pd.DataFrame(index=[2, 1], columns=["A"])
    with pytest.raises(AssertionError):
        Portfolio(prices=prices, units=prices, aum=1e6)


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


def test_metrics(portfolio):
    print(portfolio.reports.metrics())


def test_snapshot(portfolio: Portfolio) -> None:
    fig = portfolio.snapshot()
    fig.show()


def test_weights(portfolio: Portfolio) -> None:
    """
    Test that the weights property correctly calculates the weight of each asset in the portfolio.

    This test verifies that the weights are correctly calculated by dividing the equity by the NAV.
    In a real portfolio, the sum of weights would typically equal 1.0 for a fully invested portfolio,
    but in our test fixture, the NAV is loaded from a separate file and may not equal the sum of equity values.

    Parameters
    ----------
    portfolio : Portfolio
        The Portfolio fixture to test
    """
    # Calculate expected weights by dividing equity by NAV
    expected_weights = portfolio.equity.apply(lambda x: x / portfolio.nav)

    # Verify that the weights property returns the expected values
    pd.testing.assert_frame_equal(portfolio.weights, expected_weights)


def test_from_cashpos_returns() -> None:
    """
    Test that the from_cashpos_returns method correctly creates a portfolio from returns and cashposition.

    This test creates a returns DataFrame and a cashposition DataFrame, calls from_cashpos_returns
    with these inputs and a sample aum value, and verifies that the resulting Portfolio object
    has the expected properties.
    """
    # Create a sample returns DataFrame
    returns = pd.DataFrame({"A": [0.05, 0.03, -0.02, 0.01], "B": [0.02, 0.01, 0.03, -0.01]})

    # Create a sample cashposition DataFrame
    # The cashposition values don't matter much for this test, as they're just used to calculate units
    cashposition = pd.DataFrame({"A": [100, 110, 105, 110], "B": [200, 205, 215, 210]}, index=returns.index)

    # Sample aum value
    aum = 1e6

    # Call from_cashpos_returns
    portfolio = Portfolio.from_cashpos_returns(returns, cashposition, aum)

    # Verify that the portfolio has the expected properties
    # 1. The prices should be calculated from the returns using returns2prices
    from cvx.simulator.utils.rescale import returns2prices

    expected_prices = returns2prices(returns)

    # Set the index name to match the portfolio's index name
    expected_prices.index.name = "Date"

    # Convert portfolio.prices to pandas if it's a polars DataFrame
    prices_pd = portfolio.prices
    if not isinstance(prices_pd, pd.DataFrame):
        prices_pd = portfolio.prices.to_pandas()

    pd.testing.assert_frame_equal(prices_pd, expected_prices)

    # 2. The units should be calculated by dividing cashposition by prices
    expected_units = cashposition.div(expected_prices, fill_value=0.0)
    pd.testing.assert_frame_equal(portfolio.units, expected_units)

    # 3. The aum should be the provided aum value
    assert portfolio.aum == aum
