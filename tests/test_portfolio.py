"""Tests for the Portfolio class in the cvx.simulator package.

This module contains tests for the Portfolio class, which represents a portfolio
of assets with methods for calculating various metrics (NAV, profit, drawdown, etc.)
and analyzing performance. The tests verify that the Portfolio correctly calculates
these metrics and provides the expected visualization capabilities.
"""

import numpy as np
import pandas as pd
import polars as pl
import polars.testing as pdt
import pytest

from cvxsimulator.builder import polars2pandas
from cvxsimulator.portfolio import Portfolio


@pytest.fixture()
def portfolio(prices: pd.DataFrame, nav: pd.Series) -> Portfolio:
    """Create a Portfolio fixture for testing.

    This fixture creates a Portfolio instance with the provided price data,
    unit positions of 1.0 for all assets, and the provided NAV series.

    Parameters
    ----------
    prices : pd.DataFrame
        DataFrame of asset prices with dates as index and assets as columns
    nav : pd.Series
        Series of portfolio NAV values over time

    Returns:
    -------
    Portfolio
        A Portfolio instance initialized with the provided data

    """
    units = pd.DataFrame(index=prices.index, columns=prices.columns, data=1.0)
    return Portfolio(prices=prices, units=units, aum=nav)


def test_prices_polars(prices: pd.DataFrame, prices_pl: pl.DataFrame):
    """Compare two dataframes by moving to polars and comparing.

    Args:
        prices (pd.DataFrame): Input pandas DataFrame to be converted to a Polars DataFrame.
        prices_pl (pl.DataFrame): Input Polars DataFrame to compare against.

    Raises:
    ------
    AssertionError: If the two dataframes are not equal after conversion.

    """
    pdt.assert_frame_equal(prices_pl, pl.from_pandas(prices.reset_index()))


def test_prices_pandas(prices: pd.DataFrame, prices_pl: pl.DataFrame):
    """Compare two dataframes by moving to pandas and comparing.

    This function compares a pandas DataFrame and a polars DataFrame after converting
    the polars DataFrame into a pandas DataFrame. Both DataFrames are expected to have
    financial price data with identical structures, and this verifies their eq.

    Args:
        prices (pd.DataFrame): A pandas DataFrame containing price data.
        prices_pl (pl.DataFrame): A polars DataFrame containing price data.

    Raises:
    ------
    AssertionError: If the converted polars DataFrame does not match the pandas DataFrame.

    """
    pd.testing.assert_frame_equal(prices_pl.to_pandas().set_index("date"), prices)


def test_assets(portfolio: Portfolio, prices: pd.DataFrame) -> None:
    """Test that the portfolio assets match the price data columns.

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
    """Test that the portfolio index matches the price data index.

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
    """Test that the portfolio prices match the input price data.

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
    """Test that the turnover calculations are correct.

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
    """Test that the relative turnover calculations are correct.

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
    """Test that the __getitem__ method works correctly.

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
    """Test that the units property returns the correct position data.

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
    """Test that the returns property calculates asset returns correctly.

    This test verifies that the Portfolio.returns property correctly calculates
    the percentage change in prices for each asset.

    Parameters
    ----------
    portfolio : Portfolio
        The Portfolio fixture to test

    """
    pd.testing.assert_frame_equal(portfolio.returns, portfolio.prices.pct_change())


def test_cashposition(portfolio: Portfolio) -> None:
    """Test that the cashposition property calculates position values correctly.

    This test verifies that the Portfolio.cashposition property correctly calculates
    the cash value of each position by multiplying units by prices.

    Parameters
    ----------
    portfolio : Portfolio
        The Portfolio fixture to test

    """
    pd.testing.assert_frame_equal(portfolio.cashposition, portfolio.prices * portfolio.units)


def test_sharpe(portfolio: Portfolio) -> None:
    """Test that the sharpe method calculates the Sharpe ratio correctly.

    This test verifies that the Portfolio.sharpe method correctly calculates
    the Sharpe ratio for the portfolio and returns the expected value.

    Parameters
    ----------
    portfolio : Portfolio
        The Portfolio fixture to test

    """
    assert portfolio.sharpe(periods=252) == pytest.approx(-0.1021095912448208)


def test_monotonic() -> None:
    """Test that Portfolio initialization fails with non-monotonic index.

    This test verifies that the Portfolio class correctly raises an AssertionError
    when initialized with price data that has a non-monotonic index.
    """
    prices = pd.DataFrame(index=[2, 1], columns=["A"])
    with pytest.raises(ValueError):
        Portfolio(prices=prices, units=prices, aum=1e6)


def test_equity(portfolio: Portfolio) -> None:
    """Test that the equity property calculates position values correctly.

    This test verifies that the Portfolio.equity property correctly calculates
    the cash value of each position by multiplying units by prices.

    Parameters
    ----------
    portfolio : Portfolio
        The Portfolio fixture to test

    """
    pd.testing.assert_frame_equal(portfolio.equity, portfolio.prices * portfolio.units)


def test_profit(portfolio: Portfolio) -> None:
    """Test that the profit property calculates portfolio profit correctly.

    This test verifies that the Portfolio.profit property correctly calculates
    the profit as the difference in portfolio value between consecutive time points.

    Parameters
    ----------
    portfolio : Portfolio
        The Portfolio fixture to test

    """
    pd.testing.assert_series_equal(
        portfolio.profit,
        portfolio.equity.sum(axis=1).diff().fillna(0.0),
        check_names=False,
    )


def test_metrics(portfolio: Portfolio) -> None:
    """Test that the portfolio metrics can be displayed.

    This test verifies that the Portfolio.reports.metrics() method can be called
    without errors and prints the resulting metrics to the console.

    Parameters
    ----------
    portfolio : Portfolio
        The Portfolio fixture to test

    """
    print(portfolio.reports.metrics())


def test_snapshot(portfolio: Portfolio) -> None:
    """Test that the portfolio snapshot can be generated.

    This test verifies that the Portfolio.snapshot() method can be called
    without errors and returns a figure object.

    Parameters
    ----------
    portfolio : Portfolio
        The Portfolio fixture to test

    """
    fig = portfolio.snapshot()
    print(fig.to_dict())


def test_weights(portfolio: Portfolio) -> None:
    """Test that the weights property correctly calculates the weight of each asset in the portfolio.

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


def test_csv_route(prices_pl, prices):
    """Tests the CSV data conversion route.

    This function checks the conversion of a Polars DataFrame to a Pandas DataFrame using
    a helper function. It performs a series of assertions to validate the transformation,
    ensuring the integrity of data types and the uniformity of the output.

    Parameters
    ----------
    prices_pl : polars.DataFrame
        The input Polars DataFrame containing price data.
    prices : pandas.DataFrame
        The reference Pandas DataFrame for comparison.

    Raises:
    ------
    AssertionError
        If the converted DataFrame does not match the reference DataFrame in structure,
        content, index data type, or column data types.

    """
    ppp = polars2pandas(prices_pl, date_col="date")
    pd.testing.assert_frame_equal(ppp, prices)

    assert ppp.index.dtype == "datetime64[ns]"
    assert np.all(np.array(ppp.dtypes) == np.dtype("float64"))
