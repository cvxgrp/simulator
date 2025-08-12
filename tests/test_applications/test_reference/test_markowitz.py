"""Tests for the Markowitz portfolio optimization implementation.

This module contains tests for the Markowitz portfolio optimization implementation
in the test_applications.test_reference.markowitz module. It verifies that the
optimization correctly computes portfolio weights based on expected returns and
covariance, and that the resulting portfolio behaves as expected when simulated
over time with interest on cash and borrowing fees.
"""

import pytest

from cvxsimulator.builder import Builder

from .markowitz import (
    OptimizationInput,
    basic_markowitz,
    synthetic_returns,
)


@pytest.fixture()
def means(prices):
    """Create synthetic expected returns for testing.

    This fixture generates synthetic expected returns using the synthetic_returns
    function with a specified information ratio and forward smoothing. The returns
    are shifted to simulate forward-looking expectations.

    Parameters
    ----------
    prices : pd.DataFrame
        DataFrame of asset prices with dates as index and assets as columns

    Returns:
    -------
    pd.DataFrame
        DataFrame of synthetic expected returns

    """
    forward_smoothing = 5
    return synthetic_returns(prices, information_ratio=0.15, forward_smoothing=forward_smoothing).shift(-1).dropna()


@pytest.fixture()
def builder(prices):
    """Create a builder for testing."""
    return Builder(
        prices=prices,
        initial_aum=1e6,
    )


@pytest.fixture()
def feasible(prices):
    """Create a subset of dates for testing.

    This fixture creates a subset of dates from the price data, excluding
    the first 200 and last 10 dates. This represents the dates for which
    the Markowitz optimization will be performed.

    Parameters
    ----------
    prices : pd.DataFrame
        DataFrame of asset prices with dates as index and assets as columns

    Returns:
    -------
    pd.DatetimeIndex
        A subset of dates from the price data

    """
    return prices.index[200:-10]


@pytest.fixture()
def covariance(prices):
    """Create a covariance matrix for each date in the price data.

    This fixture calculates the covariance matrix of asset returns for each date
    in the price data using an exponentially weighted moving average with a halflife
    of 125 days. The result is a dictionary mapping each date to its corresponding
    covariance matrix.

    Parameters
    ----------
    prices : pd.DataFrame
        DataFrame of asset prices with dates as index and assets as columns

    Returns:
    -------
    dict
        A dictionary mapping dates to covariance matrices

    """
    returns = prices.pct_change().dropna()
    covariance_df = returns.ewm(halflife=125).cov()  # At time t includes data up to t
    return {day: covariance_df.loc[day] for day in returns.index}


def test_markowitz(builder, feasible, covariance, means, spreads):
    """Test the markowitz portfolio complete with interest on cash and borrowing fees."""
    # We loop over the entire history the builder
    for t, state in builder:
        # the very first and the very last elements are ignored
        if t[-1] in feasible:
            # earn interest rate on the existing cash
            r = 0.02
            state.cash = (1 + r / 365) ** state.days * state.cash

            # pay fee on gmv to the broker
            # alternate the model here...
            state.cash -= (1 + 0.0025 / 365) ** state.days * state.gmv - state.gmv

            # define the input needed for the optimizer
            _input = OptimizationInput(
                mean=means.loc[t[-1]],
                covariance=covariance[t[-1]],
                risk_target=0.01,
            )

            #  optimize portfolio
            w = basic_markowitz(_input)

            # update weights in builder
            builder.weights = w

            # the builder keeps also track of the state
            # some quantities are only post-trading interesting
            costs = (state.trades.abs() * (state.prices * spreads.loc[t[-1]] / 2)).sum()

            # reduce the aum by the costs paid...
            builder.aum = state.aum - costs

    # build the portfolio
    # portfolio = builder.build()

    # portfolio.snapshot(title="Markowitz Portfolio", aggregate=True)
