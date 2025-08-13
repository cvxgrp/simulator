"""Tests for a simple trend-following portfolio strategy.

This module contains tests for a simple trend-following portfolio strategy
that uses the difference between fast and slow moving averages to determine
position signs. It verifies that the resulting portfolio has the expected
Sharpe ratio.
"""

from __future__ import annotations

import numpy as np
import pytest

from cvxsimulator import Portfolio


def f(prices, fast=32, slow=96):
    """Construct position signs based on moving average crossover.

    This function calculates the sign of the difference between a fast and slow
    exponential moving average of prices. A positive value indicates a long position,
    a negative value indicates a short position, and zero indicates no position.

    Parameters
    ----------
    prices : pd.DataFrame
        DataFrame of asset prices with dates as index and assets as columns
    fast : int, optional
        The com parameter for the fast exponential moving average, by default 32
    slow : int, optional
        The com parameter for the slow exponential moving average, by default 96

    Returns:
    -------
    pd.DataFrame
        DataFrame of position signs (-1, 0, or 1) with the same shape as prices

    """
    s = prices.ewm(com=slow, min_periods=100).mean()
    f = prices.ewm(com=fast, min_periods=100).mean()
    return np.sign(f - s)


def test_portfolio(prices):
    """Test portfolio.

    Args:
        prices: adjusted prices of futures

    """
    portfolio = Portfolio.from_cashpos_prices(prices=prices, cashposition=1e6 * f(prices), aum=1e6)
    assert portfolio.sharpe() == pytest.approx(0.5330485719520409)
