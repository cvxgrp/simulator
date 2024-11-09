"""test portfolio"""
from __future__ import annotations

import numpy as np
import pytest

from cvx.simulator.portfolio import Portfolio
from cvx.simulator.utils.metric import sharpe


# take two moving averages and apply sign-function
def f(prices, fast=32, slow=96):
    """
    construct cash position
    """
    s = prices.ewm(com=slow, min_periods=100).mean()
    f = prices.ewm(com=fast, min_periods=100).mean()
    return np.sign(f - s)


def test_portfolio(prices):
    """
    test portfolio

    Args:
        prices: adjusted prices of futures
    """
    portfolio = Portfolio.from_cashpos_prices(
        prices=prices, cashposition=1e6 * f(prices), aum=1e6
    )
    assert sharpe(portfolio.nav.pct_change().dropna()) == pytest.approx(
        0.5330704741938855
    )
