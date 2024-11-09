"""test portfolio"""
from __future__ import annotations

import numpy as np
import pytest

from cvx.simulator.portfolio import Portfolio
from cvx.simulator.utils.metric import sharpe


# take two moving averages and apply the sign-function, adjust by volatility
def f(prices, fast=32, slow=96, volatility=32):
    """
    construct cash position
    """
    s = prices.ewm(com=slow, min_periods=100).mean()
    f = prices.ewm(com=fast, min_periods=100).mean()
    std = prices.pct_change().ewm(com=volatility, min_periods=100).std()
    return np.sign(f - s) / std


def test_portfolio(prices):
    """
    test portfolio

    Args:
        prices: adjusted prices of futures
    """
    portfolio = Portfolio.from_cashpos_prices(
        prices=prices, cashposition=1e6 * f(prices), aum=1e8
    )
    assert sharpe(portfolio.nav.pct_change()) == pytest.approx(0.6231488411522045)
