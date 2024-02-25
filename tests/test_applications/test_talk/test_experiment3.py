from __future__ import annotations

import numpy as np
import pytest
from tinycta.signal import osc, returns_adjust

from cvx.simulator.portfolio import Portfolio
from cvx.simulator.utils.metric import sharpe


# take two moving averages and apply the sign-function, adjust by volatility
def f(prices, slow=96, fast=32, vola=96, clip=3):
    """
    construct cash position
    """
    # construct a fake-price, those fake-prices have homescedastic returns
    price_adj = returns_adjust(prices, com=vola, min_periods=100, clip=clip).cumsum()
    # compute mu
    mu = np.tanh(osc(price_adj, fast=fast, slow=slow))
    return mu / prices.pct_change().ewm(com=slow, min_periods=100).std()


def test_portfolio(prices):
    """
    test portfolio

    Args:
        prices: adjusted prices of futures
    """
    portfolio = Portfolio.from_cashpos_prices(
        prices=prices, cashposition=1e6 * f(prices), aum=1e8
    )
    assert sharpe(portfolio.nav.pct_change()) == pytest.approx(0.8999500486718532)
