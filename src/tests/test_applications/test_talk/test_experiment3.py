from __future__ import annotations

import numpy as np
import pytest
from jquantstats.api import build_data
from tinycta.signal import osc, returns_adjust

from cvx.simulator.portfolio import Portfolio


def adj(price, com, min_periods, clip):
    # assert np.all(price.dropna() > 0.0)
    # assert (price.dropna() > 0).all()
    r = np.log(price.dropna()).diff()
    return (r / r.ewm(com=com, min_periods=min_periods).std()).clip(-clip, +clip)


@pytest.fixture()
def portfolio(prices) -> Portfolio:
    # take two moving averages and apply the sign-function, adjust by volatility
    def f(slow=96, fast=32, vola=96, clip=3):
        """
        construct cash position
        """

        # construct a fake-price, those fake-prices have homescedastic returns
        # price_adj = prices.apply(adj, com=vola, min_periods=100, clip=clip).cumsum()

        price_adj = returns_adjust(prices, com=vola, min_periods=100, clip=clip).cumsum()
        # compute mu
        mu = np.tanh(osc(price_adj, fast=fast, slow=slow))
        return mu / prices.pct_change().ewm(com=slow, min_periods=100).std()

    portfolio = Portfolio.from_cashpos_prices(prices=prices, cashposition=1e6 * f(), aum=1e8)

    return portfolio


def test_portfolio(portfolio):
    """
    test portfolio
    """
    data = build_data(returns=portfolio.nav_pl)

    assert data.stats.sharpe()["NAV"] == pytest.approx(0.9134164184741005)
