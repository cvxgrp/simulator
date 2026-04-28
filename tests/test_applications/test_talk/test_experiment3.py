"""Test Experiment 3."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cvx.simulator.portfolio import Portfolio


def osc(prices: pd.DataFrame, fast: int = 32, slow: int = 96, scaling: bool = True) -> pd.DataFrame:
    """Compute EWMA-based oscillator, optionally scaled by its standard deviation."""
    diff = prices.ewm(com=fast - 1).mean() - prices.ewm(com=slow - 1).mean()
    s = diff.std() if scaling else 1
    return diff / s


def returns_adjust(price: pd.DataFrame, com: int = 32, min_periods: int = 300, clip: float = 4.2) -> pd.DataFrame:
    """Return volatility-normalised log returns clipped to [-clip, +clip]."""
    r = price.apply(np.log).diff()
    return (r / r.ewm(com=com, min_periods=min_periods).std()).clip(-clip, +clip)


@pytest.fixture
def portfolio(prices) -> Portfolio:
    """Compute the portfolio."""

    # take two moving averages and apply the sign-function, adjust by volatility
    def f(slow=96, fast=32, vola=96, clip=3):
        """Construct cash position."""
        # construct a fake-price, those fake-prices have homescedastic returns
        # price_adj = prices.apply(adj, com=vola, min_periods=100, clip=clip).cumsum()

        price_adj = returns_adjust(prices, com=vola, min_periods=100, clip=clip).cumsum()
        # compute mu
        mu = np.tanh(osc(price_adj, fast=fast, slow=slow))
        return mu / prices.pct_change().ewm(com=slow, min_periods=100).std()

    portfolio = Portfolio.from_cashpos_prices(prices=prices, cashposition=1e6 * f(), aum=1e8)

    return portfolio


def test_portfolio(portfolio):
    """Test portfolio."""
    assert portfolio.sharpe() == pytest.approx(0.9134164184741005)
