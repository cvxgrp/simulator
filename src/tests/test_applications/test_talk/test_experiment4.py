"""Test Experiment 4."""

from __future__ import annotations

import numpy as np
import pytest
from tinycta.signal import osc, returns_adjust

from cvxsimulator.portfolio import Portfolio


# take two moving averages and apply the sign-function, adjust by volatility
def f(prices, slow=96, fast=32, vola=96, clip=3):
    """Construct cash position."""
    mu = np.tanh(prices.apply(returns_adjust, com=vola, clip=clip).cumsum().apply(osc, fast=fast, slow=slow))
    volatility = prices.pct_change().ewm(com=vola, min_periods=vola).std()

    # compute the series of Euclidean norms by compute the sum of squares for each row
    euclid_norm = np.sqrt((mu * mu).sum(axis=1))

    # Divide each column of mu by the Euclidean norm
    risk_scaled = mu.apply(lambda x: x / euclid_norm, axis=0)

    return risk_scaled / volatility


def test_portfolio(prices):
    """Test portfolio.

    Args:
        prices: adjusted prices of futures

    """
    portfolio = Portfolio.from_cashpos_prices(prices=prices, cashposition=1e6 * f(prices), aum=1e8)
    assert portfolio.sharpe() == pytest.approx(0.9824232063067163)
