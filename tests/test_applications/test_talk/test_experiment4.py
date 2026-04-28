"""Test Experiment 4."""

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
