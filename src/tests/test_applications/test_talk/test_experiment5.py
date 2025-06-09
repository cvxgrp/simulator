"""Test Experiment 5."""

from __future__ import annotations

import numpy as np
import pytest
from tinycta.linalg import inv_a_norm, solve
from tinycta.signal import osc, returns_adjust, shrink2id

from cvxsimulator.builder import Builder

correlation = 200


def test_portfolio(prices):
    """Test portfolio.

    Args:
        prices: adjusted prices of futures

    """
    vola = 96
    clip = 4.2
    corr = 200
    shrinkage = 0.5

    returns_adj = prices.apply(returns_adjust, com=vola, clip=clip)

    # DCC by Engle
    cor = returns_adj.ewm(com=corr, min_periods=corr).corr()

    mu = np.tanh(returns_adj.cumsum().apply(osc)).values
    vo = prices.pct_change().ewm(com=vola, min_periods=vola).std().values

    builder = Builder(prices=prices, initial_aum=1e8)

    for n, (t, state) in enumerate(builder):
        mask = state.mask
        matrix = shrink2id(cor.loc[t[-1]].values, lamb=shrinkage)[mask, :][:, mask]
        expected_mu = np.nan_to_num(mu[n][mask])
        expected_vo = np.nan_to_num(vo[n][mask])
        risk_position = solve(matrix, expected_mu) / inv_a_norm(expected_mu, matrix)
        builder.cashposition = risk_position / expected_vo
        # you could correct here with trading costs
        builder.aum = state.aum

    portfolio = builder.build()
    assert portfolio.sharpe() == pytest.approx(1.3347932969566416)

    # assert sharpe(portfolio.nav.pct_change().dropna()) == pytest.approx(1.3348481418003217)

    # portfolio.metrics()["Sharpe"] == pytest.approx(1.2778671597915794)
