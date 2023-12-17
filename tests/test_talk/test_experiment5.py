"""test portfolio"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from tinycta.linalg import inv_a_norm, solve
from tinycta.signal import osc, returns_adjust, shrink2id

from cvx.simulator import FuturesBuilder

correlation = 200


def f(prices, vola=96, clip=4.2, corr=200, shrinkage=0.5):
    """
    construct cash position
    """
    returns_adj = prices.apply(returns_adjust, com=vola, clip=clip)

    # this is a lot faster than Pandas...
    position = np.nan * np.zeros_like(prices.values)

    # DCC by Engle
    cor = returns_adj.ewm(com=corr, min_periods=corr).corr()

    mu = np.tanh(returns_adj.cumsum().apply(osc)).values
    vo = prices.pct_change().ewm(com=vola, min_periods=vola).std().values

    for n, t in enumerate(prices.index):
        matrix = shrink2id(cor.loc[t].values, lamb=shrinkage)
        risk_position = solve(matrix, mu[n]) / inv_a_norm(mu[n], matrix)
        position[n] = risk_position / vo[n]

    return pd.DataFrame(index=prices.index, columns=prices.keys(), data=position)


def test_portfolio(prices):
    """
    test portfolio

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

    builder = FuturesBuilder(prices=prices, aum=1e8)

    for n, (t, state) in enumerate(builder):
        mask = state.mask
        matrix = shrink2id(cor.loc[t[-1]].values, lamb=shrinkage)[mask, :][:, mask]
        expected_mu = np.nan_to_num(mu[n][mask])
        expected_vo = np.nan_to_num(vo[n][mask])
        risk_position = solve(matrix, expected_mu) / inv_a_norm(expected_mu, matrix)
        builder.cashposition = risk_position / expected_vo

    portfolio = builder.build()

    assert portfolio.nav.pct_change().sharpe() == pytest.approx(1.316275716921172)

    # portfolio.metrics()["Sharpe"] == pytest.approx(1.2778671597915794)
