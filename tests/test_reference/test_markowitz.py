import pytest

from cvx.simulator.builder import builder as _builder
from tests.test_reference.markowitz import (
    OptimizationInput,
    basic_markowitz,
    synthetic_returns,
)


@pytest.fixture()
def means(prices):
    forward_smoothing = 5
    return (
        synthetic_returns(
            prices, information_ratio=0.15, forward_smoothing=forward_smoothing
        )
        .shift(-1)
        .dropna()
    )


@pytest.fixture()
def builder(prices, spreads, rf):
    return _builder(
        prices=prices,
        initial_cash=1e6,
        risk_free_rate=rf,
        borrow_rate=5 * rf,
        spreads=spreads,
    )


@pytest.fixture()
def feasible(prices):
    return prices.index[200:-10]


@pytest.fixture()
def covariance(prices):
    returns = prices.pct_change().dropna()
    covariance_df = returns.ewm(halflife=125).cov()  # At time t includes data up to t
    return {day: covariance_df.loc[day] for day in returns.index}


def test_a(builder, feasible, covariance, means):
    """
    Test that the prices are the same
    :param data: the data (fixture)
    """
    for t, state in builder:
        if t[-1] in feasible:
            print(state.cash_interest)
            print(state.borrow_fees)
            print(state.cash)
            print(state.spreads)
            print(state.prices)
            print(state.weights)
            print(state.cashposition)
            print(state.position)
            print(state.leverage)
            print(state.nav)
            print(state.assets)

            _input = OptimizationInput(
                mean=means.loc[t[-1]],
                covariance=covariance[t[-1]],
                risk_target=0.01,
            )

            w, _ = basic_markowitz(_input)
            builder.weights = w

    portfolio = builder.build()
    portfolio.snapshot()
