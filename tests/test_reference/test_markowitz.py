import pytest

from cvx.simulator.builder import Builder
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
    return Builder(
        prices=prices,
        initial_cash=1e6,
        risk_free_rate=rf,
        borrow_rate=3 * rf,
        input_data={"spreads": spreads},
        # trading_cost_model=SpreadModel(),
    )


@pytest.fixture()
def feasible(prices):
    return prices.index[200:-10]


@pytest.fixture()
def covariance(prices):
    returns = prices.pct_change().dropna()
    covariance_df = returns.ewm(halflife=125).cov()  # At time t includes data up to t
    return {day: covariance_df.loc[day] for day in returns.index}


def test_markowitz(builder, feasible, covariance, means):
    """
    Test the markowitz portfolio complete with interest on cash and borrowing fees.
    """
    # We loop over the entire history the builder
    for t, state in builder:
        # the very first and the very last elements are ignored
        if t[-1] in feasible:
            # State is exposing numerous quantities
            print(state.cash_interest)
            print(state.borrow_fees)
            print(state.cash)
            #
            print(state.spreads)
            print(state.prices)
            print(state.weights)
            print(state.cashposition)
            print(state.position)
            print(state.leverage)
            print(state.nav)
            print(state.assets)

            # We define the input needed for the optimizer
            _input = OptimizationInput(
                mean=means.loc[t[-1]],
                covariance=covariance[t[-1]],
                risk_target=0.01,
            )

            #  optimize portfolio
            w, _ = basic_markowitz(_input)

            # update weights in builder
            # also possible to update positions or cash-positions
            builder.weights = w

            # the builder keeps also track of the state
            # some quantities are only post-trading interesting
            print(state.trades)
            # should trading costs be given per asset?
            print(state.trading_costs)
            print(state.cash)

    # build the portfolio
    portfolio = builder.build()
    portfolio.snapshot()

    print(portfolio.cash_interest)
    print(portfolio.borrow_fees)
    print(portfolio.trading_costs)
    print(portfolio.cash)
    print(portfolio.flow)
    print(portfolio.borrow_rate)
    print(portfolio.risk_free_rate)

    # The portfolio object is exposing to numerous analytics via quantstats
    portfolio.html(output="report.html")
    portfolio.snapshot()
    portfolio.nav.plot()
