import pandas as pd
import pytest

from cvx.simulator.builder import builder as _builder


@pytest.fixture()
def builder(prices):
    return _builder(prices=prices)


@pytest.fixture()
def builder_weights(prices):
    weights = pd.DataFrame(index=prices.index, columns=prices.columns, data=1.0/7)
    return _builder(prices, weights=weights)


def test_prices(builder, prices):
    pd.testing.assert_frame_equal(builder.prices, prices.ffill())


def test_initial_cash(builder):
    assert builder.initial_cash == 1e6


def test_assets(builder, prices):
    assert set(builder.assets) == set(prices.columns)


def test_index(builder, prices):
    assert set(builder.index) == set(prices.index)


def test_trading_cost_model_is_none(builder):
    assert builder.trading_cost_model is None


def test_stocks(builder, prices):
    pd.testing.assert_frame_equal(builder.stocks, 0.0*prices.ffill())


def test_build_empty(builder, prices):
    portfolio = builder.build()
    pd.testing.assert_frame_equal(portfolio.prices, prices.ffill())
    pd.testing.assert_frame_equal(portfolio.stocks, 0.0*prices.ffill())
    pd.testing.assert_series_equal(portfolio.profit, pd.Series(index=prices.index[1:], data=0.0))
    pd.testing.assert_series_equal(portfolio.nav, pd.Series(index=prices.index, data=1e6))


def test_iteration(builder):
    assert {t[-1] for t, _ in builder} == set(builder.index)


def test_iteration_state(builder):
    for t, state in builder:
        assert state.leverage == 0
        assert state.nav == 1e6
        assert state.value == 0.0
        pd.testing.assert_series_equal(state.weights, pd.Series(index=builder.assets, data=0.0), check_names=False)
        pd.testing.assert_series_equal(builder[t[-1]], pd.Series(index=builder.assets, data=0.0), check_names=False)


def test_build(builder_weights, prices):
    # build the portfolio directly
    portfolio = builder_weights.build()

    # loop and set the weights explicitly
    for t, state in builder_weights:
        builder_weights.set_weights(time=t[-1], weights=pd.Series(index=builder_weights.assets, data=1.0/7.0))

    # build again
    portfolio2 = builder_weights.build()

    # verify both methods give the same result
    pd.testing.assert_series_equal(portfolio.nav, portfolio2.nav)


def test_set_weights(prices):
    b = _builder(prices=prices[["B", "C"]].head(5), initial_cash=50000)
    for times, state in b:
        b.set_weights(time=times[-1], weights=pd.Series(index=["B","C"], data=0.5))

    portfolio = b.build()
    assert portfolio.nav.values[-1] == pytest.approx(49773.093729)


def test_set_cashpositions(prices):
    b = _builder(prices=prices[["B", "C"]].head(5), initial_cash=50000)
    for times, state in b:
        b.set_cashposition(time=times[-1], cashposition=pd.Series(index=["B", "C"], data=state.nav / 2))

    portfolio = b.build()
    assert portfolio.nav.values[-1] == pytest.approx(49773.093729)
