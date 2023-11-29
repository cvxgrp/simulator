"""
testing the builder
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cvx.simulator.builder import builder as _builder
from cvx.simulator.trading_costs import LinearCostModel


@pytest.fixture()
def builder(prices):
    """
    Fixture for the builder
    :param prices: the prices frame (fixture)
    """
    return _builder(prices=prices)


@pytest.fixture()
def builder_weights(prices):
    """
    Fixture for the builder with 1/n weights
    :param prices: the prices frame (fixture)
    """
    weights = pd.DataFrame(index=prices.index, columns=prices.columns, data=1.0 / 7)
    return _builder(prices, weights=weights)


def test_prices(builder, prices):
    """
    Test that the prices of the builder are the same as the prices
    :param builder: the builder object (fixture)
    :param prices: the prices frame (fixture)
    """
    pd.testing.assert_frame_equal(builder.prices, prices.ffill())


def test_initial_cash(builder):
    """
    Test that the initial cash is 1e6
    :param builder: the builder object (fixture)
    """
    assert builder.initial_cash == 1e6


def test_index(builder, prices):
    """
    Test that the index of the builder is the same as the index of the prices
    :param builder: the builder object (fixture)
    :param prices: the prices frame (fixture)
    """
    assert set(builder.index) == set(prices.index)


def test_trading_cost_model_is_none(builder):
    """
    Test that the trading cost model is None
    :param builder: the builder object (fixture)
    """
    assert builder.trading_cost_model is None


def test_stocks(builder):
    """
    Test that the stocks are all zero
    :param builder: the builder object (fixture)
    :param prices: the prices frame (fixture)
    """
    pd.testing.assert_frame_equal(builder.stocks, np.NaN * builder.prices)


def test_build_empty(builder, prices):
    """
    Test that the portfolio is still empty
    :param builder: the builder object (fixture)
    :param prices: the prices frame (fixture)
    """
    portfolio = builder.build()
    pd.testing.assert_frame_equal(portfolio.prices, prices.ffill())
    pd.testing.assert_frame_equal(portfolio.stocks, np.NaN * prices.ffill())
    pd.testing.assert_series_equal(
        portfolio.profit, pd.Series(index=prices.index[1:], data=0.0)
    )
    pd.testing.assert_series_equal(
        portfolio.nav, pd.Series(index=prices.index, data=1e6)
    )


def test_iteration(builder):
    """
    iterate through builder and verify that the yielded times match the index
    :param builder: the builder object (fixture)
    """
    assert {t[-1] for t, _ in builder} == set(builder.index)


def test_iteration_state(builder):
    """
    iterate through builder and verify that the state is correct
    :param builder: the builder object (fixture)
    """
    for t, state in builder:
        assert state.leverage == 0
        assert state.nav == 1e6
        assert state.value == 0.0
        pd.testing.assert_series_equal(
            state.weights, pd.Series(index=state.assets, data=0.0), check_names=False
        )
        pd.testing.assert_series_equal(
            builder[t[-1]],
            pd.Series(index=state.assets, data=np.NaN),
            check_names=False,
        )


def test_build(builder_weights):
    """
    Test that the portfolio is built correctly
    :param builder_weights: the builder with 1/n weights (fixture)
    """
    # build the portfolio directly
    portfolio = builder_weights.build()

    # loop and set the weights explicitly
    for t, state in builder_weights:
        builder_weights.set_weights(
            time=t[-1], weights=pd.Series(index=state.assets, data=1.0 / 7.0)
        )

    # build again
    portfolio2 = builder_weights.build()

    # verify both methods give the same result
    pd.testing.assert_series_equal(portfolio.nav, portfolio2.nav)


def test_set_weights(prices):
    """
    Test that the weights are set correctly
    :param prices: the prices frame (fixture)
    """
    b = _builder(prices=prices[["B", "C"]].head(5), initial_cash=50000)
    for times, state in b:
        b.set_weights(time=times[-1], weights=pd.Series(index=["B", "C"], data=0.5))

    portfolio = b.build()
    assert portfolio.nav.values[-1] == pytest.approx(49773.093729)


def test_set_cashpositions(prices):
    """
    Test that the cashpositions are set correctly
    :param prices: the prices frame (fixture)
    """
    b = _builder(prices=prices[["B", "C"]].head(5), initial_cash=50000)
    for times, state in b:
        b.set_cashposition(
            time=times[-1], cashposition=pd.Series(index=["B", "C"], data=state.nav / 2)
        )

    portfolio = b.build()
    assert portfolio.nav.values[-1] == pytest.approx(49773.093729)


def test_set_position(prices):
    b = _builder(prices=prices[["B", "C"]].head(5), initial_cash=50000)
    for times, state in b:
        b.set_position(
            time=times[-1],
            position=pd.Series(index=["B", "C"], data=state.nav / (state.prices * 2)),
        )
    portfolio = b.build()
    assert portfolio.nav.values[-1] == pytest.approx(49773.093729)


def test_with_costmodel(prices):
    b = _builder(
        prices=prices[["B", "C"]].head(5),
        initial_cash=50000,
        trading_cost_model=LinearCostModel(factor=0.0010),
    )

    for times, state in b:
        b.set_position(
            time=times[-1],
            position=pd.Series(index=["B", "C"], data=state.nav / (state.prices * 2)),
        )

    portfolio = b.build()
    assert portfolio.nav.values[-1] == pytest.approx(49722.58492364325)


def test_input_data(prices):
    b = _builder(prices=prices, initial_cash=50000, volume=prices.ffill())
    for t, state in b:
        print(state.volume)
        pd.testing.assert_series_equal(state.prices, state.input_data["volume"])


def test_weights_on_wrong_days(resource_dir):
    prices = pd.read_csv(
        resource_dir / "priceNaN.csv", index_col=0, parse_dates=True, header=0
    )

    b = _builder(prices=prices, initial_cash=50000)
    t = prices.index

    for t, state in b:
        with pytest.raises(ValueError):
            b.set_weights(
                t[-1], pd.Series(index={"A", "B", "C"}, data=[0.5, 0.25, 0.25])
            )

        with pytest.raises(ValueError):
            b.set_cashposition(t[-1], pd.Series(index={"A", "B", "C"}, data=[5, 5, 5]))

        with pytest.raises(ValueError):
            b.set_position(t[-1], pd.Series(index={"A", "B", "C"}, data=[5, 5, 5]))

        with pytest.raises(ValueError):
            b[t[-1]] = pd.Series(index={"A", "B", "C"}, data=[5, 5, 5])

    for t, state in b:
        b.set_weights(
            t[-1],
            pd.Series(
                index=prices.loc[t[-1]].dropna().index,
                data=np.random.rand(6),
                dtype=float,
            ),
        )
