# -*- coding: utf-8 -*-
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


def test_assets(builder, prices):
    """
    Test that the assets of the builder are the same as the columns of the prices
    :param builder: the builder object (fixture)
    :param prices: the prices frame (fixture)
    """
    assert set(builder.assets) == set(prices.columns)


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
    pd.testing.assert_frame_equal(builder.stocks, 0.0 * builder.prices)


def test_build_empty(builder, prices):
    """
    Test that the portfolio is still empty
    :param builder: the builder object (fixture)
    :param prices: the prices frame (fixture)
    """
    portfolio = builder.build()
    pd.testing.assert_frame_equal(portfolio.prices, prices.ffill())
    pd.testing.assert_frame_equal(portfolio.stocks, 0.0 * prices.ffill())
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
            state.weights, pd.Series(index=builder.assets, data=0.0), check_names=False
        )
        pd.testing.assert_series_equal(
            builder[t[-1]], pd.Series(index=builder.assets, data=0.0), check_names=False
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
            time=t[-1], weights=pd.Series(index=builder_weights.assets, data=1.0 / 7.0)
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


def test_box(resource_dir):
    """
    Test that the box constraints are working
    """
    # load the price data
    prices = pd.read_csv(
        resource_dir / "prices.csv", index_col=0, header=0, parse_dates=True
    )

    # load the market capitalization
    market_cap = pd.read_csv(
        resource_dir / "market_cap.csv", index_col=0, header=0, parse_dates=True
    )

    # load the trade volume
    volume = pd.read_csv(
        resource_dir / "volume.csv", index_col=0, header=0, parse_dates=True
    )

    # load the target weights
    weights = pd.read_csv(
        resource_dir / "target_weights.csv", index_col=0, header=0, parse_dates=True
    )

    # define the builder
    # we can hold max 6% of the market cap of a stock
    # we can trade max 20% of the volume of a stock
    builder = _builder(
        prices=prices,
        initial_cash=1e6,
        market_cap=market_cap,
        trade_volume=volume,
        weights=weights,
        max_cap_fraction=0.06,
        min_cap_fraction=-0.03,
        max_trade_fraction=0.2,
        min_trade_fraction=-0.2,
    )

    # build the portfolio
    portfolio = builder.build()

    # the weights of the portfolio should match my manual offline calculation
    pd.testing.assert_frame_equal(
        portfolio.weights,
        pd.DataFrame(
            index=prices.index,
            columns=prices.columns,
            data=[[0.2, 0.4], [0.4, 0.8], [0.6, 1.2], [0.6, 1.2]],
        ),
    )


def test_returns(prices):
    """
    Test that the returns are calculated correctly
    :param prices: the prices frame (fixture)
    """
    b = _builder(prices=prices, initial_cash=50000)
    pd.testing.assert_frame_equal(b.returns, prices.pct_change().dropna())


def test_cov(prices):
    b = _builder(prices=prices, initial_cash=50000)
    for time, mat in b.cov(min_periods=50, com=50):
        # print(time)
        # print(mat)
        assert np.all(np.isfinite(mat))
