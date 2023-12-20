import numpy as np
import pandas as pd
import pytest

from cvx.simulator._state.state import State


@pytest.fixture()
def state(prices):
    state = State()
    state.prices = prices.iloc[0]
    return state


def test_assets_partial(prices):
    state = State()
    state.prices = prices[["A", "B", "C"]].iloc[0]
    pd.testing.assert_index_equal(state.assets, pd.Index(["A", "B", "C"]))


def test_assets_full(prices):
    state = State()
    state.prices = prices.iloc[0]
    pd.testing.assert_index_equal(state.assets, prices.columns)


def test_trade_no_init_pos(state):
    x = pd.Series({"B": 25.0, "C": -15.0, "D": 40.0}).sub(state.position, fill_value=0)
    pd.testing.assert_series_equal(x, pd.Series({"B": 25.0, "C": -15.0, "D": 40.0}))


def test_gap(prices):
    state = State()
    state.prices = prices.iloc[0]
    assert state.days == 0
    state.time = prices.index[0]
    assert state.days == 0
    state.time = prices.index[1]
    assert state.days == 1


def test_set_position(state):
    state.position = pd.Series({"B": 25.0, "C": -15.0, "D": 40.0})
    pd.testing.assert_series_equal(
        state.position.dropna(), pd.Series({"B": 25.0, "C": -15.0, "D": 40.0})
    )


def test_value(state):
    state.position = pd.Series({"B": 25.0, "C": -15.0, "D": 40.0})
    assert state.value == pytest.approx(-206047.2)


def test_mask():
    state = State()
    state.prices = pd.Series({"A": 1.0, "B": 2.0, "C": np.NaN})
    np.testing.assert_array_equal(state.mask, np.array([True, True, False]))


def test_init(prices):
    state = State()
    state.aum = 1e4

    assert state.cash == 1e4
    assert state.aum == 1e4
    assert state.nav == 1e4
    assert state.value == 0.0
    assert state.profit == 0.0
    assert state.gmv == 0.0
    assert state.leverage == 0.0

    state.prices = prices.iloc[0]

    assert state.cash == 1e4
    assert state.aum == 1e4
    assert state.nav == 1e4
    assert state.value == 0.0
    assert state.profit == 0.0
    assert state.gmv == 0.0
    assert state.leverage == 0.0

    # update position and weights
    state.position = np.ones(len(state.assets))

    assert state.cash == 1e4 - prices.iloc[0].sum()
    assert state.aum == 1e4
    assert state.nav == 1e4
    assert state.value == prices.iloc[0].sum()
    assert state.profit == 0.0

    pd.testing.assert_series_equal(
        state.weights, prices.iloc[0].div(state.nav), check_names=False
    )

    assert state.leverage == pytest.approx(state.weights.abs().sum())

    # update prices
    state.prices = prices.iloc[1]
    assert state.profit == 13.119999999995343
    assert state.cash == 1e4 - prices.iloc[0].sum()
    assert state.aum == 1e4 + state.profit
    assert state.nav == 1e4 + state.profit
    assert state.value == prices.iloc[1].sum()

    # update cash
    state.cash = 1.01 * state.cash
    assert state.cash == 1.01 * (1e4 - prices.iloc[0].sum())
