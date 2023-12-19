import numpy as np
import pandas as pd
import pytest

from cvx.simulator.equity.state import EquityState as State


@pytest.fixture()
def state(prices):
    state = State()
    state.prices = prices.iloc[0]
    return state


def test_trade(prices):
    state = State()
    state.prices = prices[["A", "B", "C"]].iloc[0]
    state.position = np.array([10, 20, -10])

    x = pd.Series({"B": 25, "C": -15, "D": 40}).sub(state.position, fill_value=0)
    pd.testing.assert_series_equal(
        x, pd.Series({"A": -10.0, "B": 5.0, "C": -5.0, "D": 40.0})
    )


def test_trade_no_init_pos(state):
    x = pd.Series({"B": 25.0, "C": -15.0, "D": 40.0}).sub(state.position, fill_value=0)
    pd.testing.assert_series_equal(x, pd.Series({"B": 25.0, "C": -15.0, "D": 40.0}))


def test_cash(state):
    assert state.cash == 1e6


def test_nav(state):
    assert state.nav == 1e6


def test_cash_position(state):
    state.position = pd.Series({"B": 25.0, "C": -15.0, "D": 40.0})
    assert state.cash == 1206047.2


def test_value_position(state):
    state.position = pd.Series({"B": 25.0, "C": -15.0, "D": 40.0})
    assert state.value == -206047.2


def test_nav_position(state):
    state.position = pd.Series({"B": 25.0, "C": -15.0, "D": 40.0})
    assert state.nav == 1e6


def test_gmv_position(state):
    state.position = pd.Series({"B": 25.0, "C": -15.0, "D": 40.0})
    assert state.gmv == 1670455.8


def test_short(state, prices):
    state.position = pd.Series({"B": 25.0, "C": -15.0, "D": 40.0})
    assert state.short == -938251.5
    assert state.short == -15.0 * prices["C"].iloc[0]


def test_state():
    prices = pd.Series(data=[2.0, 3.0])
    positions = pd.Series(data=[100, 300])
    cash = 400
    state = State(cash=cash)
    state.prices = prices
    state.position = positions
    # value is the money in units
    assert state.value == 1100.0
    assert state.cash == -700.0
    # nav is the value plus the cash
    assert state.nav == 400.0
    # weights are the positions divided by the value
    pd.testing.assert_series_equal(
        state.weights, pd.Series(data=[2.0 / 4.0, 9.0 / 4.0])
    )
    # leverage is the value divided by the nav
    assert state.leverage == 11.0 / 4.0
